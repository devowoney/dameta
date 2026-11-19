"""
delta = UNSUPERVISED delta learned optimizer (2026-11-18, pilot; variant `delta`, delta.output = free).

The existing Meta-B delta (meta_b.py: previous-step evaluation with the chain rule, inner rollout, ALIGN with the
teacher) with two changes:
  1. the module outputs the update itself, Delta_hat = M_theta(x^k, grad J(x^k), k) (same network as delta_exact,
     meta_b_direct.DirectDelta, with the K-sub-step inner rollout), so INFERENCE NEEDS NO TEACHER:
         x^{k+1} = x^k + Delta_hat                                         (run_direct, 1 gradient / iteration)
  2. no replay / double backward: the graph of the move actually made, Delta_hat^{k-1}(theta), is stored; at
     iteration k the gradient g^k = grad J(x^k) (needed for the next move anyway) gives the exact chain rule
         d J(x^k) / d theta = g^k . d Delta_hat^{k-1} / d theta
     -> ONE meta step per outer iteration on
         l_meta = w_perf l_perf + w_align l_align + l_reg
         l_perf  = < g^k, Delta_hat^{k-1}(theta) > / < g^{k-1}, f^{k-1} >     (~ -1 for a teacher-sized move;
                                                                             theta-gradient = dJ(x^k)/dtheta / denom)
         l_align = | Delta_hat_theta(x^k) - Delta^k |^2 / | Delta^k |^2,    Delta^k = -f^k teacher update at x^k
         l_reg   = lambda_reg mean(theta^2)
     The first iteration has no previous move: align (+ reg) only. The teacher (ReferenceStep, same lr as classical)
     runs on the learner's own states during TRAINING only (alignment target and the l_perf denominator).
Order per iteration: g^k -> f^k -> meta step (l_perf through the stored graph, built with the current theta) ->
Delta_hat^k with the updated theta (graph stored) -> x^{k+1}.
Magnitude init l0 from one teacher run on the train cases (as delta_exact); best epoch by the frozen val score
J(x^{select_k}) over val cases (no teacher; default 90th percentile, diverged = inf).
"""

import copy
import time
from dataclasses import dataclass
from typing import Optional

import torch

from ..da.cost import FourDVarCost
from ..da.reference_step import ReferenceStep
from .meta_b_direct import DirectDelta, run_direct

TINY = torch.finfo(torch.float64).tiny


@dataclass
class FreeDeltaConfig:
    w_perf_delta: float = 2.5
    w_align: float = 1.0
    lambda_reg: float = 1e-4
    meta_lr: float = 1e-3
    grad_clip_norm: float = 1.0
    n_epochs: int = 1
    batch: int = 100
    episode_iters: int = 100
    train_seed: int = 0
    select_epoch: bool = False
    select_k: int = 10
    select_stat: str = "q90"         # val score over cases of J(x^select_k): q90 (90th percentile, diverged = inf;
                                     # the median hides a diverging minority) | median


def free_delta_episode(learner: DirectDelta, cost: FourDVarCost, c0: torch.Tensor, method: str, lr: float,
                       n_iter: int, opt: Optional[torch.optim.Optimizer], cfg: FreeDeltaConfig) -> dict:
    """One online training episode on a batch of cases (opt=None: no theta update)."""
    assert learner.cfg.coords == "raw" and learner.cfg.obs_feat == "y" and not learner.cfg.hist_m, \
        "unsupervised free delta: coords cvt / innovation features / history not implemented yet (delta_exact only)"
    net = learner.net
    params = list(net.parameters())
    n_params = sum(p.numel() for p in params)
    c = c0.detach().clone()
    obs = learner.obs_feats(cost)
    teacher = ReferenceStep(c, method=method, lr=lr, cost=cost if method == "lbfgs_wolfe" else None)
    ok = torch.ones(c.shape[0], dtype=torch.bool)
    prev, diag = None, []
    net.train()
    for k in range(n_iter):
        j, g = cost.grad(c)
        ok = ok & torch.isfinite(j) & torch.isfinite(g).all(1)
        if not ok.any():
            break
        with torch.no_grad():
            f = teacher(c, g, ok.to(c.dtype), j=j)
        d = {"k": k, "J_med": float(j[ok].median())}
        if opt is not None:
            opt.zero_grad()
            l_reg = cfg.lambda_reg * sum(p.pow(2).sum() for p in params) / n_params
            l_meta = l_reg
            tgt = -f
            den_a = tgt.pow(2).sum(1)
            a_ok = ok & torch.isfinite(den_a) & (den_a > 0)
            if a_ok.any():
                o = obs[a_ok] if obs is not None else None
                l_align = ((learner.delta(c[a_ok], g[a_ok], k, o) - tgt[a_ok]).pow(2).sum(1) / den_a[a_ok]).mean()
                l_meta = l_meta + cfg.w_align * l_align
                d["L_align"] = float(l_align.detach())
            if prev is not None:
                p_ok = ok & prev["ok"]
                if p_ok.any():
                    l_perf = ((g[p_ok] * prev["D"][p_ok]).sum(1) / prev["den"][p_ok]).mean()
                    l_meta = l_meta + cfg.w_perf_delta * l_perf
                    d["L_perf"] = float(l_perf.detach())
            if torch.isfinite(l_meta):
                l_meta.backward()
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip_norm)
                if all(torch.isfinite(p.grad).all() for p in params if p.grad is not None):
                    opt.step()
        D = learner.delta(c, g, k, obs)                              # move with the updated theta; graph kept
        D_ok = ok & torch.isfinite(D).all(1)
        prev = {"D": D, "ok": D_ok, "den": (g * f).sum(1).clamp_min(TINY)}
        with torch.no_grad():
            step = torch.where(D_ok[:, None], D.detach(), torch.zeros_like(c))
            d["step_ratio_med"] = float((step.norm(dim=1) / f.norm(dim=1).clamp_min(TINY))[ok].median())
        c = c + step
        diag.append(d)
    net.eval()
    return {"c": c, "diag": diag}


def train_free_delta(learner: DirectDelta, make_cost, x0_b: torch.Tensor, method: str, lr: float,
                     cfg: FreeDeltaConfig, cost_val: Optional[FourDVarCost] = None,
                     x0_val: Optional[torch.Tensor] = None, log=None) -> list:
    """Episodes over batches of train cases (theta + Adam state persist); optional best epoch on val (frozen,
    learner only, median J after select_k iterations)."""
    d = learner.net.d
    pack = lambda xb: torch.cat([xb, torch.zeros(xb.shape[0], d - 3, dtype=xb.dtype)], 1)  # noqa: E731
    opt = torch.optim.Adam(learner.net.parameters(), lr=cfg.meta_lr)
    gen = torch.Generator().manual_seed(cfg.train_seed)
    n, hist = x0_b.shape[0], []
    best = (float("inf"), copy.deepcopy(learner.net.state_dict()), -1)

    def val_score():
        r = run_direct(cost_val, x0_val, learner, max_iter=cfg.select_k, tol=0.0)
        jk = torch.tensor(r["J_iter"][:, -1])
        jk = torch.where(torch.isfinite(jk), jk, torch.full_like(jk, 1e30))     # diverged case = +inf
        return float(jk.median() if cfg.select_stat == "median" else torch.quantile(jk, 0.9))

    if cfg.select_epoch:
        best = (val_score(), best[1], -1)
        hist.append({"epoch": -1, "val_score": best[0]})
        if log is not None:
            log(hist[-1])
    for ep in range(cfg.n_epochs):
        perm = torch.randperm(n, generator=gen)
        for b0 in range(0, n, cfg.batch):
            idx = perm[b0:b0 + cfg.batch].sort().values
            t0 = time.perf_counter()
            r = free_delta_episode(learner, make_cost(idx.numpy()), pack(x0_b[idx]), method, lr, cfg.episode_iters,
                                   opt, cfg)
            dg = r["diag"]
            h = {"epoch": ep, "batch": b0 // cfg.batch, "J0_med": dg[0]["J_med"], "J_end_med": dg[-1]["J_med"],
                 "L_perf_mean": float(torch.tensor([x["L_perf"] for x in dg if "L_perf" in x]).mean()),
                 "L_align_mean": float(torch.tensor([x["L_align"] for x in dg if "L_align" in x]).mean()),
                 "step_ratio_med": float(torch.tensor([x["step_ratio_med"] for x in dg]).nanmedian()),
                 "s": time.perf_counter() - t0}
            hist.append(h)
            if log is not None:
                log(h)
        if cfg.select_epoch:
            score = val_score()
            if score < best[0]:
                best = (score, copy.deepcopy(learner.net.state_dict()), ep)
            hist.append({"epoch": ep, "val_score": score, "best_epoch": best[2]})
            if log is not None:
                log(hist[-1])
    if cfg.select_epoch:
        learner.net.load_state_dict(best[1])
    learner.net.eval()
    return hist
