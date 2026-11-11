"""
Meta-B — teacher-relative learned update rule (bice "delta inner loss" mechanism), ported to L63 4D-Var.

Source (read-only): bice branch `meta-learner-one-task`, src/gd_optimic/meta_learner/bilevel_optimizer.py
(`one_task_update`, `delta_chain`, `_delta_inner_step`). See memory/findings.md "Reference implementation".

Outer iteration k (one forward + adjoint run = one gradient evaluation, the cost unit of the study):
    g = dJ/dc(c^k)                       (FourDVarCost.grad)
    f = teacher step f(c^k)              (ReferenceStep: GD lr*g, or Adagrad with its own state)
    chain, K applications of S, NO dynamics call:
        c_{m+1} = c_m - (f / K) * max(1 + s_theta(c_m, g, k), 0),   m = 0..K-1,   c^{k+1} = c_K
With s = 0 the chain is exactly one teacher step; S can rescale each component of f by a factor in
[0, 1 + output_scale] but never flip it (a learned, state-dependent diagonal preconditioner relative to the teacher).

Meta update (train mode, online, num_meta_steps Adam steps on theta per outer iteration, before the IC move):
    phase ALIGN (m < grad_align_steps, and always at the first outer iteration):  w_align l_align + l_reg
    phase TRANS (next grad_trans_steps):                                      w_align l_align + w_perf l_perf + l_reg
    phase PERF  (remaining):                                                  w_perf l_perf + l_reg
    l_align = mean(s^2) at c^k                         (stay at the teacher step = imitation)
    l_perf  = mean_cases[ -<g^k, c^{k-1} - c_K(theta)> / <g^{k-1}, f^{k-1}> ]   (Delta inner loss)
              c_K(theta) replays the previous chain from c^{k-1} with the current theta; first-order model of
              J(c^k) - J(c^{k-1}), whose theta-gradient equals dJ(c_K(theta))/dtheta / denom exactly (chain rule).
              ~ -1 when the move equals one teacher step, < -1 when the stretched move does better.

Changes vs bice (L63 port; rows in memory/branches/m4-meta-b/decisions.md):
    - batch of independent cases: l_perf normalised PER CASE then averaged; l_align averaged over cases
    - gradient input g_std = g / RMS(g) per case and per control block (x0 | eta) instead of per-channel std over
      the spatial field (L63 has 3 components only; per-case keeps cases independent in frozen inference)
    - state input x_std = (c - climatological mean) / std (x0 block: train-truth statistics; eta block: 0 / sigma_q)
    - no ocean mask, no activation checkpointing (d <= 33), no forward-model call for reporting (the next outer
      gradient evaluation returns J(c^{k+1}) anyway)
    - l_reg = lambda * (lambda * sum(theta^2) / n_theta) as in bice (lambda applied twice) behind flag
      `reg_square_lambda` (default True = faithful port; pilot to confirm intent)
"""

import copy
import time
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import numpy as np
import torch

from ..da.cost import FourDVarCost
from ..da.reference_step import ReferenceStep
from .networks import StretchNet

TINY = torch.finfo(torch.float64).tiny


@dataclass
class MetaBConfig:
    """Meta-B hyper-parameters (defaults = bice `meta_learner` defaults in delta mode)."""

    K: int = 20                     # chain length (bice delta_inner_k)
    num_meta_steps: int = 10        # Adam steps on theta per outer iteration
    grad_align_steps: int = 6       # ALIGN meta-steps
    grad_trans_steps: int = 2       # TRANS meta-steps (then PERF)
    w_align: float = 1.0
    w_perf_delta: float = 2.5
    lambda_reg: float = 1e-4
    reg_square_lambda: bool = True  # bice: lambda applied twice
    meta_lr: float = 1e-3
    grad_clip_norm: float = 1.0


def control_stats(x_true_train: torch.Tensor, n_obs: int, sigma_q: float, weak: bool):
    """Per-component mean / std of the control used to standardise S's state input. x_true_train [n, T, 3]."""
    m, s = x_true_train.reshape(-1, 3).mean(0), x_true_train.reshape(-1, 3).std(0)
    if not weak:
        return m, s
    ne = 3 * (n_obs - 1)
    return torch.cat([m, torch.zeros(ne, dtype=m.dtype)]), torch.cat([s, torch.full((ne,), sigma_q, dtype=m.dtype)])


class MetaB:
    """Holds S_theta, its meta-optimizer and the previous move (for l_perf). One instance per dynamics x constraint."""

    def __init__(self, net: StretchNet, cfg: MetaBConfig, ctrl_mean: torch.Tensor, ctrl_std: torch.Tensor):
        self.net, self.cfg = net, cfg
        self.ctrl_mean, self.ctrl_std = ctrl_mean, ctrl_std
        self.blocks = [(0, 3)] + ([(3, net.d)] if net.d > 3 else [])      # x0 | eta
        self.opt = torch.optim.Adam(net.parameters(), lr=cfg.meta_lr, weight_decay=0.0)  # bice delta mode: no WD
        self._prev = None

    def reset(self):
        """Start a new episode (new batch of cases): forget the previous move."""
        self._prev = None

    # ---- S and the chain ----------------------------------------------------------------------------------------
    def standardize_grad(self, g: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(g)
        for a, b in self.blocks:
            blk = g[:, a:b]
            out[:, a:b] = blk / blk.pow(2).mean(1, keepdim=True).sqrt().clamp_min(TINY)
        return out

    def stretch(self, c: torch.Tensor, g_std: torch.Tensor, k: int) -> torch.Tensor:
        return self.net((c - self.ctrl_mean) / self.ctrl_std, g_std, k)

    def chain(self, c: torch.Tensor, g_std: torch.Tensor, f: torch.Tensor, k: int) -> torch.Tensor:
        """c_K after K stretched sub-steps of f (gradient and f held fixed). Differentiable in theta."""
        for _ in range(self.cfg.K):
            c = c - f / self.cfg.K * (1.0 + self.stretch(c, g_std, k)).clamp_min(0.0)
        return c

    # ---- online meta update (Delta inner loss) ------------------------------------------------------------------
    def meta_update(self, c: torch.Tensor, g: torch.Tensor, f: torch.Tensor, k: int) -> dict:
        cfg, prev = self.cfg, self._prev
        c, g, f = c.detach(), g.detach(), f.detach()
        g_std = self.standardize_grad(g)
        params = list(self.net.parameters())
        n_params = sum(p.numel() for p in params)
        diag = {"k": k, "L_perf_applied": float("nan"), "rho": float("nan")}
        if prev is not None:   # diagnostics of the move actually applied last iteration (exact, theta-independent)
            with torch.no_grad():
                step = prev["step"]
                diag["L_perf_applied"] = float((-(g * step).sum(1) / prev["denom"]).mean())
                rho = (g * step).sum(1) / (prev["g"] * step).sum(1)
                diag["rho"] = float(rho[torch.isfinite(rho)].median()) if torch.isfinite(rho).any() else float("nan")

        self.net.train()
        for m in range(cfg.num_meta_steps):
            if prev is None or m < cfg.grad_align_steps:
                phase = "ALIGN"
            elif m < cfg.grad_align_steps + cfg.grad_trans_steps:
                phase = "TRANS"
            else:
                phase = "PERF"
            self.opt.zero_grad()
            l_align = self.stretch(c, g_std, k).pow(2).mean()
            sq = sum(p.pow(2).sum() for p in params) / n_params
            l_reg = cfg.lambda_reg * (cfg.lambda_reg * sq if cfg.reg_square_lambda else sq)
            l_meta = l_reg + (cfg.w_align * l_align if phase in ("ALIGN", "TRANS") else 0.0)
            if phase in ("TRANS", "PERF"):
                total = prev["c"] - self.chain(prev["c"], prev["g_std"], prev["f"], prev["k"])
                l_perf = (-(g * total).sum(1) / prev["denom"]).mean()
                l_meta = l_meta + cfg.w_perf_delta * l_perf
                diag["L_perf"] = float(l_perf.detach())
            l_meta.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=cfg.grad_clip_norm)
            self.opt.step()
            diag["L_align"], diag["L_reg"] = float(l_align.detach()), float(l_reg.detach())
        self.net.eval()
        return diag

    @torch.no_grad()
    def advance(self, c: torch.Tensor, g: torch.Tensor, f: torch.Tensor, k: int, record: bool = False) -> torch.Tensor:
        """Apply the chain (theta frozen). With record=True, remember the move for the next l_perf."""
        g_std = self.standardize_grad(g)
        c_new = self.chain(c, g_std, f, k)
        if record:
            self._prev = {"c": c.detach().clone(), "g": g.detach().clone(), "g_std": g_std, "f": f.detach().clone(),
                          "k": k, "step": (c - c_new).detach(), "denom": (g * f).sum(1).clamp_min(TINY)}
        return c_new

    # ---- checkpoint ---------------------------------------------------------------------------------------------
    def state_dict(self) -> dict:
        n = self.net
        return {"net": n.state_dict(), "cfg": asdict(self.cfg), "ctrl_mean": self.ctrl_mean, "ctrl_std": self.ctrl_std,
                "net_kwargs": dict(d=n.d, hidden=n.head.in_features, n_layers=sum(isinstance(l, torch.nn.Linear)
                                   for l in n.body), emb_dim=n.emb.net[2].out_features, output_scale=n.output_scale,
                                   log_k=n.emb.log_input)}

    @classmethod
    def from_state_dict(cls, sd: dict) -> "MetaB":
        net = StretchNet(**sd["net_kwargs"], zero_init=False).to(sd["ctrl_mean"].dtype)
        net.load_state_dict(sd["net"])
        net.eval()
        return cls(net, MetaBConfig(**sd["cfg"]), sd["ctrl_mean"], sd["ctrl_std"])


def run_meta_b(cost: FourDVarCost, x0_init: torch.Tensor, learner: MetaB, method: str, lr: float,
               max_iter: int = 1000, tol: float = 1e-6, eta_init: Optional[torch.Tensor] = None,
               train: bool = False) -> dict:
    """
    Meta-B outer loop; same per-case stopping rule, cost accounting and output record as `run_fourdvar`, plus
        step_ratio [n, K]  |c^k - c^{k+1}| / |f(c^k)|   (1 = teacher step size)
        meta_diag  list of per-iteration meta-update diagnostics (train mode only)
    train=True: online meta-update of theta before every move (pass tol=0 to run the full episode).
    train=False: frozen inference (theta untouched).
    """
    n = x0_init.shape[0]
    if cost.weak and eta_init is None:
        eta_init = torch.zeros(n, cost.cfg.n_obs - 1, 3, dtype=x0_init.dtype)
    c = cost.pack(x0_init, eta_init).detach().clone()
    teacher = ReferenceStep(c, method=method, lr=lr)
    learner.reset()
    if not train:
        learner.net.eval()

    active = torch.ones(n, dtype=c.dtype)
    n_evals = torch.zeros(n, dtype=torch.long)
    x0_hist, j_hist, cos_hist, ratio_hist, meta_diag = [c[:, :3].clone()], [], [], [], []
    t0 = time.perf_counter()
    j_prev = None
    for k in range(max_iter):
        j, g = cost.grad(c)
        n_evals += active.long()
        j_hist.append(j.clone() if j_prev is None else torch.where(active.bool(), j, j_hist[-1]))
        if j_prev is not None:
            rel = (j_prev - j).abs() / j_prev.abs().clamp_min(1e-300)
            newly_done = active.bool() & ((rel < tol) | ~torch.isfinite(j))
            active = active * (~newly_done).to(c.dtype)
        if active.sum() == 0:
            break
        f = teacher(c, g, active)
        if train:
            meta_diag.append(learner.meta_update(c, g, f, k))
        c_new = learner.advance(c, g, f, k, record=train)
        step = c - c_new
        cos_hist.append(torch.nn.functional.cosine_similarity(step, g, dim=1))
        ratio = step.norm(dim=1) / f.norm(dim=1).clamp_min(TINY)
        ratio_hist.append(torch.where(active.bool(), ratio, torch.full_like(ratio, float("nan"))))
        c = c_new
        x0_hist.append(c[:, :3].clone())
        j_prev = j
    wall = time.perf_counter() - t0

    with torch.no_grad():
        j_final = cost(*cost.unpack(c))
    j_hist.append(j_final)
    k_run = len(x0_hist) - 1
    pad = max_iter - k_run
    x0_iter = torch.stack(x0_hist, 1)
    x0_iter = torch.cat([x0_iter, x0_iter[:, -1:].expand(n, pad, 3)], 1) if pad > 0 else x0_iter
    j_iter = torch.stack(j_hist[: k_run + 1], 1)
    j_iter = torch.cat([j_iter, j_iter[:, -1:].expand(n, pad)], 1) if pad > 0 else j_iter

    def _pad(h):
        h = torch.stack(h, 1) if h else torch.zeros(n, 0, dtype=c.dtype)
        return torch.cat([h, torch.full((n, max_iter - h.shape[1]), float("nan"), dtype=c.dtype)], 1).numpy()

    out = {
        "x0_iter": x0_iter.numpy(), "J_iter": j_iter.numpy(), "cos_step_grad": _pad(cos_hist),
        "step_ratio": _pad(ratio_hist), "n_grad_evals": n_evals.numpy(),
        "converged": ((active == 0) & (n_evals < max_iter) & torch.isfinite(j_final)).numpy(),
        "x0_a": c[:, :3].numpy(), "wall_s": wall, "meta_diag": meta_diag,
    }
    if cost.weak:
        out["eta_final"] = cost.unpack(c)[1].numpy()
    return out


def train_meta_b(learner: MetaB, make_cost: Callable[[np.ndarray], FourDVarCost], x0_b: torch.Tensor,
                 method: str, lr: float, n_epochs: int = 1, batch: int = 100, episode_iters: int = 100,
                 seed: int = 0, log: Optional[Callable[[dict], None]] = None) -> list:
    """
    Meta-train across training ICs (main protocol, frozen inference afterwards): each episode = one batch of train
    cases optimised for `episode_iters` outer iterations with online Delta inner-loss updates (theta and its Adam
    state persist across episodes). make_cost(idx) builds the 4D-Var cost of cases idx. Returns episode summaries.
    """
    rng = np.random.default_rng(seed)
    n, history = x0_b.shape[0], []
    for epoch in range(n_epochs):
        perm = rng.permutation(n)
        for b0 in range(0, n, batch):
            idx = np.sort(perm[b0:b0 + batch])
            res = run_meta_b(make_cost(idx), x0_b[idx], learner, method, lr, max_iter=episode_iters, tol=0.0,
                             train=True)
            md = res["meta_diag"]
            summ = {"epoch": epoch, "batch": b0 // batch, "J0_med": float(np.median(res["J_iter"][:, 0])),
                    "J_end_med": float(np.median(res["J_iter"][:, -1])),
                    "L_perf_applied_mean": float(np.nanmean([d["L_perf_applied"] for d in md[1:]])),
                    "step_ratio_med": float(np.nanmedian(res["step_ratio"])), "wall_s": res["wall_s"]}
            history.append(summ)
            if log is not None:
                log(summ)
    return history


def clone_learner(learner: MetaB) -> MetaB:
    """Deep copy (frozen snapshot for evaluation while training continues)."""
    return copy.deepcopy(learner)
