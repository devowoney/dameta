"""
M5 cell 2 — learned OPTIMIZER, SUPERVISED, gradient input: the Meta-B stretch chain (same network S_theta(x, grad J,
emb(k)), same K-application chain, same frozen inference `run_meta_b`) trained to reproduce the classical teacher's
iterate m steps ahead (pilot 2026-11-14: m = 5):

    pair (x^k, g^k, f^k, k)  ->  target x^{k+m}   from the classical teacher run (run_fourdvar, same lr / stopping)
    loss = mean_pairs |chain_theta(x^k; g^k, f^k, k) - x^{k+m}|^2 / |x^k - x^{k+m}|^2      (relative, scale-free)
    f^k  = x^k - x^{k+1} (the teacher step actually taken); x^{k+m} beyond the end of the run = final iterate.

One chain step can rescale each teacher-step component by at most (1 + output_scale) and never flip it, so the
lookahead target is only partly reachable when the teacher path curves; the residual is reported (val loss).
Best epoch on val-IC pairs. Inference = one gradient evaluation per iteration, like cell 1.
"""

import copy
import time
from dataclasses import dataclass

import torch

from ..da.cost import FourDVarCost
from ..da.fourdvar import run_fourdvar
from .meta_b import MetaB


@dataclass
class LookaheadConfig:
    m: int = 5
    lr: float = 1e-3
    epochs: int = 30
    batch_size: int = 1024
    seed: int = 0


def collect_lookahead_pairs(cost: FourDVarCost, x0_b: torch.Tensor, method: str, lr: float, m: int = 5,
                            max_iter: int = 1000, tol: float = 1e-6) -> dict:
    """Teacher trajectory of every case -> pairs of active iterations: x, g, f, k, target x^{k+m}."""
    traj, grads, act = [], [], []

    def hook(k, c, g, active):
        traj.append(c.clone())
        grads.append(g.clone())
        act.append(active.bool().clone())

    res = run_fourdvar(cost, x0_b, method, lr, max_iter, tol, on_step=hook)
    eta = torch.tensor(res["eta_final"]) if cost.weak else None
    final = cost.pack(torch.tensor(res["x0_a"]), eta)
    T = torch.stack(traj + [final], 1)                              # [n, k_run + 1, d] (last = final iterate)
    G, A = torch.stack(grads, 1), torch.stack(act, 1)               # [n, k_run, d], [n, k_run]
    n, k_run = A.shape
    # final iterate of a frozen case = its value after the last move; hook values of stopped cases already equal it
    xs, gs, fs, ks, tg = [], [], [], [], []
    for k in range(k_run):
        a = A[:, k]
        if not a.any():
            continue
        nxt = T[:, k + 1]
        tgt = T[:, min(k + m, k_run)]
        xs.append(T[a, k]); gs.append(G[a, k]); fs.append((T[:, k] - nxt)[a]); tg.append(tgt[a])  # noqa: E702
        ks.append(torch.full((int(a.sum()),), k, dtype=T.dtype))
    return {"x": torch.cat(xs), "g": torch.cat(gs), "f": torch.cat(fs), "k": torch.cat(ks), "target": torch.cat(tg)}


def lookahead_loss(learner: MetaB, p: dict, idx) -> torch.Tensor:
    """Relative squared error of the chain output vs the m-step teacher target (per-sample k embedding)."""
    x, g, f, k, t = (p[key][idx] for key in ("x", "g", "f", "k", "target"))
    out = learner.chain(x, learner.standardize_grad(g), f, k)
    den = (x - t).pow(2).sum(1).clamp_min(1e-30)
    return ((out - t).pow(2).sum(1) / den).mean()


def fit_lookahead(learner: MetaB, train: dict, val: dict, cfg: LookaheadConfig, log=None) -> list:
    gen = torch.Generator().manual_seed(cfg.seed)
    opt = torch.optim.Adam(learner.net.parameters(), lr=cfg.lr)
    n, best, best_state, hist = train["x"].shape[0], float("inf"), None, []
    vidx = torch.arange(val["x"].shape[0])
    for ep in range(cfg.epochs):
        t0, tot = time.perf_counter(), 0.0
        learner.net.train()
        perm = torch.randperm(n, generator=gen)
        for b0 in range(0, n, cfg.batch_size):
            i = perm[b0:b0 + cfg.batch_size]
            opt.zero_grad()
            loss = lookahead_loss(learner, train, i)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(i)
        learner.net.eval()
        with torch.no_grad():
            vl = float(lookahead_loss(learner, val, vidx))
        h = {"epoch": ep, "train_rel": tot / n, "val_rel": vl, "s": time.perf_counter() - t0}
        hist.append(h)
        if log is not None:
            log(h)
        if vl < best:
            best, best_state = vl, copy.deepcopy(learner.net.state_dict())
    learner.net.load_state_dict(best_state)
    learner.net.eval()
    return hist
