"""
Classical iterative 4D-Var (configurations E1 with M1, E3 with the emulator): c^{k+1} = c^k - f(c^k).

All cases are solved in parallel as one batch, but convergence is tracked per case: a case stops (is frozen) at the
first iteration where its relative cost decrease |J^k - J^{k-1}| / J^{k-1} < tol, or at max_iter. Its iteration count
= number of gradient evaluations it used (one forward + one adjoint run each) — the cost metric of the study.

Output = D3 analysis record (dict of numpy arrays, written to NetCDF by the runner):
    x0_iter       [n, K + 1, 3]  initial-condition iterates (K = max_iter; frozen cases repeat their last value)
    eta_final     [n, n_obs - 1, 3] (weak only)
    J_iter        [n, K + 1]     cost along iterations
    cos_step_grad [n, K]         cos(step, gradient) per iteration (G2 with the M1 gradient; = 1 for GD)
    n_grad_evals  [n]            gradient evaluations used until convergence
    converged     [n] bool       tol reached before max_iter
    wall_s        scalar         wall-clock of the whole batch
"""

import time
from typing import Optional

import numpy as np
import torch

from .cost import FourDVarCost
from .reference_step import ReferenceStep


def run_fourdvar(cost: FourDVarCost, x0_init: torch.Tensor, method: str, lr: float, max_iter: int = 1000,
                 tol: float = 1e-6, eta_init: Optional[torch.Tensor] = None) -> dict:
    n = x0_init.shape[0]
    if cost.weak and eta_init is None:
        eta_init = torch.zeros(n, cost.cfg.n_obs - 1, 3, dtype=x0_init.dtype)   # first guess: perfect model
    c = cost.pack(x0_init, eta_init).detach().clone()
    teacher = ReferenceStep(c, method=method, lr=lr)

    active = torch.ones(n, dtype=c.dtype)                 # 1 = still iterating
    n_evals = torch.zeros(n, dtype=torch.long)
    x0_hist, j_hist, cos_hist = [c[:, :3].clone()], [], []
    t0 = time.perf_counter()
    j_prev = None
    for _ in range(max_iter):
        j, g = cost.grad(c)
        n_evals += active.long()
        j_hist.append(j.clone() if j_prev is None else torch.where(active.bool(), j, j_hist[-1]))
        if j_prev is not None:                                      # per-case stopping rule on relative decrease
            rel = (j_prev - j).abs() / j_prev.abs().clamp_min(1e-300)
            newly_done = active.bool() & ((rel < tol) | ~torch.isfinite(j))
            active = active * (~newly_done).to(c.dtype)
        if active.sum() == 0:
            break
        f = teacher(c, g, active)
        cos_hist.append(torch.nn.functional.cosine_similarity(f, g, dim=1))
        c = c - f
        x0_hist.append(c[:, :3].clone())
        j_prev = j
    wall = time.perf_counter() - t0

    # Final cost at the returned control (one extra forward run, not counted: reporting only)
    with torch.no_grad():
        j_final = cost(*cost.unpack(c))
    j_hist.append(j_final)
    k_run = len(x0_hist) - 1
    pad = max_iter - k_run                                          # pad histories to fixed length K + 1
    x0_iter = torch.stack(x0_hist, 1)
    x0_iter = torch.cat([x0_iter, x0_iter[:, -1:].expand(n, pad, 3)], 1) if pad > 0 else x0_iter
    # j_hist holds J(c^0..c^{k_run}) in its first k_run + 1 entries in both exit cases (early break: last grad call
    # already evaluated c^{k_run}; max_iter: the appended j_final is J(c^K)).
    j_iter = torch.stack(j_hist[: k_run + 1], 1)
    j_iter = torch.cat([j_iter, j_iter[:, -1:].expand(n, pad)], 1) if pad > 0 else j_iter
    cos = torch.stack(cos_hist, 1) if cos_hist else torch.zeros(n, 0, dtype=c.dtype)
    cos = torch.cat([cos, torch.full((n, max_iter - cos.shape[1]), float("nan"), dtype=c.dtype)], 1)
    out = {
        "x0_iter": x0_iter.numpy(), "J_iter": j_iter.numpy(), "cos_step_grad": cos.numpy(),
        "n_grad_evals": n_evals.numpy(), "converged": ((active == 0) & (n_evals < max_iter) & torch.isfinite(j_final)).numpy(),
        "x0_a": c[:, :3].numpy(), "wall_s": wall,
    }
    if cost.weak:
        out["eta_final"] = cost.unpack(c)[1].numpy()
    return out


@torch.no_grad()
def forecast_from_analysis(step, x0_a: torch.Tensor, cfg, n_forecast: int, eta: Optional[torch.Tensor] = None):
    """
    Analysis trajectory over the window (with eta if weak), then free forecast of n_forecast steps from the
    end-of-window analysis state x_L. Returns the forecast [n, n_forecast + 1, 3] for t = L .. L + F (lead 0 .. F).
    """
    from .cost import window_trajectory
    x = window_trajectory(step, x0_a, cfg, eta)[:, -1]
    states = [x]
    for _ in range(n_forecast):
        x = step(x)
        states.append(x)
    return torch.stack(states, 1)
