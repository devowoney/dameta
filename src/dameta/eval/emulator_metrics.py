"""
Emulator diagnostics and gradient fidelity G1 (reference = exact M1 derivatives, available for Lorenz-63).

All functions are batched over cases, float64, and take `step` callables (M1: L63Model.step; M2: L63Emulator.step).

    free_forecast_rmse   emulator vs truth from the TRUE initial condition: RMSE per lead (pure model error)
    lyapunov_exponent    leading Lyapunov exponent of a discrete model (renormalized TL propagation)
    tl_fidelity          cos and relative error between emulator and M1 tangent-linear propagation over the window
    gradient_fidelity    G1: cos(grad J_M2, grad J_M1) and ||grad J_M2 - grad J_M1|| / ||grad J_M1|| for the SAME
                         4D-Var problem (obs, background, B, R[, Q]) at the same control
"""

import torch

from ..da.cost import CostConfig, FourDVarCost


@torch.no_grad()
def free_forecast_rmse(step, x0: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """x0 [n, 3], truth [n, T + 1, 3] -> RMSE over variables [n, T + 1]."""
    states, x = [x0], x0
    for _ in range(truth.shape[1] - 1):
        x = step(x)
        states.append(x)
    return ((torch.stack(states, 1) - truth) ** 2).mean(-1).sqrt()


def lyapunov_exponent(step, x0: torch.Tensor, n_steps: int, dt: float, seed: int = 0) -> torch.Tensor:
    """Leading exponent per trajectory [n]: mean log growth of a renormalized tangent vector over n_steps."""
    v = torch.randn(x0.shape, generator=torch.Generator().manual_seed(seed), dtype=x0.dtype)
    v = v / v.norm(dim=-1, keepdim=True)
    x, total = x0, torch.zeros(x0.shape[0], dtype=x0.dtype)
    for _ in range(n_steps):
        x, v = torch.func.jvp(step, (x,), (v,))
        x, v = x.detach(), v.detach()
        nv = v.norm(dim=-1)
        total += nv.log()
        v = v / nv[:, None]
    return total / (n_steps * dt)


def tl_fidelity(step_emu, step_ref, x0: torch.Tensor, n_steps: int, seed: int = 0) -> dict:
    """
    Propagate the same random perturbation with both tangent-linear models over n_steps, each linearized along its
    OWN trajectory from x0 (what 4D-Var through the emulator actually uses). Returns cos and relative error [n].
    """
    dx = torch.randn(x0.shape, generator=torch.Generator().manual_seed(seed), dtype=x0.dtype)

    def final(step):
        def f(x):
            for _ in range(n_steps):
                x = step(x)
            return x
        return f

    _, tl_e = torch.func.jvp(final(step_emu), (x0,), (dx,))
    _, tl_r = torch.func.jvp(final(step_ref), (x0,), (dx,))
    cos = torch.nn.functional.cosine_similarity(tl_e, tl_r, dim=-1)
    rel = (tl_e - tl_r).norm(dim=-1) / tl_r.norm(dim=-1)
    return {"cos": cos.detach(), "rel_err": rel.detach()}


def gradient_fidelity(step_emu, step_ref, y, mask, x0_b, cfg: CostConfig, control: torch.Tensor, weak: bool) -> dict:
    """G1 at a given packed control [n, d]: same cost definition, dynamics swapped. Returns cos and rel_err [n]."""
    c_emu = FourDVarCost(step_emu, y, mask, x0_b, cfg, weak=weak)
    c_ref = FourDVarCost(step_ref, y, mask, x0_b, cfg, weak=weak)
    _, g_e = c_emu.grad(control)
    _, g_r = c_ref.grad(control)
    cos = torch.nn.functional.cosine_similarity(g_e, g_r, dim=-1)
    rel = (g_e - g_r).norm(dim=-1) / g_r.norm(dim=-1)
    return {"cos": cos, "rel_err": rel}
