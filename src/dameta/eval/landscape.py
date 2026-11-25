"""
Optimization-landscape diagnostics of 4D-Var (M8, branch m8-classic-4dv, 2026-11-25).

For a batched cost J(c) (FourDVarCost: cases independent, control c [n, d]; d = 3 strong, 21 weak):
    hessian_batch  H [n, d, d]  exact Hessian of J w.r.t. the packed control by double backward (one reverse pass per
                                control component; since cases do not interact, the i-th pass of sum_n g_n,i gives row i
                                of every case's Hessian). Works for L63 and the emulator (SiLU MLP + RK4, C^inf).
    spectrum       lambda_min, lambda_max, n_neg (# eigenvalues < 0), kappa_H = lambda_max / lambda_min if
                   lambda_min > 0 else NaN (pilot 2026-11-25: kappa only for positive-definite H).
    cos_truth      cos(x_true(0) - x0^k, -grad_x0 J(c^k)): ~1 the negative gradient points to the true IC, ~0 it is
                   orthogonal to the useful correction (weak: x0 block of the gradient only).
"""

import torch


def hessian_batch(cost, c: torch.Tensor) -> torch.Tensor:
    """Per-case Hessian [n, d, d] of cost(*cost.unpack(c)) at the packed control c [n, d]."""
    x = c.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        j = cost(*cost.unpack(x))
        (g,) = torch.autograd.grad(j.sum(), x, create_graph=True)
        rows = []
        for i in range(x.shape[1]):
            (h_i,) = torch.autograd.grad(g[:, i].sum(), x, retain_graph=i < x.shape[1] - 1)
            rows.append(h_i)
    h = torch.stack(rows, 1).detach()
    return 0.5 * (h + h.transpose(1, 2))


def spectrum(h: torch.Tensor) -> dict:
    """Eigen-summary of symmetric [n, d, d] (NaN cases propagate to NaN)."""
    finite = torch.isfinite(h).flatten(1).all(1)
    ev = torch.full(h.shape[:2], float("nan"), dtype=h.dtype)
    if finite.any():
        ev[finite] = torch.linalg.eigvalsh(h[finite])
    lmin, lmax = ev[:, 0], ev[:, -1]
    n_neg = torch.where(finite, (ev < 0).sum(1).to(h.dtype), torch.full_like(lmin, float("nan")))
    kappa = torch.where(lmin > 0, lmax / lmin, torch.full_like(lmin, float("nan")))
    return {"eig": ev, "lam_min": lmin, "lam_max": lmax, "n_neg": n_neg, "kappa_H": kappa}


def cos_truth(x0: torch.Tensor, g_x0: torch.Tensor, x0_true: torch.Tensor) -> torch.Tensor:
    """cos(x_true - x0, -grad_x0 J) per case [n]."""
    return torch.nn.functional.cosine_similarity(x0_true - x0, -g_x0, dim=1)
