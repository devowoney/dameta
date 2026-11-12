"""
Backward Lyapunov vectors (BLVs) along the D1 truth trajectories — for P3 (milestone 4).

BLVs at time t need the trajectory BEFORE t. D1 stores the truth from t0 only, but the spin-up is reproducible:
`spinup_history` replays the split's seeded generator exactly as `data.generate._sample_ics` (pre-spin-up draw ->
spin-up -> region rejection) while keeping the trajectory, giving the `spinup` steps that precede each x_true[:, 0]
(bit-identical end state, checked by the caller).

`blv_along`: Benettin / QR iteration of the tangent-linear model (exact Jacobian of the discrete RK4 step by
autodiff) along a trajectory: Q_{t+1} R_{t+1} = J(x_t) Q_t, Q_0 = I. After the transient (~ 1000 steps = 10 TU here,
convergence ~ exp(-(lambda_1 - lambda_2) T) ~ 1e-4), the columns of Q_t are the orthonormal BLVs at t ordered
lambda_1 > lambda_2 > lambda_3 (L63: ~ 0.9, 0, -14.6); log|diag R| / dt averages to the Lyapunov exponents.
"""

from typing import Iterable

import numpy as np
import torch
import xarray as xr

from ..data.generate import DataConfig, region_radius, split_seeds
from ..dynamics import L63Model, L63Params


@torch.no_grad()
def spinup_history(ds: xr.Dataset, cfg: DataConfig = DataConfig()) -> torch.Tensor:
    """Truth over the spin-up preceding each IC of a D1 split file: [n_ic, spinup + 1, 3] (last = x_true[:, 0])."""
    params = L63Params(sigma=ds.attrs["sigma"], rho=ds.attrs["rho"], beta=ds.attrs["beta"], dt=ds.attrs["dt"])
    model, split, n_ic = L63Model(params), ds.attrs["split"], ds.sizes["ic"]
    gen = torch.Generator().manual_seed(split_seeds(int(ds.attrs["seed"]))[split])
    centre = torch.tensor([0.0, 0.0, 25.0], dtype=torch.float64)
    batch = max(256, 4 * n_ic) if cfg.heldout_region else n_ic
    kept, n_kept = [], 0
    while n_kept < n_ic:                                  # same draws and order as generate._sample_ics
        x0 = centre + cfg.ic_init_std * torch.randn(batch, 3, generator=gen, dtype=torch.float64)
        traj = model.rollout(x0, cfg.spinup)
        if cfg.heldout_region:
            r = region_radius(traj[:, -1])
            traj = traj[r < cfg.ood_r_in] if split == "test_ood" else traj[r >= cfg.ood_r_out]
        kept.append(traj)
        n_kept += traj.shape[0]
    return torch.cat(kept)[:n_ic]


def blv_along(step, traj: torch.Tensor, at: Iterable[int], q0: torch.Tensor = None):
    """
    QR iteration along traj [n, T + 1, 3]. Returns ({t: Q_t [n, 3, 3] (columns = BLVs)}, log-growth sums [n, 3],
    final Q). `at` = time indices (0..T) whose Q is kept; q0 = starting basis (default identity).
    """
    n = traj.shape[0]
    jac = torch.func.vmap(torch.func.jacrev(step))
    q = torch.eye(3, dtype=traj.dtype).expand(n, 3, 3).clone() if q0 is None else q0
    at, out, logs = set(at), {}, torch.zeros(n, 3, dtype=traj.dtype)
    if 0 in at:
        out[0] = q.clone()
    for t in range(traj.shape[1] - 1):
        q, r = torch.linalg.qr(jac(traj[:, t]) @ q)
        d = torch.diagonal(r, dim1=-2, dim2=-1)
        q = q * torch.sign(d)[:, None, :]                 # unique QR (positive diag R)
        logs += d.abs().log()
        if t + 1 in at:
            out[t + 1] = q.clone()
    return out, logs, q


def project_fraction(v: np.ndarray, basis: np.ndarray, k: int = 1) -> np.ndarray:
    """Fraction of |v|^2 in span of the first k columns of basis [n, 3, 3]; v [n, 3]."""
    c = np.einsum("ni,nij->nj", v, basis)[:, :k]
    return (c**2).sum(1) / np.maximum((v**2).sum(1), 1e-300)


def project_abs(v: np.ndarray, basis: np.ndarray, j: int = 0) -> np.ndarray:
    """|<v, b_j>| per case."""
    return np.abs(np.einsum("ni,ni->n", v, basis[:, :, j]))
