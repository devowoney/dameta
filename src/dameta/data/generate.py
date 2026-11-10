"""
D1 — seeded twin-experiment dataset from M1 (physical Lorenz-63).

Output schema (xarray.Dataset, float64), one file per split: tmp/data/l63_<split>_seed<S>.nc
    x_true    (ic, t, var)          truth, t = 0 .. L + F (window + forecast), from M1
    y_obs     (ic, t_obs, var)      y = x_true[t_obs] + eps, eps ~ N(0, sigma_o^2 I); ALL variables are stored,
                                    the observation operator H is applied through obs_mask
    obs_mask  (pattern, t_obs, var) bool; pattern "full" = all variables, "x_only" = first variable only
    x0_b      (ic, var)             background x0_b = x_true[t=0] + eps_b, eps_b ~ N(0, sigma_b^2 I)
    coords    ic, t (step index), time (= t * dt), t_obs (step index of obs), var ("x","y","z"), pattern
    attrs     sigma, rho, beta, dt, window L, forecast F, obs_every, sigma_o2, sigma_b, spinup, split, seed,
              split_seed, dameta version

Held-out region (feature flag `heldout_region`, pilot decision 2026-11-10): the lobe-transition region near the
z-axis, r_xy = sqrt(x^2 + y^2), is reserved for a harder out-of-distribution split:
    train / val / test (in-distribution)  ->  initial conditions with r_xy >= ood_r_out
    test_ood                              ->  initial conditions with r_xy <  ood_r_in
The gap ood_r_in < r_xy < ood_r_out (default 3 -> 4, i.e. one background std) is never sampled, so no in-distribution
IC lies within that distance of an OOD IC. Rejection sampling on spun-up states; the flag off = no filtering.

Reproducibility: one master seed -> numpy SeedSequence.spawn() -> one independent stream per split
(train / val / test / test_ood). Within a split, a single torch.Generator draws, in a fixed order: IC seeds, background
noise, observation noise. Same seed => bit-identical file content.
"""

from dataclasses import dataclass, asdict

import numpy as np
import torch
import xarray as xr

from .. import __version__
from ..dynamics import L63Model, L63Params

SPLITS = ("train", "val", "test", "test_ood")  # fixed order: defines which spawned stream each split gets


@dataclass(frozen=True)
class DataConfig:
    """Approved DA configuration v0 (memory/original.md, 2026-11-10)."""

    window: int = 100          # L: assimilation-window length in model steps (1 time unit at dt=0.01)
    forecast: int = 500        # F: forecast steps beyond the window (5 time units)
    obs_every: int = 10        # observation interval in steps -> obs at t = 0, 10, ..., L (11 times)
    sigma_o2: float = 2.0      # observation-error variance, R = sigma_o2 * I
    sigma_b: float = 1.0       # background-error std, B = sigma_b^2 * I
    spinup: int = 1000         # steps to bring random draws onto the attractor
    ic_init_std: float = 10.0  # std of the random pre-spin-up draw (centred on (0, 0, 25))
    heldout_region: bool = True  # feature flag: reserve the lobe-transition region (small r_xy) for test_ood
    ood_r_in: float = 3.0        # test_ood ICs: r_xy < ood_r_in   (~14% of the attractor measure)
    ood_r_out: float = 4.0       # in-distribution ICs: r_xy >= ood_r_out   (~77%); gap = 1 background std


def split_seeds(seed: int) -> dict:
    """Master seed -> independent integer seed per split (numpy SeedSequence, statistically independent streams)."""
    children = np.random.SeedSequence(seed).spawn(len(SPLITS))
    return {name: int(child.generate_state(1, dtype=np.uint64)[0] % (2**63 - 1)) for name, child in zip(SPLITS, children)}


def region_radius(x: torch.Tensor) -> torch.Tensor:
    """r_xy = distance to the z-axis; small values = lobe-transition region of Lorenz-63."""
    return torch.hypot(x[..., 0], x[..., 1])


def _sample_ics(n_ic: int, split: str, gen: torch.Generator, cfg: DataConfig, model: L63Model) -> torch.Tensor:
    """Draw n_ic spun-up attractor states, restricted to the split's region when cfg.heldout_region is on."""
    centre = torch.tensor([0.0, 0.0, 25.0], dtype=torch.float64)
    batch = max(256, 4 * n_ic) if cfg.heldout_region else n_ic  # oversample when rejecting (OOD keeps ~14%)
    kept, n_kept = [], 0
    while n_kept < n_ic:
        x = model.final_state(centre + cfg.ic_init_std * torch.randn(batch, 3, generator=gen, dtype=torch.float64),
                              cfg.spinup)
        if cfg.heldout_region:
            r = region_radius(x)
            x = x[r < cfg.ood_r_in] if split == "test_ood" else x[r >= cfg.ood_r_out]
        kept.append(x)
        n_kept += x.shape[0]
    return torch.cat(kept)[:n_ic]


@torch.no_grad()
def generate_split(n_ic: int, split: str, seed: int, cfg: DataConfig, params: L63Params) -> xr.Dataset:
    """Generate one split of D1 (see module docstring for the schema)."""
    s_seed = split_seeds(seed)[split]
    gen = torch.Generator().manual_seed(s_seed)
    model = L63Model(params)
    dtype = torch.float64

    # 1) Initial conditions on the attractor: random draw around the attractor centre, then spin-up,
    #    then (if heldout_region) keep only states in the region assigned to this split (rejection sampling).
    x0_true = _sample_ics(n_ic, split, gen, cfg, model)

    # 2) Background = truth + N(0, sigma_b^2 I)
    x0_b = x0_true + cfg.sigma_b * torch.randn(n_ic, 3, generator=gen, dtype=dtype)

    # 3) Truth over window + forecast range
    n_t = cfg.window + cfg.forecast
    x_true = model.rollout(x0_true, n_t)                                     # [ic, n_t + 1, 3]

    # 4) Observations at t_obs = 0, obs_every, ..., window (inside the window only)
    t_obs = np.arange(0, cfg.window + 1, cfg.obs_every)
    y_obs = x_true[:, t_obs, :] + np.sqrt(cfg.sigma_o2) * torch.randn(n_ic, len(t_obs), 3, generator=gen, dtype=dtype)

    # 5) Observation patterns (H as a mask): full state, x-only
    mask = np.zeros((2, len(t_obs), 3), dtype=bool)
    mask[0] = True          # "full"
    mask[1, :, 0] = True    # "x_only"

    ds = xr.Dataset(
        data_vars={
            "x_true": (("ic", "t", "var"), x_true.numpy()),
            "y_obs": (("ic", "t_obs", "var"), y_obs.numpy()),
            "obs_mask": (("pattern", "t_obs", "var"), mask),
            "x0_b": (("ic", "var"), x0_b.numpy()),
        },
        coords={
            "ic": np.arange(n_ic),
            "t": np.arange(n_t + 1),
            "time": ("t", np.arange(n_t + 1) * params.dt),
            "t_obs": t_obs,
            "var": ["x", "y", "z"],
            "pattern": ["full", "x_only"],
        },
        attrs={
            **{k: float(v) for k, v in asdict(params).items()},
            **{k: (float(v) if isinstance(v, float) else int(v)) for k, v in asdict(cfg).items()},
            "split": split,
            "region": ("ood: r_xy < ood_r_in" if split == "test_ood" else "in-dist: r_xy >= ood_r_out")
                      if cfg.heldout_region else "none",
            "seed": int(seed),
            "split_seed": int(s_seed),
            "dameta_version": __version__,
            "description": "D1 Lorenz-63 twin experiment (truth from M1, noisy obs, background)",
        },
    )
    return ds
