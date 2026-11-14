"""
D1 v2 — truth + observations for the model-error-isolated M5 experiments (memory/branches/m5-meta-grid/blueprint.md).

Changes vs D1 (generate.py, untouched):
  - window L = 110 steps (1.1 TU = Lyapunov time of the false model, common to F1 and F2), obs every 10 steps -> 12
    observation times (t = 0, 10, ..., 110); forecast range F = 500 steps beyond the window.
  - the truth over the `history` steps BEFORE t0 is stored (x_true_past, lag 0 = t0), so that time-lagged
    false-model forecasts valid at t0 can be built (background.py).
  - NO background in this file: the first guess and B depend on the false model (see background.py).
Output schema (one file per split, tmp/data/l63v2_<split>_seed<S>.nc, float64):
    x_true      (ic, t, var)        truth for t = 0 .. L + F (t0 = window start)
    x_true_past (ic, lag, var)      truth at t0 - lag, lag = 0 .. history (lag 0 = x_true[:, 0])
    y_obs       (ic, t_obs, var)    y = x_true[t_obs] + N(0, sigma_o2 I), all variables stored
    obs_mask    (pattern, t_obs, var)  "full" / "x_only"
Reproducibility: same master seed -> SeedSequence.spawn per split (as D1), one torch.Generator per split drawing, in a
fixed order: pre-spin-up states, observation noise. Held-out region rule as D1, applied to x(t0).
"""

from dataclasses import asdict, dataclass

import numpy as np
import torch
import xarray as xr

from .. import __version__
from ..dynamics import L63Model, L63Params
from .generate import region_radius, split_seeds


@dataclass(frozen=True)
class DataConfigV2:
    window: int = 110            # L steps (1.1 TU)
    forecast: int = 500          # F steps beyond the window
    obs_every: int = 10          # obs at t = 0, 10, ..., L  -> 12 times
    sigma_o2: float = 2.0        # R = sigma_o2 I
    history: int = 100           # truth steps stored before t0 (max lag of the lagged ensemble, 1 TU)
    spinup: int = 1000           # steps to reach the attractor (before t0 - history)
    ic_init_std: float = 10.0
    heldout_region: bool = True
    ood_r_in: float = 3.0
    ood_r_out: float = 4.0


@torch.no_grad()
def _sample_trajectories(n_ic: int, split: str, gen: torch.Generator, cfg: DataConfigV2, model: L63Model):
    """Spun-up truth from t0 - history to t0 for n_ic cases whose x(t0) lies in the split's region."""
    centre = torch.tensor([0.0, 0.0, 25.0], dtype=torch.float64)
    batch = max(256, 4 * n_ic) if cfg.heldout_region else n_ic
    kept, n_kept = [], 0
    while n_kept < n_ic:
        x = model.final_state(centre + cfg.ic_init_std * torch.randn(batch, 3, generator=gen, dtype=torch.float64),
                              cfg.spinup)
        past = model.rollout(x, cfg.history)                      # [b, history + 1, 3], index -1 = t0
        if cfg.heldout_region:
            r = region_radius(past[:, -1])
            past = past[r < cfg.ood_r_in] if split == "test_ood" else past[r >= cfg.ood_r_out]
        kept.append(past)
        n_kept += past.shape[0]
    return torch.cat(kept)[:n_ic]


@torch.no_grad()
def generate_split_v2(n_ic: int, split: str, seed: int, cfg: DataConfigV2, params: L63Params) -> xr.Dataset:
    s_seed = split_seeds(seed)[split]
    gen = torch.Generator().manual_seed(s_seed)
    model = L63Model(params)
    past = _sample_trajectories(n_ic, split, gen, cfg, model)       # t0 - history .. t0
    x_true = model.rollout(past[:, -1], cfg.window + cfg.forecast)  # t0 .. t0 + L + F
    t_obs = np.arange(0, cfg.window + 1, cfg.obs_every)
    y_obs = x_true[:, t_obs] + np.sqrt(cfg.sigma_o2) * torch.randn(n_ic, len(t_obs), 3, generator=gen,
                                                                     dtype=torch.float64)
    mask = np.zeros((2, len(t_obs), 3), dtype=bool)
    mask[0] = True
    mask[1, :, 0] = True
    n_t = cfg.window + cfg.forecast
    return xr.Dataset(
        data_vars={
            "x_true": (("ic", "t", "var"), x_true.numpy()),
            "x_true_past": (("ic", "lag", "var"), past.flip(1).numpy()),   # lag 0 = t0
            "y_obs": (("ic", "t_obs", "var"), y_obs.numpy()),
            "obs_mask": (("pattern", "t_obs", "var"), mask),
        },
        coords={"ic": np.arange(n_ic), "t": np.arange(n_t + 1), "time": ("t", np.arange(n_t + 1) * params.dt),
                "lag": np.arange(cfg.history + 1), "t_obs": t_obs, "var": ["x", "y", "z"],
                "pattern": ["full", "x_only"]},
        attrs={**{k: float(v) for k, v in asdict(params).items()},
               **{k: (float(v) if isinstance(v, float) else int(v)) for k, v in asdict(cfg).items()},
               "split": split, "region": ("ood: r_xy < ood_r_in" if split == "test_ood" else
                                          "in-dist: r_xy >= ood_r_out") if cfg.heldout_region else "none",
               "seed": int(seed), "split_seed": int(s_seed), "dameta_version": __version__, "dataset": "D1v2",
               "description": "D1 v2: truth (L63) + noisy obs; window = Lyapunov time of the false model; "
                              "truth history before t0 for time-lagged false-model backgrounds"},
    )
