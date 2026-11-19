"""Observation features for meta-learner inputs (2026-11-18, pilot: "observation value and its time")."""

import torch


def obs_time_features(cost, ctrl_mean: torch.Tensor, ctrl_std: torch.Tensor) -> torch.Tensor:
    """[n, 7 n_obs] per case: y standardised by the train-truth climatology (x0 block of ctrl_mean / ctrl_std) and
    masked (3 n_obs), the observation mask (3 n_obs), the observation times t_i / L (n_obs; same for all cases)."""
    n = cost.y.shape[0]
    m = cost.mask.expand(n, -1, -1)
    y = (cost.y - ctrl_mean[:3]) / ctrl_std[:3] * m
    t = (cost.t_obs.to(cost.y.dtype) / cost.cfg.window).expand(n, -1)
    return torch.cat([y.flatten(1), m.flatten(1), t], 1)
