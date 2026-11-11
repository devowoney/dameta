"""
Meta-learner networks for L63 (milestone 4).

StretchNet = S_theta of Meta-B, the L63 port of bice `meta_learner/network_s.py::UNetMetaGrad2D` (delta inner-loss
mode). bice uses a 2-D UNet over ocean fields; L63 has no spatial dimensions (d = 3 strong, 3 + 3 (n_obs - 1) weak),
so S is an MLP on the concatenation [x_std, g_std, emb(k)] -> s in R^d. Kept from bice:
    - iteration embedding emb(k) = Linear(1, e) -> SiLU -> Linear(e, e)
    - output bound  s = output_scale * tanh(out / output_scale)   (|s| < output_scale)
Changes vs bice (feature flags, see memory/branches/m4-meta-b/decisions.md):
    - emb input is log1p(k) instead of raw k (k runs to ~1000 outer iterations here; raw k saturates the SiLU MLP)
    - zero_init: last layer initialised to 0, so s = 0 exactly at start (the chain then IS the teacher step)
"""

from typing import Union

import torch
import torch.nn as nn


class IterationEmbedding(nn.Module):
    """emb(k): small MLP on log1p(k) (bice: on raw k)."""

    def __init__(self, dim: int = 8, log_input: bool = True):
        super().__init__()
        self.log_input = log_input
        self.net = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, k: Union[int, torch.Tensor], n: int) -> torch.Tensor:
        """Return [n, dim] (the same outer iteration k for every case of the batch)."""
        w = self.net[0].weight
        k = torch.as_tensor(k, dtype=w.dtype, device=w.device).reshape(1, 1)
        k = torch.log1p(k) if self.log_input else k
        return self.net(k).expand(n, -1)


class StretchNet(nn.Module):
    """S_theta(x_std, g_std, k) -> s [n, d], bounded by output_scale (bice delta mode recommends ~5)."""

    def __init__(self, d: int, hidden: int = 64, n_layers: int = 2, emb_dim: int = 8, output_scale: float = 5.0,
                 zero_init: bool = True, log_k: bool = True):
        super().__init__()
        self.d, self.output_scale = d, float(output_scale)
        self.emb = IterationEmbedding(emb_dim, log_input=log_k)
        layers, width = [], 2 * d + emb_dim
        for _ in range(n_layers):
            layers += [nn.Linear(width, hidden), nn.SiLU()]
            width = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, d)
        if zero_init:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x_std: torch.Tensor, g_std: torch.Tensor, k: Union[int, torch.Tensor]) -> torch.Tensor:
        h = torch.cat([x_std, g_std, self.emb(k, x_std.shape[0])], dim=1)
        out = self.head(self.body(h))
        return self.output_scale * torch.tanh(out / self.output_scale)
