"""Batched L-BFGS teacher direction (ReferenceStep method 'lbfgs', 2026-11-16)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dameta.da.reference_step import ReferenceStep  # noqa: E402


def _quadratic(n=4, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.linalg.qr(torch.randn(n, d, d, generator=g, dtype=torch.float64))[0]
    ev = torch.logspace(0, 3, d, dtype=torch.float64)                       # condition number 1e3
    a = q @ torch.diag_embed(ev.expand(n, d)) @ q.transpose(1, 2)
    x_star = torch.randn(n, d, generator=g, dtype=torch.float64)
    return a, x_star


def test_lbfgs_converges_on_quadratic():
    a, x_star = _quadratic()
    c = torch.zeros_like(x_star)
    t = ReferenceStep(c, method="lbfgs", lr=1.0)
    err0 = (c - x_star).norm(dim=1)
    for _ in range(60):
        g = torch.einsum("nij,nj->ni", a, c - x_star)
        c = c - t(c, g)
    assert ((c - x_star).norm(dim=1) / err0).max() < 1e-6


def test_lbfgs_descent_and_frozen_cases():
    a, x_star = _quadratic(seed=1)
    c = torch.zeros_like(x_star)
    t = ReferenceStep(c, method="lbfgs", lr=0.5)
    active = torch.tensor([1.0, 1.0, 0.0, 1.0], dtype=torch.float64)
    for _ in range(10):
        g = torch.einsum("nij,nj->ni", a, c - x_star)
        f = t(c, g, active)
        assert ((g * f).sum(1)[active.bool()] > 0).all()                   # descent direction
        assert f[2].abs().max() == 0                                         # frozen case does not move
        c = c - f
