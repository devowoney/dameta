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



class _QuadCost:
    """Minimal FourDVarCost-like quadratic: grad(c) -> (J, g) batched; per-case copies via the constructor."""
    def __init__(self, a, x_star):
        self.a, self.x_star = a, x_star

    def grad(self, c):
        r = c - self.x_star
        g = torch.einsum("nij,nj->ni", self.a, r)
        return 0.5 * (r * g).sum(1), g


def test_lbfgs_wolfe_matches_torch_lbfgs():
    """Teacher 'lbfgs_wolfe' (batched, first trial shared) reproduces torch.optim.LBFGS(strong_wolfe) iterates."""
    a, x_star = _quadratic(n=3, d=6)
    cost = _QuadCost(a, x_star)
    t = ReferenceStep(torch.zeros_like(x_star), method="lbfgs_wolfe", lr=1.0, cost=cost)
    t._case_cost = lambda i: _QuadCost(a[i:i + 1], x_star[i:i + 1])
    c = torch.zeros_like(x_star)
    ours = [c.clone()]
    for _ in range(6):
        j, g = cost.grad(c)
        c = c - t(c, g, j=j)
        ours.append(c.clone())
    for i in range(3):
        x = torch.zeros(6, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([x], lr=1.0, max_iter=1, history_size=10, tolerance_grad=1e-12,
                                tolerance_change=1e-14, line_search_fn="strong_wolfe")
        for k in range(6):
            def closure():
                opt.zero_grad()
                r = x - x_star[i]
                f = 0.5 * r @ a[i] @ r
                f.backward()
                return f
            opt.step(closure)
            assert torch.allclose(x.detach(), ours[k + 1][i], rtol=1e-6, atol=1e-8), (i, k)
