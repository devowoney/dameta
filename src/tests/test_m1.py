"""
Milestone 1 verification — M1 (Lorenz-63 + RK4) and D1 data generation.

Plain-python test script (pytest is not installed in env `oceanai`; we do not install packages without asking).
Run from the project root:   python src/tests/test_m1.py
Each test prints a one-line verdict with the measured number; any failure raises AssertionError.

Behavioral rule "correctness first": TL Taylor test + adjoint dot-product test must pass before any 4D-Var run.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on the path

from dameta.data import DataConfig, generate_split, split_seeds  # noqa: E402
from dameta.dynamics import L63Model, L63Params, l63_jacobian, l63_tendency  # noqa: E402

torch.set_default_dtype(torch.float64)
P = L63Params()
M = L63Model(P)
WINDOW = 100  # approved window length L (steps)
RESULTS = []  # (name, measured value, threshold) — collected for the findings.md entry


def _on_attractor(seed: int, n: int = 1) -> torch.Tensor:
    """Helper: n states on the attractor (random draw + 1000-step spin-up)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.tensor([0.0, 0.0, 25.0]) + 10.0 * torch.randn(n, 3, generator=g)
    with torch.no_grad():
        return M.final_state(x, 1000)


def report(name: str, value: float, thr: str, ok: bool) -> None:
    RESULTS.append((name, value, thr, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<46s} value={value:.3e}  ({thr})")
    assert ok, name


def test_jacobian_analytic_vs_autodiff():
    """Autodiff Jacobian of the tendency equals the analytic Lorenz-63 Jacobian."""
    x = _on_attractor(0, 8)
    j_ad = torch.vmap(torch.func.jacrev(lambda v: l63_tendency(v, P)))(x)
    err = (j_ad - l63_jacobian(x, P)).abs().max().item()
    report("tendency Jacobian: autodiff vs analytic", err, "max abs < 1e-12", err < 1e-12)


def test_tangent_linear_taylor():
    """TL Taylor test over the window: ||M(x+a dx) - M(x)|| / ||a TL dx|| -> 1 as a -> 0 (first-order accurate)."""
    x = _on_attractor(1)[0]
    dx = torch.randn(3, generator=torch.Generator().manual_seed(11))
    f = lambda v: M.final_state(v, WINDOW)  # noqa: E731
    _, tl = torch.func.jvp(f, (x,), (dx,))
    ratios = []
    for a in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        ratios.append(((f(x + a * dx) - f(x)).norm() / (a * tl).norm()).item())
    err = abs(ratios[-1] - 1.0)
    print("      Taylor ratios (a=1e-2..1e-6):", ", ".join(f"{r:.8f}" for r in ratios))
    report("TL Taylor test |ratio-1| at a=1e-6", err, "< 1e-4", err < 1e-4)


def test_adjoint_dot_product():
    """Dot-product test: <TL dx, dy> == <dx, ADJ dy> to machine precision (window L=100)."""
    x = _on_attractor(2)[0]
    g = torch.Generator().manual_seed(12)
    dx, dy = torch.randn(3, generator=g), torch.randn(3, generator=g)
    f = lambda v: M.final_state(v, WINDOW)  # noqa: E731
    _, tl = torch.func.jvp(f, (x,), (dx,))
    _, vjp_fn = torch.func.vjp(f, x)
    (adj,) = vjp_fn(dy)
    lhs, rhs = torch.dot(tl, dy).item(), torch.dot(dx, adj).item()
    rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs))
    report("adjoint dot-product test (relative)", rel, "< 1e-12", rel < 1e-12)


def test_4dvar_gradient_fd():
    """Autodiff gradient of a strong-constraint J(x0) = Jb + Jo matches central finite differences."""
    ds = generate_split(1, "val", 0, DataConfig(), P)
    y = torch.tensor(ds.y_obs.values[0])            # [11, 3]
    x0_b = torch.tensor(ds.x0_b.values[0])
    t_obs = torch.tensor(ds.t_obs.values)

    def J(x0):  # minimal Jb + Jo, full-state H, B = I, R = 2 I (proper cost module comes in milestone 2)
        traj = M.rollout(x0, WINDOW)
        return 0.5 * ((x0 - x0_b) ** 2).sum() + 0.5 * ((traj[t_obs] - y) ** 2).sum() / 2.0

    x0 = x0_b.clone().requires_grad_(True)
    (g_ad,) = torch.autograd.grad(J(x0), x0)
    h, g_fd = 1e-6, torch.zeros(3)
    for i in range(3):
        e = torch.zeros(3)
        e[i] = h
        g_fd[i] = (J(x0_b + e) - J(x0_b - e)) / (2 * h)
    rel = ((g_ad - g_fd).norm() / g_ad.norm()).item()
    report("4D-Var gradient: autodiff vs central FD", rel, "relative < 1e-6", rel < 1e-6)


def test_lyapunov_exponent():
    """
    Leading Lyapunov exponent ~ 0.906: physical check of the integrator.
    Mean over 16 trajectories x 50 time units (single 50 TU runs scatter by ~0.06; a 300 TU x 16 run gave
    0.904 +/- 0.012, see memory/findings.md).
    """
    x = _on_attractor(3, 16)
    v = torch.randn(16, 3, generator=torch.Generator().manual_seed(13))
    v = v / v.norm(dim=-1, keepdim=True)
    total, n = torch.zeros(16), 5000
    for _ in range(n):
        x, v = torch.func.jvp(M.step, (x,), (v,))   # batched TL propagation of one perturbation per trajectory
        nv = v.norm(dim=-1)
        total += torch.log(nv)
        v = v / nv[:, None]
    lam = (total / (n * P.dt)).mean().item()
    report("leading Lyapunov exponent (16 traj mean)", lam, "0.85 < lambda1 < 0.96 (ref 0.906)", 0.85 < lam < 0.96)


def test_reproducibility_and_splits():
    """Same seed -> bit-identical data; train/val/test streams distinct."""
    cfg = DataConfig()
    a = generate_split(5, "train", 42, cfg, P)
    b = generate_split(5, "train", 42, cfg, P)
    same = all(np.array_equal(a[v].values, b[v].values) for v in ["x_true", "y_obs", "x0_b"])
    report("reproducibility (same seed, bit-identical)", 0.0 if same else 1.0, "== 0", same)
    seeds = split_seeds(42)
    distinct = len(set(seeds.values())) == 3
    c = generate_split(5, "test", 42, cfg, P)
    gap = float(np.abs(a.x_true.values[:, 0] - c.x_true.values[:, 0]).min())
    report("train/test IC separation (min abs diff)", gap, "> 0 and distinct seeds", distinct and gap > 0)


def test_data_schema_and_statistics():
    """D1 schema (shapes, dtype, attrs) and noise statistics match the approved configuration."""
    cfg = DataConfig()
    ds = generate_split(400, "train", 42, cfg, P)
    n_t = cfg.window + cfg.forecast + 1
    ok_shape = (ds.x_true.shape == (400, n_t, 3) and ds.y_obs.shape == (400, 11, 3) and ds.x0_b.shape == (400, 3)
                and ds.obs_mask.shape == (2, 11, 3) and ds.x_true.dtype == np.float64)
    report("schema shapes / float64", 0.0 if ok_shape else 1.0, "== 0", ok_shape)
    eps_o = ds.y_obs.values - ds.x_true.values[:, ds.t_obs.values]
    var_o = float(eps_o.var())
    report("obs-error variance", var_o, "|var - 2| < 0.1", abs(var_o - cfg.sigma_o2) < 0.1)
    var_b = float((ds.x0_b.values - ds.x_true.values[:, 0]).var())
    report("background-error variance", var_b, "|var - 1| < 0.1", abs(var_b - cfg.sigma_b**2) < 0.1)
    zmin, zmax = float(ds.x_true.values[..., 2].min()), float(ds.x_true.values[..., 2].max())
    ok_att = 0.0 < zmin and zmax < 50.0 and np.isfinite(ds.x_true.values).all()
    report("truth on attractor (z range)", zmax, f"0 < z (min {zmin:.2f}) and z < 50", ok_att)


if __name__ == "__main__":
    t0 = time.time()
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\nAll {len(RESULTS)} checks passed in {time.time() - t0:.1f}s")
