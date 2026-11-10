"""
Milestone 2 verification — 4D-Var costs (strong/weak), reference step (GD/Adagrad teacher), classical outer loop.

Plain-python test script (no pytest in env `oceanai`). Run from the project root:   python src/tests/test_m2.py
Requires D1 data in tmp/data (python src/gen_data.py).
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on the path

from dameta.da import CostConfig, FourDVarCost, ReferenceStep, forecast_from_analysis, run_fourdvar  # noqa: E402
from dameta.dynamics import L63Model  # noqa: E402

torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parents[2]
DS = xr.open_dataset(ROOT / "tmp/data/l63_val_seed42.nc").isel(ic=slice(0, 16))
M, CFG = L63Model(), CostConfig()
Y, XB, XT = torch.tensor(DS.y_obs.values), torch.tensor(DS.x0_b.values), torch.tensor(DS.x_true.values)
MASK = torch.tensor(DS.obs_mask.sel(pattern="full").values)
RESULTS = []


def report(name, value, thr, ok):
    RESULTS.append((name, value, thr, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<50s} value={value:.3e}  ({thr})")
    assert ok, name


def test_weak_reduces_to_strong():
    """Weak-constraint cost with eta = 0 equals the strong-constraint cost exactly."""
    js = FourDVarCost(M.step, Y, MASK, XB, CFG, weak=False)(XB)
    jw = FourDVarCost(M.step, Y, MASK, XB, CFG, weak=True)(XB, torch.zeros(16, CFG.n_obs - 1, 3))
    err = (js - jw).abs().max().item()
    report("weak(eta=0) == strong", err, "max abs < 1e-12", err < 1e-12)


def test_truth_window_consistency():
    """Window trajectory from the true x0 reproduces x_true over the window; forecast continues it to L + F."""
    fc = forecast_from_analysis(M.step, XT[:, 0], CFG, 500)
    err = (fc - XT[:, CFG.window:CFG.window + 501]).abs().max().item()
    report("forecast_from_analysis(truth) == x_true[L:L+F]", err, "max abs < 1e-9", err < 1e-9)


def test_weak_gradient_fd():
    """Weak-constraint gradient w.r.t. packed control (x0, eta): autodiff vs central FD directional derivative."""
    cost = FourDVarCost(M.step, Y[:4], MASK, XB[:4], CFG, weak=True)
    g = torch.Generator().manual_seed(5)
    c = cost.pack(XB[:4], 0.05 * torch.randn(4, CFG.n_obs - 1, 3, generator=g))
    d = torch.randn(c.shape, generator=g)
    _, grad = cost.grad(c)
    h = 1e-6
    fd = (cost(*cost.unpack(c + h * d)) - cost(*cost.unpack(c - h * d))) / (2 * h)
    ad = (grad * d).sum(1)
    rel = ((fd - ad).abs() / ad.abs()).max().item()
    report("weak gradient: autodiff vs FD (directional)", rel, "relative < 1e-6", rel < 1e-6)


def test_reference_step():
    """GD step = lr*g; Adagrad step matches the manual formula over 3 calls; inactive rows get no step/no state."""
    g = torch.Generator().manual_seed(6)
    c, grads = torch.randn(4, 3, generator=g), [torch.randn(4, 3, generator=g) for _ in range(3)]
    gd = ReferenceStep(c, "gd", lr=0.1)
    e_gd = (gd(c, grads[0]) - 0.1 * grads[0]).abs().max().item()
    ada, acc, err = ReferenceStep(c, "adagrad", lr=0.1), torch.zeros(4, 3), 0.0
    for gk in grads:                                   # torch Adagrad defaults: eps = 1e-10, no decay
        acc = acc + gk**2
        err = max(err, (ada(c, gk) - 0.1 * gk / (acc.sqrt() + 1e-10)).abs().max().item())
    active = torch.tensor([1.0, 0.0, 1.0, 1.0])
    frozen = ReferenceStep(c, "gd", lr=0.1)(c, grads[0], active)[1].abs().max().item()
    report("ReferenceStep GD / Adagrad / frozen row", max(e_gd, err, frozen), "max abs < 1e-12",
           max(e_gd, err, frozen) < 1e-12)


def test_outer_loop_bookkeeping():
    """J_iter[:, 0] = J(background); GD with a small lr decreases J monotonically; converged => evals < max_iter."""
    cost = FourDVarCost(M.step, Y, MASK, XB, CFG)
    res = run_fourdvar(cost, XB, "gd", lr=1e-4, max_iter=60, tol=1e-6)
    e0 = np.abs(res["J_iter"][:, 0] - cost(XB).numpy()).max()
    report("J_iter[:,0] == J(x0_b)", e0, "< 1e-10", e0 < 1e-10)
    dj = np.diff(res["J_iter"], axis=1)
    report("GD lr=1e-4: J non-increasing (max dJ)", dj.max(), "<= 1e-10", dj.max() <= 1e-10)
    ok = bool(np.all(res["n_grad_evals"][res["converged"]] < 60)) and res["x0_iter"].shape == (16, 61, 3)
    report("shapes / converged-evals consistency", 0.0 if ok else 1.0, "== 0", ok)


def test_analysis_beats_background():
    """Adagrad 4D-Var (strong, full obs) reduces the IC error w.r.t. the background on average."""
    cost = FourDVarCost(M.step, Y, MASK, XB, CFG)
    res = run_fourdvar(cost, XB, "adagrad", lr=0.1, max_iter=300, tol=1e-6)
    ra = np.sqrt(((res["x0_a"] - DS.x_true.values[:, 0]) ** 2).mean(-1))
    rb = np.sqrt(((DS.x0_b.values - DS.x_true.values[:, 0]) ** 2).mean(-1))
    report("median rmse_x0 analysis / background", float(np.median(ra) / np.median(rb)), "< 1", np.median(ra) < np.median(rb))


if __name__ == "__main__":
    t0 = time.time()
    for name, fn in [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]:
        fn()
    print(f"\nAll {len(RESULTS)} checks passed in {time.time() - t0:.1f}s")
