"""
D1 v2 verification (truth + obs with truth history; time-lagged false-model backgrounds).

Plain-python test script. Run from the project root after `python src/gen_data_v2.py`:   python src/tests/test_d1v2.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dameta.data.background import lagged_background  # noqa: E402
from dameta.data.generate import region_radius  # noqa: E402
from dameta.data.generate_v2 import DataConfigV2, generate_split_v2  # noqa: E402
from dameta.dynamics import L63Model, L63Params  # noqa: E402

torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "tmp/data"
RESULTS = []


def report(name, value, thr, ok):
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<64s} value={value:.3e}  ({thr})")
    assert ok, name


def test_reproducible():
    ds = xr.open_dataset(D / "l63v2_val_seed42.nc")
    new = generate_split_v2(200, "val", 42, DataConfigV2(), L63Params())
    err = max(float(np.abs(ds[v].values - new[v].values).max()) for v in ("x_true", "x_true_past", "y_obs"))
    report("val split regenerates bit-identically", err, "== 0", err == 0)


def test_truth_consistency():
    m = L63Model()
    for split in ("train", "test_ood"):
        ds = xr.open_dataset(D / f"l63v2_{split}_seed42.nc")
        past, xt = torch.tensor(ds.x_true_past.values), torch.tensor(ds.x_true.values)
        e0 = (past[:, 0] - xt[:, 0]).abs().max().item()
        e1 = (m.final_state(past[:, 100], 100) - xt[:, 0]).abs().max().item()
        e2 = (m.final_state(xt[:, 0], 110) - xt[:, 110]).abs().max().item()
        report(f"{split}: history lag 0 == x(t0); 100 steps from lag 100 == x(t0); window end", max(e0, e1, e2),
               "== 0", max(e0, e1, e2) == 0)


def test_region_obs():
    ds = xr.open_dataset(D / "l63v2_test_ood_seed42.nc")
    r_ood = float(region_radius(torch.tensor(ds.x_true.values[:, 0])).max())
    ds2 = xr.open_dataset(D / "l63v2_test_seed42.nc")
    r_in = float(region_radius(torch.tensor(ds2.x_true.values[:, 0])).min())
    report("held-out region on x(t0): max r_ood < 3 and min r_in >= 4", r_ood, f"< 3 (r_in {r_in:.2f} >= 4)",
           r_ood < 3 and r_in >= 4)
    tr = xr.open_dataset(D / "l63v2_train_seed42.nc")
    n_obs = tr.sizes["t_obs"]
    noise = tr.y_obs.values - tr.x_true.values[:, tr.t_obs.values]
    s = float(noise.std())
    report("12 obs times (0..110 every 10), noise std ~ sqrt(2)", s, f"n_obs {n_obs} == 12, |std - 1.414| < 0.02",
           n_obs == 12 and abs(s - np.sqrt(2)) < 0.02)


def test_backgrounds():
    ds = xr.open_dataset(D / "l63v2_val_seed42.nc").isel(ic=slice(0, 50))
    truth_bg = lagged_background(ds, L63Model().step, "truth")
    e = float(np.abs(truth_bg.members.values - ds.x_true.values[:, None, 0]).max())
    report("true-model lagged members collapse onto x_true(t0)", e, "< 1e-9", e < 1e-9)
    bg = xr.open_dataset(D / "l63v2_val_seed42_bg_f1.nc")
    f1 = L63Model(L63Params(rho=27.5))
    manual = f1.final_state(torch.tensor(xr.open_dataset(D / "l63v2_val_seed42.nc").x_true_past.values[:, 50]), 50)
    e = float(np.abs(bg.x0_b.values - manual.numpy()).max())
    report("F1 first guess == 50-step F1 forecast from x_true(t0 - 0.5 TU)", e, "== 0", e == 0)
    b = bg.B.values
    sym = float(np.abs(b - b.transpose(0, 2, 1)).max())
    mineig = float(np.linalg.eigvalsh(b).min())
    report("F1 B symmetric and positive definite", mineig, f"> 0 (asym {sym:.1e})", sym < 1e-12 and mineig > 0)
    for name in ("f1", "rho275"):
        bgx = xr.open_dataset(D / f"l63v2_test_seed42_bg_{name}.nc")
        ok = int(bgx.tau_b) == 50 and np.array_equal(bgx.lag_steps.values, np.arange(10, 101, 10))
        report(f"{name}: tau_b = 50, lags 10..100", float(bgx.tau_b), "== 50", ok)


if __name__ == "__main__":
    for name, fn in [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]:
        fn()
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
