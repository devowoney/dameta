"""
M5 cost verification on D1 v2: case-specific hybrid B in J_b, BPE cost (control [x0, sigma, rho, beta]).

Plain-python test script. Run from the project root (after gen_data_v2.py and tune_hybrid_b.py):
    python src/tests/test_cost_v2.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dameta.da import CostConfig, FourDVarCost, ParamFourDVarCost  # noqa: E402
from dameta.data.background import load_hybrid_background  # noqa: E402
from dameta.dynamics import L63Model, L63Params  # noqa: E402

torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "tmp/data"
DS = xr.load_dataset(D / "l63v2_val_seed42.nc").isel(ic=slice(0, 16))
CFG = CostConfig(window=110, obs_every=10)
Y, MASK = torch.tensor(DS.y_obs.values), torch.tensor(DS.obs_mask.sel(pattern="full").values)
XT = torch.tensor(DS.x_true.values[:, 0])
F1 = L63Model(L63Params(rho=27.5))
THETA_B = torch.tensor([10.0, 27.5, 8.0 / 3.0])
THETA_STD = 0.1 * THETA_B
RESULTS = []


def report(name, value, thr, ok):
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<64s} value={value:.3e}  ({thr})")
    assert ok, name


def background():
    x0_b, b_h, b_inv, ab = load_hybrid_background(D, "val", "f1")
    return x0_b[:16], b_h[:16], b_inv[:16], ab


def fd_check(cost, c, n_dir=3, eps=1e-6):
    _, g = cost.grad(c)
    torch.manual_seed(0)
    worst = 0.0
    for _ in range(n_dir):
        v = torch.randn_like(c)
        jp, jm = cost(*cost.unpack(c + eps * v)), cost(*cost.unpack(c - eps * v))
        fd = (jp - jm) / (2 * eps)
        an = (g * v).sum(1)
        worst = max(worst, float(((fd - an).abs() / an.abs().clamp_min(1e-12)).max()))
    return worst


def test_hybrid_loader():
    x0_b, b_h, b_inv, ab = background()
    tuned = json.loads((D / "hybrid_B_seed42.json").read_text())["f1"]
    e = float((b_h @ b_inv - torch.eye(3)).abs().max())
    report("hybrid B uses tuned (alpha, s2); B_h B_h^-1 == I", e,
           f"alpha {ab['alpha']} == {tuned['alpha']}", ab["alpha"] == tuned["alpha"] and ab["s2"] == tuned["s2"] and e < 1e-10)


def test_binv_reduces_to_scalar():
    x0_b = background()[0]
    sb = CFG.sigma_b
    j0 = FourDVarCost(F1.step, Y, MASK, x0_b, CFG)(XT)
    j1 = FourDVarCost(F1.step, Y, MASK, x0_b, CFG, b_inv=torch.eye(3).expand(16, 3, 3) / sb**2)(XT)
    err = float((j0 - j1).abs().max())
    report("b_inv = I / sigma_b^2 reproduces the scalar J_b exactly", err, "< 1e-12", err < 1e-12)


def test_grad_hybrid():
    x0_b, _, b_inv, _ = background()
    cost = FourDVarCost(F1.step, Y, MASK, x0_b, CFG, b_inv=b_inv)
    rel = fd_check(cost, x0_b.clone())
    report("BICE (F1, hybrid B, window 110): gradient == finite difference", rel, "< 1e-5", rel < 1e-5)


def test_bpe_consistency_and_grad():
    x0_b, _, b_inv, _ = background()
    bpe = ParamFourDVarCost(Y, MASK, x0_b, CFG, THETA_B, THETA_STD, b_inv=b_inv)
    bice = FourDVarCost(F1.step, Y, MASK, x0_b, CFG, b_inv=b_inv)
    j_bpe, j_bice = bpe(XT, THETA_B.expand(16, 3)), bice(XT)
    err = float((j_bpe - j_bice).abs().max() / j_bice.abs().max())
    report("BPE at theta = F1 params == BICE with F1 (J_p = 0)", err, "< 1e-10", err < 1e-10)
    c = bpe.pack(x0_b)
    rel = fd_check(bpe, c)
    report("BPE gradient over [x0, sigma, rho, beta] == finite difference", rel, "< 1e-5", rel < 1e-5)


def test_bpe_identifiability():
    """The observations prefer the TRUE rho (28) to the biased prior value (27.5) from the true x0 (J_o only)."""
    x0_b, _, b_inv, _ = background()
    bpe = ParamFourDVarCost(Y, MASK, x0_b, CFG, THETA_B, THETA_STD, b_inv=b_inv)
    jo = lambda th: bpe.terms(XT, th.expand(16, 3))["Jo"]  # noqa: E731
    better = float((jo(torch.tensor([10.0, 28.0, 8 / 3])) < jo(THETA_B)).double().mean())
    report("J_o lower at true rho = 28 than at prior 27.5 (fraction of cases)", better, "> 0.8", better > 0.8)


if __name__ == "__main__":
    for name, fn in [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]:
        fn()
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
