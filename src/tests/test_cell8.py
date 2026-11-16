"""
M5 cell 8 verification — adjoint-free learned gradient (forward run -> grad J_o): MLP and adjoint-structured versions.

Plain-python test script. Run from the project root (after gen_data_v2.py and tune_hybrid_b.py):
    python src/tests/test_cell8.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dameta.da import CostConfig, FourDVarCost, run_fourdvar  # noqa: E402
from dameta.data.background import load_hybrid_background  # noqa: E402
from dameta.dynamics import L63Model, L63Params  # noqa: E402
from dameta.dynamics.emulator import L63Emulator  # noqa: E402
from dameta.meta import control_stats  # noqa: E402
from dameta.meta.learned_gradient import (LGConfig, adjoint_recursion, analytic_grad, build_lg,  # noqa: E402
                                          collect_pairs, collect_sampled_pairs, forward_run, jo_grad,
                                          run_learned_gradient)

torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "tmp/data"
CFG = CostConfig(window=110, obs_every=10)
F1 = L63Model(L63Params(rho=27.5))
EMU = L63Emulator.load(ROOT / "tmp/emulators/rho275.pt")
for _p in EMU.parameters():
    _p.requires_grad_(False)
RESULTS = []


def report(name, value, thr, ok):
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<70s} value={value:.3e}  ({thr})")
    assert ok, name


def costs(split, n, weak=False, step=F1.step):
    ds = xr.load_dataset(D / f"l63v2_{split}_seed42.nc").isel(ic=slice(0, n))
    x0_b, _, b_inv, _ = load_hybrid_background(D, split, "f1")
    y, mask = torch.tensor(ds.y_obs.values), torch.tensor(ds.obs_mask.sel(pattern="full").values)
    mk = lambda s: FourDVarCost(s, y, mask, x0_b[:n], CFG, weak=weak, b_inv=b_inv[:n])  # noqa: E731
    return mk(step), mk(EMU.step), x0_b[:n]


def ctrl(cost, x0_b):
    return cost.pack(x0_b, torch.zeros(len(x0_b), CFG.n_obs - 1, 3) + 0.01) if cost.weak else x0_b.clone()


def stats():
    return control_stats(torch.tensor(xr.load_dataset(D / "l63v2_train_seed42.nc").x_true.values), CFG.n_obs, 0.1,
                         False)


def test_forward_and_decomposition():
    for weak in (False, True):
        cf, _, xb = costs("val", 8, weak)
        c = ctrl(cf, xb)
        f = forward_run(cf, c)
        e1 = float((f["J"] - cf(*cf.unpack(c))).abs().max())
        g = analytic_grad(cf, c) + jo_grad(cf, c)
        e2 = float(((g - cf.grad(c)[1]).norm(dim=1) / cf.grad(c)[1].norm(dim=1)).max())
        report(f"forward J == cost J; grad_b(+q) analytic + grad_o == full grad ({'weak' if weak else 'strong'})",
               max(e1, e2), "< 1e-12, obs 72 features", e1 < 1e-12 and e2 < 1e-12 and f["obs"].shape[1] == 72)


def test_adjoint_recursion_exact():
    """With the EXACT one-interval tangent-linear operators of F1, the recursion == autodiff grad J_o."""
    def interval(x):
        for _ in range(CFG.obs_every):
            x = F1.step(x)
        return x
    jac = torch.func.vmap(torch.func.jacrev(interval))
    for weak in (False, True):
        cf, _, xb = costs("val", 8, weak)
        c = ctrl(cf, xb)
        f = forward_run(cf, c)
        n_int = f["x_obs"].shape[1] - 1
        a = torch.stack([jac(f["x_obs"][:, i]) for i in range(n_int)], 1)
        g = adjoint_recursion(a, f["r"], weak)
        g_ex = jo_grad(cf, c)
        rel = float(((g - g_ex).norm(dim=1) / g_ex.norm(dim=1)).max())
        report(f"adjoint recursion with exact interval Jacobians == autodiff grad J_o ({'weak' if weak else 'strong'})",
               rel, "< 1e-10", rel < 1e-10)


class _Oracle:
    """Stand-in learner returning the EXACT grad J_o of F (tests the inference loop)."""
    def __init__(self, cost):
        self.cost = cost

    def grad_obs(self, c, k, fwd, weak=False):
        return jo_grad(self.cost, c)


def test_oracle_reproduces_classical():
    cf, _, xb = costs("val", 8)
    ref = run_fourdvar(cf, xb, "adagrad", 0.3, 150, 1e-6)
    out = run_learned_gradient(cf, xb, _Oracle(cf), "adagrad", 0.3, 150, 1e-6)
    err = float(np.abs(ref["J_iter"] - out["J_iter"]).max())
    same = bool((ref["n_grad_evals"] == out["n_fwd_evals"]).all()) and int(out["n_grad_evals"].sum()) == 0
    report("oracle grad_o: loop == classical Adagrad; 1 fwd per iteration, 0 grads", err, "< 1e-9", err < 1e-9 and same)


def test_trajectory_pairs():
    cf, ce, xb = costs("train", 16)
    p = collect_pairs(cf, ce, xb, "adagrad", 0.3, max_iter=40)
    k0 = p["k"] == 0
    e = max(float((p["x"][k0] - xb).abs().max()), float((p["g"][k0] - jo_grad(ce, xb)).abs().max()))
    report("trajectory pairs: k=0 states == first guess; targets == emulator grad J_o", e, "== 0", e == 0)


def test_sampled_pairs_and_adjoint_fit():
    cfg = LGConfig(arch="adjoint", data="sampled", samples_per_case=32, traj_stride=0, epochs=20, batch_size=256)
    cf, ce, xb = costs("train", 64)
    p = collect_sampled_pairs(cf, ce, xb, "adagrad", 0.3, cfg)
    i0 = slice(0, 32)                                         # the 32 samples of case 0
    b = torch.linalg.inv(cf.b_inv[0])
    emp = torch.cov((p["x"][i0] - xb[0]).T)
    ratio = float(torch.trace(emp) / torch.trace(b))
    e = float((p["g"][:64] - jo_grad(FourDVarCost(EMU.step, ce.y[:2].repeat_interleave(32, 0), ce.mask,
                                                  ce.x0_b[:2].repeat_interleave(32, 0), CFG,
                                                  b_inv=ce.b_inv[:2].repeat_interleave(32, 0)), p["x"][:64])).abs().max())
    report("sampled pairs: 64 x 32 controls, B-shaped spread (tr ratio), emulator targets", ratio,
           "0.3 < ratio < 6, targets exact", p["x"].shape[0] == 64 * 32 and 0.3 < ratio < 6 and e == 0)
    cv, cev, xbv = costs("val", 16)
    pv = collect_sampled_pairs(cv, cev, xbv, "adagrad", 0.3, cfg)
    m, s = stats()
    lg = build_lg(3, CFG.n_obs, cfg, m, s, p)
    hist = lg.fit(p, pv)
    print("       val cos", [round(h["val_cos_med"], 3) for h in hist[::4]], " val rel err",
          [round(h["val_relerr_med"], 3) for h in hist[::4]])
    best = max(h["val_cos_med"] for h in hist)
    report("adjoint-structured fit smoke: val median cos(g_o^, emulator grad_o)", best, "> 0.9", best > 0.9)


if __name__ == "__main__":
    for name, fn in [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]:
        fn()
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
