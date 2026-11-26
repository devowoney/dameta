"""
M8 landscape diagnostics (src/dameta/eval/landscape.py): Hessian, spectrum rules, truth-correction cosine.

Plain-python test script. Run from the project root (needs tmp/data D1 v2 + backgrounds, tmp/emulators):
    python src/tests/test_landscape.py
"""

import sys
from pathlib import Path

import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dameta.da import CostConfig, FourDVarCost  # noqa: E402
from dameta.data.background import load_hybrid_background  # noqa: E402
from dameta.dynamics.models import build_model, load_registry  # noqa: E402
from dameta.eval.landscape import cos_truth, hessian_batch, spectrum  # noqa: E402

torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "tmp/data"
N = 6
RESULTS = []


def report(name, value, thr, ok):
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<64s} value={value:.3e}  ({thr})")
    assert ok, name


class Quad:
    """J(c) = 1/2 c^T A c per case (exact Hessian A)."""

    def __init__(self, a):
        self.a = a

    def unpack(self, c):
        return (c,)

    def __call__(self, c):
        return 0.5 * torch.einsum("ni,nij,nj->n", c, self.a, c)


def fd_hessian(cost, c, h=1e-5):
    """Central FD of the autograd gradient: column i = (g(c + h e_i) - g(c - h e_i)) / 2h."""
    cols = []
    for i in range(c.shape[1]):
        cp, cm = c.clone(), c.clone()
        cp[:, i] += h
        cm[:, i] -= h
        cols.append((cost.grad(cp)[1] - cost.grad(cm)[1]) / (2 * h))
    hh = torch.stack(cols, 2)
    return 0.5 * (hh + hh.transpose(1, 2))


def problem(model, weak, pattern="full"):
    ds = xr.load_dataset(D / "l63v2_test_seed42.nc").isel(ic=slice(0, N))
    x0_b, _, b_inv, _ = load_hybrid_background(D, "test", model, 42)
    step, _, _ = build_model(model, load_registry(), ROOT / "tmp/emulators")
    ccfg = CostConfig(window=int(ds.attrs["window"]), obs_every=int(ds.attrs["obs_every"]), sigma_o2=2.0, sigma_q=0.1)
    cost = FourDVarCost(step, torch.tensor(ds.y_obs.values), torch.tensor(ds.obs_mask.sel(pattern=pattern).values),
                        x0_b[:N], ccfg, weak=weak, b_inv=b_inv[:N])
    return ds, cost, x0_b[:N]


def main():
    # 1. analytic quadratic: exact Hessian and spectrum rules
    torch.manual_seed(0)
    q = torch.linalg.qr(torch.randn(3, 3, 3))[0]
    lam = torch.tensor([[1.0, 2.0, 10.0], [-1.0, 3.0, 4.0], [-2.0, -1.0, 5.0]])
    a = q @ torch.diag_embed(lam) @ q.transpose(1, 2)
    h = hessian_batch(Quad(a), torch.randn(3, 3))
    report("quadratic: |H - A|_max", float((h - a).abs().max()), "< 1e-12", float((h - a).abs().max()) < 1e-12)
    sp = spectrum(h)
    report("spectrum: kappa case 0 = 10", float(sp["kappa_H"][0]), "== 10", abs(float(sp["kappa_H"][0]) - 10) < 1e-9)
    report("spectrum: kappa NaN if lambda_min <= 0", float(sp["kappa_H"][1:].isnan().sum()), "== 2",
           bool(sp["kappa_H"][1:].isnan().all()))
    report("spectrum: n_neg = (0, 1, 2)", float((sp["n_neg"] - torch.tensor([0.0, 1.0, 2.0])).abs().sum()), "== 0",
           bool((sp["n_neg"] == torch.tensor([0.0, 1.0, 2.0])).all()))
    report("spectrum: lambda_min / lambda_max", float((sp["lam_min"] - lam[:, 0]).abs().max()), "< 1e-12",
           float((sp["lam_min"] - lam[:, 0]).abs().max()) < 1e-12 and float((sp["lam_max"] - lam[:, 2]).abs().max()) < 1e-12)

    # 2. cos_truth: gradient of |x - x_t|^2 / 2 -> -g points to the truth (cos = 1), orthogonal -> 0
    xt, x = torch.randn(4, 3), torch.randn(4, 3)
    c1 = cos_truth(x, x - xt, xt)
    report("cos_truth: quadratic bowl at the truth = 1", float((c1 - 1).abs().max()), "< 1e-12", float((c1 - 1).abs().max()) < 1e-12)
    d = xt - x
    orth = torch.linalg.cross(d, torch.randn(4, 3))
    report("cos_truth: orthogonal gradient = 0", float(cos_truth(x, orth, xt).abs().max()), "< 1e-12",
           float(cos_truth(x, orth, xt).abs().max()) < 1e-12)

    # 3. 4D-Var Hessian vs central FD of the adjoint gradient (F1 / F2 L63, E1 / E2 emulator; strong and weak)
    for model, weak in (("f1", False), ("f1", True), ("f2", False), ("e1", False), ("e1", True), ("e2", False)):
        ds, cost, x0_b = problem(model, weak)
        c = cost.pack(x0_b, torch.zeros(N, cost.cfg.n_obs - 1, 3) if weak else None) + 0.1 * torch.randn(N, 3 * cost.cfg.n_obs if weak else 3)
        hx = hessian_batch(cost, c)
        hf = fd_hessian(cost, c)
        err = float(((hx - hf).flatten(1).norm(dim=1) / hx.flatten(1).norm(dim=1)).max())
        report(f"{model} {'weak' if weak else 'strong'}: rel |H_ad - H_fd| (d = {c.shape[1]})", err, "< 1e-5", err < 1e-5)

    print(f"{sum(RESULTS)}/{len(RESULTS)} PASS")


if __name__ == "__main__":
    main()
