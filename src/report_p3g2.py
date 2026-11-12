"""
P3 and G2 for E1 (classical) vs E2-B (Meta-B delta, locked config, 3 seeds); strong constraint, full obs, M1.

Usage (project root):  python src/report_p3g2.py [--splits test test_ood]
Writes tmp/results/e2b/p3g2_<split>.{csv,md} (per-method medians) and prints the paired tests.

Definitions (per case; medians over the split's 200 ICs; E2-B: per-seed medians, mean +- std over seeds):
  P3          fraction of the analysis increment d = x0_a - x0_b along the unstable BLV b1(t0)   (North Star)
  P3_12       same, in span(b1, b2) (unstable + neutral)
  err0_u      |<x0_a - x0_true, b1(t0)>|           unstable component of the IC analysis error
  errL / errL_u / errL_s
              analysis error at the END of the window e_L = x_a(L) - x_true(L) (x_a from x0_a with M1): norm, and
              |<e_L, b1(L)>|, |<e_L, b3(L)>|  — the forecast starts from x_a(L), so errL_u drives forecast growth
  G2_cos      mean over the case's moves of cos(step, -grad J_M1) ; G2_cos10 = first 10 moves only
  G2_dec      fraction of moves that decrease the true cost J_M1
BLVs: QR iteration of the exact discrete TL model along the reproduced 1000-step spin-up (data/generate replay,
checked bit-identical to x_true[:, 0]) and on along the truth window; checks printed (exponents, invariance).
Paired tests: Wilcoxon signed-rank, E2-B (per-case mean over the 3 seeds) vs E1 Adagrad.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from scipy.stats import wilcoxon

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dameta.da import CostConfig, window_trajectory  # noqa: E402
from dameta.dynamics import L63Model, L63Params  # noqa: E402
from dameta.eval.lyapunov import blv_along, project_abs, project_fraction, spinup_history  # noqa: E402
from report_e1 import to_markdown  # noqa: E402

torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "tmp/results"


def blvs(ds: xr.Dataset, model: L63Model, L: int):
    hist = spinup_history(ds)
    xt = torch.tensor(ds.x_true.values)
    gap = (hist[:, -1] - xt[:, 0]).abs().max().item()
    assert gap == 0.0, f"spin-up replay does not reproduce x_true[:, 0] (max diff {gap})"
    q, logs, _ = blv_along(model.step, hist, at=[hist.shape[1] - 1])
    b_t0 = q[hist.shape[1] - 1]
    qL, logsL, _ = blv_along(model.step, xt[:, : L + 1], at=[L], q0=b_t0)
    b_L = qL[L]
    dt = ds.attrs["dt"]
    lam = (logs / ((hist.shape[1] - 1) * dt)).mean(0).numpy()
    # invariance check: J(x_t0) b1(t0) must be parallel to b1(t0 + 1)
    j0 = torch.func.vmap(torch.func.jacrev(model.step))(xt[:, 0])
    v = (j0 @ b_t0[:, :, :1])[:, :, 0]
    q1, _, _ = blv_along(model.step, xt[:, :2], at=[1], q0=b_t0)
    inv = torch.nn.functional.cosine_similarity(v, q1[1][:, :, 0], dim=1).abs().min().item()
    print(f"  BLV checks: replay gap {gap:.1e}; exponents (spin-up mean) {np.round(lam, 3)}; "
          f"min |cos(J b1, b1(t+1))| {inv:.12f}")
    return b_t0.numpy(), b_L.numpy()


def g2(rec: xr.Dataset) -> dict:
    evals, conv = rec.n_grad_evals.values, rec.converged.values
    moves = np.where(conv, evals - 1, evals)
    cos, j = rec.cos_step_grad.values, rec.J_iter.values
    k = np.arange(cos.shape[1])[None, :]
    act = k < moves[:, None]
    dec = (j[:, 1:] < j[:, :-1]) & act
    with np.errstate(invalid="ignore"):
        g_all = np.nansum(np.where(act, cos, 0), 1) / np.maximum(act.sum(1), 1)
        a10 = act & (k < 10)
        g_10 = np.nansum(np.where(a10, cos, 0), 1) / np.maximum(a10.sum(1), 1)
    return {"G2_cos": g_all, "G2_cos10": g_10, "G2_dec": dec.sum(1) / np.maximum(act.sum(1), 1)}


def per_case(rec: xr.Dataset, ds: xr.Dataset, b_t0, b_L, model, ccfg) -> dict:
    x0a, x0b, xt = rec.x0_a.values, ds.x0_b.values, ds.x_true.values
    L = ccfg.window
    with torch.no_grad():
        xaL = window_trajectory(model.step, torch.tensor(x0a), ccfg)[:, L].numpy()
    eL = xaL - xt[:, L]
    d = x0a - x0b
    out = {"P3": project_fraction(d, b_t0, 1), "P3_12": project_fraction(d, b_t0, 2),
           "err0": np.linalg.norm(x0a - xt[:, 0], axis=1), "err0_u": project_abs(x0a - xt[:, 0], b_t0, 0),
           "errL": np.linalg.norm(eL, axis=1), "errL_u": project_abs(eL, b_L, 0), "errL_s": project_abs(eL, b_L, 2),
           "fc_1TU": rec.rmse_fc.sel(lead=100).values}
    out.update(g2(rec))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["test", "test_ood"])
    a = ap.parse_args()
    ccfg = CostConfig()
    for split in a.splits:
        print(f"== {split}")
        ds = xr.open_dataset(ROOT / f"tmp/data/l63_{split}_seed42.nc")
        model = L63Model(L63Params(sigma=ds.attrs["sigma"], rho=ds.attrs["rho"], beta=ds.attrs["beta"],
                                   dt=ds.attrs["dt"]))
        b_t0, b_L = blvs(ds, model, ccfg.window)
        recs = {"E1 adagrad": [RES / f"e1/strong_full_adagrad_lr0.1_{split}.nc"],
                "E1 gd": [RES / f"e1/strong_full_gd_lr0.001_{split}.nc"],
                "E2-B": [RES / f"e2b/delta_strong_full_adagrad_lr0.1_C1_s{s}_{split}.nc" for s in range(3)]}
        cases, rows = {}, []
        for name, paths in recs.items():
            per = [per_case(xr.open_dataset(p), ds, b_t0, b_L, model, ccfg) for p in paths]
            cases[name] = {k: np.mean([p[k] for p in per], 0) for k in per[0]}       # per-case seed mean
            meds = pd.DataFrame([{k: float(np.nanmedian(v)) for k, v in p.items()} for p in per])
            row = {"method": name, "n_seeds": len(per)}
            for k in meds.columns:
                row[k] = meds[k].mean()
                if len(per) > 1:
                    row[k + "_sd"] = meds[k].std()
            rows.append(row)
        df = pd.DataFrame(rows)
        cols = ["method", "n_seeds"] + [c for c in df.columns if c not in ("method", "n_seeds") and
                                        not c.endswith("_sd")] + [c for c in df.columns if c.endswith("_sd")]
        df = df[cols]
        tests = []
        for k in ("P3", "P3_12", "err0", "err0_u", "errL", "errL_u", "errL_s", "fc_1TU", "G2_cos", "G2_dec"):
            x, y = cases["E2-B"][k], cases["E1 adagrad"][k]
            ok = np.isfinite(x) & np.isfinite(y)
            p = wilcoxon(x[ok], y[ok]).pvalue
            tests.append({"metric": k, "median E2-B - Adagrad": float(np.median(x[ok] - y[ok])),
                          "frac E2-B > Adagrad": float((x[ok] > y[ok]).mean()), "wilcoxon_p": p})
        tdf = pd.DataFrame(tests)
        out = RES / "e2b"
        df.to_csv(out / f"p3g2_{split}.csv", index=False)
        tdf.to_csv(out / f"p3g2_tests_{split}.csv", index=False)
        md = to_markdown(df) + "\n\nPaired (per case, E2-B seed mean vs E1 Adagrad):\n\n" + to_markdown(tdf) + "\n"
        (out / f"p3g2_{split}.md").write_text(md)
        print(md)


if __name__ == "__main__":
    main()
