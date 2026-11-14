"""
D1 v2 classical baselines report (M5 blueprint step 4): model-error-isolated BICE / BPE and adjoint-free gradients.

Usage (project root):  python src/report_v2.py
Reads tmp/results/v2_{truth,f1,emulator,f1_bpe,f1_fd,f1_hybrid}/*.nc; writes tmp/results/v2_summary.{csv,md}.

Per record (medians over cases; 200 ICs per split):
  evals_1pct   gradient evaluations to J <= 1.01 J*, J* = best final J of GD / Adagrad of the SAME estimation
               model, problem, constraint and obs pattern (cost-to-target, as report_ic)
  ic / ic_fg   IC RMSE of the analysis / of the first guess (vs truth)
  fc_model     forecast RMSE @1 TU with the estimation model (BPE: estimated parameters) — operational skill
  fc_truth     forecast RMSE @1 TU with the TRUE model from the analysis — IC quality
  P1           true-model cost of the analysis
  gap_vs_E1    paired median (fc_truth - fc_truth of E1 with the same constraint / pattern / method)
  BPE          median estimated sigma, rho, beta (truth 10, 28, 2.667; prior 10, 27.5, 2.667)
  adjoint-free fwd_per_gradient (F1 forward runs per gradient) and |x0_a - x0_a(exact)| median
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_e1 import to_markdown  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "tmp/results"
GROUPS = {"v2_truth": "E1 (truth)", "v2_f1": "F1 (rho 27.5)", "v2_emulator": "F2 (emulator)",
          "v2_f1_bpe": "F1 BPE", "v2_f1_fd": "F1 FD-grad", "v2_f1_hybrid": "F1 hybrid-grad"}


def evals_to(j, j_star, tol=0.01):
    hit = j <= (1 + tol) * j_star[:, None]
    return np.where(hit.any(1), hit.argmax(1), np.nan)


def main():
    recs = {}
    for g in GROUPS:
        for p in sorted((RES / g).glob("*.nc")):
            ds = xr.load_dataset(p)
            c = json.loads(ds.attrs["config"])
            recs[(g, c["problem"], c["constraint"], c["pattern"], c["method"], c["split"])] = ds
    rows = []
    for key, ds in sorted(recs.items()):
        g, pb, con, pat, met, sp = key
        base = g if g in ("v2_truth", "v2_f1", "v2_emulator", "v2_f1_bpe") else "v2_f1"
        cands = [recs[k] for k in recs if k[0] == base and k[1:4] == (pb, con, pat) and k[5] == sp]
        j_star = np.nanmin(np.stack([d.J_iter.values[:, -1] for d in cands] + [ds.J_iter.values[:, -1]]), 0)
        ev = evals_to(ds.J_iter.values, j_star)
        r = {"split": sp, "estimation": GROUPS[g], "problem": pb, "constraint": con, "obs": pat, "method": met,
             "diverged": int(ds.diverged.sum()), "converged": float(ds.converged.mean()),
             "evals_1pct": float(np.nanmedian(ev)), "reached": float(np.isfinite(ev).mean()),
             "ic": float(np.nanmedian(ds.rmse_x0)), "ic_fg": float(np.median(ds.rmse_x0_b)),
             "fc_model": float(np.nanmedian(ds.rmse_fc.sel(lead=100))),
             "fc_truth": float(np.nanmedian(ds.rmse_fc_m1.sel(lead=100))),
             "fc_fg": float(np.median(ds.rmse_fc_b.sel(lead=100))), "P1": float(np.nanmedian(ds.J_true_final))}
        e1 = recs.get(("v2_truth", "bice", con, pat, met, sp))
        if e1 is not None and g != "v2_truth":
            r["gap_vs_E1"] = float(np.nanmedian(ds.rmse_fc_m1.sel(lead=100).values - e1.rmse_fc_m1.sel(lead=100).values))
        if "theta_a" in ds:
            r["sigma_a"], r["rho_a"], r["beta_a"] = (float(v) for v in np.nanmedian(ds.theta_a.values, 0))
            r["rho_IQR"] = "%.2f-%.2f" % tuple(np.nanquantile(ds.theta_a.values[:, 1], [0.25, 0.75]))
        if "gradient" in ds.attrs:
            ex = recs.get(("v2_f1", pb, con, pat, met, sp))
            r["fwd_per_grad"] = int(ds.attrs["fwd_per_gradient"])
            if ex is not None:
                r["dx0_vs_exact"] = float(np.median(np.linalg.norm(ds.x0_a.values - ex.x0_a.values, axis=1)))
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "v2_summary.csv", index=False)
    md = "## D1 v2 classical baselines (medians; fc / ic RMSE vs truth)\n\n" + to_markdown(df) + "\n"
    (RES / "v2_summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
