"""
M8-R report (pilot 2026-11-26): strong / weak 4D-Var landscape on the model rectangle, shared truth-model first guess
and B per truth, chi2-tuned scalar Q, perfect-model reference (strong only).

Per run (tmp/results/m8r/<sub>/<truth>/<model>/bice_<c>_<p>_<method>_lr<lr>_<split>.nc), per case:
    sol_deg   |x0_a - x_true(0)|              traj_deg  window + forecast trajectory error   fc1   rmse_fc @ 1 TU
    lam_min_a, lam_max_a, kappa_H_a (PD only), n_neg_a, cos_truth_a   Hessian / gradient at the analysis
    nonpd_it  fraction of recorded iterates with a non-PD Hessian      cos_it  mean cos_truth over the iterates
    evals     n_grad_evals                                             (diverged cases -> NaN, counted)
Degradation Δ = model − reference, paired per case (same truth / obs), mean ± std over cases:
    truth L1: F1−L1, F2−L1 (misspecification), E1−L1, E2−L1 (total), E1−F1, E2−F2 (approximation on L1 obs)
    truth F1: E1−F1; truth F2: E2−F2 (approximation, isolated)
The perfect-model reference is strong only (Q = 0): weak runs of the models are compared with it too; E−F pairs
compare the same scheme (strong − strong, weak − weak).

Usage (project root):  python src/report_rect_4dv.py [smoke|test]  -> tmp/results/m8r/rect_<sub>.{md,csv}
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src/scripts"))
from m8_lists import METHODS, PATTERNS, STAGES  # noqa: E402

METRICS = ("sol_deg", "traj_deg", "fc1", "lam_min_a", "lam_max_a", "kappa_H_a", "n_neg_a", "cos_truth_a",
           "nonpd_it", "cos_it", "evals")
SHOW = ("sol_deg", "traj_deg", "fc1", "lam_min_a", "kappa_H_a", "nonpd_it", "cos_it", "evals")
PAIRS = {"L1": [("f1", "l1"), ("f2", "l1"), ("e1", "l1"), ("e2", "l1"), ("e1", "f1"), ("e2", "f2"),
                ("e1b", "l1"), ("e2b", "l1"), ("e1b", "f1"), ("e2b", "f2")],
         "F1": [("e1", "f1"), ("e1b", "f1")], "F2": [("e2", "f2"), ("e2b", "f2")]}   # e1b / e2b: bad emulators


def per_case(f: Path) -> pd.DataFrame:
    ds = xr.load_dataset(f)
    div = ds.diverged.values
    nanif = lambda a: np.where(div, np.nan, a)                                      # noqa: E731
    nn = ds.n_neg.values
    rec = np.isfinite(nn)
    nonpd = np.where(rec.sum(1) > 0, ((nn > 0) & rec).sum(1) / np.maximum(rec.sum(1), 1), np.nan)
    d = {"sol_deg": ds.sol_deg.values, "traj_deg": ds.traj_deg.values,
         "fc1": ds.rmse_fc.sel(lead=100).values, "evals": nanif(ds.n_grad_evals.values.astype(float)),
         "nonpd_it": nanif(nonpd), "cos_it": nanif(np.nanmean(ds.cos_truth.values, 1))}
    for k in ("lam_min_a", "lam_max_a", "kappa_H_a", "n_neg_a", "cos_truth_a"):
        d[k] = nanif(ds[k].values.astype(float))
    out = pd.DataFrame(d)
    out["ic"] = ds.ic.values
    out["diverged"] = div
    return out


def ms(a) -> str:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return "—" if a.size == 0 else f"{a.mean():.3g} ± {a.std():.2g}"


def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else "test"
    res = ROOT / "tmp/results/m8r" / sub
    runs = {}
    for tag, (ddir, bg, ref, models) in STAGES.items():
        for m in (ref,) + models:
            for f in sorted((res / tag / m).glob("bice_*.nc")):
                c, rest = f.stem.split("_", 2)[1], f.stem.split("_", 2)[2]
                p = "x_only" if rest.startswith("x_only") else "full"
                meth = next(x for x in METHODS if f"_{x}_" in f"_{rest}")
                runs[(tag, m, c, p, meth)] = per_case(f)
    rows, drows = [], []
    for (tag, m, c, p, meth), df in runs.items():
        r = {"truth": tag, "model": m, "constraint": c, "pattern": p, "method": meth, "n": len(df),
             "n_div": int(df.diverged.sum())}
        for k in METRICS:
            v = df[k].to_numpy(float)
            r[f"{k}_mean"], r[f"{k}_std"] = np.nanmean(v), np.nanstd(v)
        rows.append(r)
    for tag, pairs in PAIRS.items():
        for a, b in pairs:
            for c in ("strong", "weak"):
                cb = "strong" if b == STAGES[tag][2] else c            # reference: strong only
                for p in PATTERNS:
                    for meth in METHODS:
                        A, B = runs.get((tag, a, c, p, meth)), runs.get((tag, b, cb, p, meth))
                        if A is None or B is None:
                            continue
                        r = {"truth": tag, "pair": f"{a}-{b}", "constraint": c, "pattern": p, "method": meth,
                             "n": len(A)}
                        for k in METRICS:
                            dv = A[k].to_numpy(float) - B[k].to_numpy(float)
                            r[f"d_{k}_mean"], r[f"d_{k}_std"] = np.nanmean(dv), np.nanstd(dv)
                            r[f"d_{k}_n"] = int(np.isfinite(dv).sum())
                        drows.append(r)
    out = ROOT / "tmp/results/m8r"
    pd.DataFrame(rows).to_csv(out / f"rect_{sub}.csv", index=False)
    pd.DataFrame(drows).to_csv(out / f"rect_{sub}_delta.csv", index=False)

    md = [f"# M8-R — 4D-Var landscape on the model rectangle ({sub})", "",
          "Shared truth-model first guess / static B (× b_infl) per truth; Q = q_m I (chi2-tuned, "
          "src/select_q.py); reference = perfect model, strong. Values: mean ± std over cases (diverged excluded).",
          "kappa_H_a over PD analyses only; nonpd_it = share of iterates with a non-PD Hessian; cos_it = mean "
          "cos_truth over iterates; fc1 = forecast RMSE at 1 TU.", ""]
    for tag in STAGES:
        for p in PATTERNS:
            md += [f"## truth {tag}, {p}", "", "| model | scheme | method | div | " + " | ".join(SHOW) + " |",
                   "|---" * (len(SHOW) + 4) + "|"]
            for (t, m, c, pp, meth), df in runs.items():
                if t == tag and pp == p:
                    md.append(f"| {m} | {c} | {meth} | {int(df.diverged.sum())}/{len(df)} | "
                              + " | ".join(ms(df[k]) for k in SHOW) + " |")
            md += ["", f"### Δ (paired per case), truth {tag}, {p}", "",
                   "| pair | scheme | method | " + " | ".join(f"Δ {k}" for k in SHOW) + " |",
                   "|---" * (len(SHOW) + 3) + "|"]
            for r in drows:
                if r["truth"] == tag and r["pattern"] == p:
                    md.append(f"| {r['pair']} | {r['constraint']} | {r['method']} | " + " | ".join(
                        f"{r[f'd_{k}_mean']:.3g} ± {r[f'd_{k}_std']:.2g}" if np.isfinite(r[f"d_{k}_mean"]) else "—"
                        for k in SHOW) + " |")
            md.append("")
    (out / f"rect_{sub}.md").write_text("\n".join(md))
    print(f"wrote {out}/rect_{sub}.{{md,csv}}, rect_{sub}_delta.csv ({len(runs)} runs)")


if __name__ == "__main__":
    main()
