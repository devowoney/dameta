"""
M5 meta-grid report on D1 v2 (strong / full): every method against an explicit REFERENCE on the same false model.

Usage (project root):  python src/report_m5.py [--splits test test_ood]
Reads tmp/results/m5_{f1,emulator}/*_c<cell>_s<seed>_<split>.nc, the classical D1 v2 records
(tmp/results/v2_{f1,emulator,truth,f1_fd,f1_hybrid}) and tmp/results/grad_cost_ratio.json (measured cost of one
gradient evaluation and of one emulator gradient in forward runs, src/measure_grad_cost.py).
Writes tmp/results/m5_summary_<split>.{csv,md}.

Reference = classical 4D-Var, Adagrad, exact adjoint, same false model and case (the strongest classical baseline).
  cost       model runs to reach J <= 1.01 J* (J* = best final J of classical GD / Adagrad, same model and case), in
             FORWARD-RUN EQUIVALENTS of the false model: gradient evaluation = measured ratio (3.7 F1, 2.5 F2), one
             forward run = 1, FD gradient = 2d + 1, hybrid gradient = 1 F1 forward + 1 emulator gradient (measured);
             network evaluations are not counted (negligible next to a model run); median over cases reaching it
  speed-up   reference cost / method cost (x 2.0 = twice as cheap); "n/a" when < 50 % of cases reach the target
  reached, IC RMSE, true-model forecast RMSE @1 TU: shown as method | reference; worse basins = cases ending with a
             final cost > 1.1 x the reference's. Meta cells: mean over 3 training seeds.
"""

import argparse
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
CELLS = {"c1": ("1", "learned optimizer, unsupervised, gradient (Δ delta)", "grad"),
         "c2": ("2", "learned optimizer, supervised, gradient (5-step lookahead)", "grad"),
         "c3": ("3", "learned optimizer, unsupervised, no gradient", "fwd"),
         "c3o": ("3+obs", "  same + observations as input", "fwd"),
         "c4": ("4", "learned optimizer, supervised, no gradient", "fwd"),
         "c4o": ("4+obs", "  same + observations as input", "fwd"),
         "c5": ("5", "learned optimization (K 20 unroll), gradient", "grad"),
         "c7": ("7", "learned optimization (K 20 unroll), no gradient", "none"),
         "c7o": ("7+obs", "  same + observations as input", "none"),
         "c8": ("8", "adjoint-free learned gradient", "fwd")}


def load(path):
    try:
        return xr.load_dataset(path)
    except (OSError, FileNotFoundError, TypeError):
        return None


def iters_to(j, j_star, tol=0.01):
    hit = j <= (1 + tol) * j_star[:, None]
    return np.where(hit.any(1), hit.argmax(1), np.nan)


def stats(ds, j_star, unit_cost, ref):
    it = iters_to(ds.J_iter.values, j_star)
    jm, jr = ds.J_iter.values[:, -1], ref.J_iter.values[:, -1]
    return {"cost": float(np.nanmedian(it * unit_cost)) if np.isfinite(it).any() else np.nan,
            "reached": float(np.isfinite(it).mean()), "ic": float(np.nanmedian(ds.rmse_x0)),
            "fc": float(np.nanmedian(ds.rmse_fc_m1.sel(lead=100))), "worse": float((jm > 1.1 * jr).sum())}


def fmt_speed(ref_cost, cost, reached, unit):
    if unit == "none":
        return "no model run"
    if reached < 0.5 or not np.isfinite(cost) or cost == 0:
        return f"n/a (reached {reached:.0%})"
    s = ref_cost / cost
    return f"x {s:.2f} " + ("faster" if s >= 1 else "(slower)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["test", "test_ood"])
    a = ap.parse_args()
    ratio = json.loads((RES / "grad_cost_ratio.json").read_text())
    for split in a.splits:
        rows = []
        for model in ("f1", "emulator"):
            r_g = ratio[f"{model}/strong"]["ratio"]
            cls = {m: load(next((RES / f"v2_{model}").glob(f"bice_strong_full_{m}_lr*_{split}.nc"))) for m in ("adagrad", "gd")}
            ref = cls["adagrad"]
            j_star = np.nanmin(np.stack([d.J_iter.values[:, -1] for d in cls.values()]), 0)
            R = stats(ref, j_star, r_g, ref)
            ref_name = "classical Adagrad (exact adjoint)"

            def add(cell, method, s, unit, seeds, sd=None):
                rows.append({"false model": "F1 (rho 27.5)" if model == "f1" else "F2 (emulator)", "cell": cell,
                             "method": method, "reference": ref_name, "seeds": seeds,
                             "cost (fwd-eq) method | ref": f"{s['cost']:.0f} | {R['cost']:.0f}",
                             "speed-up": fmt_speed(R["cost"], s["cost"], s["reached"], unit),
                             "reached method | ref": f"{s['reached']:.2f} | {R['reached']:.2f}",
                             "IC RMSE method | ref": f"{s['ic']:.3f}" + (f" ± {sd['ic']:.3f}" if sd else "")
                             + f" | {R['ic']:.3f}",
                             "fc true model @1TU method | ref": f"{s['fc']:.2f} | {R['fc']:.2f}",
                             "Δ fc vs ref": f"{100 * (s['fc'] / R['fc'] - 1):+.0f} %",
                             "worse basins": f"{s['worse']:.0f}"})

            add("-", "classical GD (exact adjoint)", stats(cls["gd"], j_star, r_g, ref), "grad", 1)
            if model == "f1":
                emu_g = ratio["emulator/strong"]["gradient_s"] / ratio["f1/strong"]["forward_s"]
                for g, label in (("fd", "classical Adagrad, finite-difference gradient (no adjoint)"),
                                 ("hybrid", "classical Adagrad, hybrid gradient (F1 cost + emulator adjoint)")):
                    p = next((RES / f"v2_f1_{g}").glob(f"bice_strong_full_adagrad_lr*_{split}.nc"), None)
                    d = load(p) if p else None
                    if d is not None:
                        unit = float(d.attrs["fwd_per_gradient"]) if g == "fd" else 1.0 + emu_g
                        add("-", label, stats(d, j_star, unit, ref), "fwd", 1)
            for c, (cell, label, unit) in CELLS.items():
                per = [s for s in (load(p) for p in sorted((RES / f"m5_{model}").glob(f"*_{c}_s[0-9]_{split}.nc")))
                       if s is not None]
                if not per:
                    continue
                u = r_g if unit == "grad" else (1.0 if unit == "fwd" else 0.0)
                df = pd.DataFrame([stats(d, j_star, u, ref) for d in per])
                add(cell, label, df.mean().to_dict(), unit, len(per), df.std().to_dict() if len(per) > 1 else None)
        out = pd.DataFrame(rows)
        path = RES / f"m5_summary_{split}"
        out.to_csv(path.with_suffix(".csv"), index=False)
        md = (f"## M5 meta grid on D1 v2 (strong / full) — {split}\n\nReference for every row: classical Adagrad with "
              f"the exact adjoint on the same false model. Cost = model runs to reach 1 % of J*, in forward-run "
              f"equivalents (gradient = {ratio['f1/strong']['ratio']:.1f} fwd on F1, "
              f"{ratio['emulator/strong']['ratio']:.1f} on F2).\n\n" + to_markdown(out) + "\n")
        path.with_suffix(".md").write_text(md)
        print(md)


if __name__ == "__main__":
    main()
