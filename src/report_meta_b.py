"""
Consolidated Meta-B (delta, locked C1) report over all configurations: E2-B (M1) and E4-B (emulators) vs classical.

Usage (project root):  python src/report_meta_b.py [--splits test test_ood] [--variant delta|supervised]
Writes tmp/results/meta_b_summary_<split>.{csv,md} (delta) or meta_b_sup_summary_<split>.{csv,md} (supervised).
Supervised: "evals" = iterations = FORWARD runs (no adjoint), so its speed-up is not in the same unit as the
classical gradient evaluations (a gradient evaluation ~ 2-3 forward runs); cos(g_hat, grad) = median gradient
fidelity w.r.t. the gradient of the dynamics used (emulator gradient for E4-B).

Per configuration (dynamics, constraint, obs pattern) and split, Meta-B = 3 training seeds (mean +- sd of per-seed
medians over the 200 ICs); classical = Adagrad with the E1/E3-selected lr ON THE SAME DYNAMICS:
  evals       gradient evaluations to J <= 1.01 J* (J* = best final J of classical GD/Adagrad on the same dynamics
              and case — emulator J for E4-B, exactly as in report_ic.add_costs); median over cases that reach it
  speedup     evals(classical Adagrad) / evals(Meta-B);  gate = speedup >= 10 AND fc within +5 %
  fc          forecast RMSE at 1 TU, forecast always run with the TRUE model M1 from the analysis
              (rmse_fc for M1 runs, rmse_fc_m1 for emulator runs, as in report_ic)
  basin+ / -  cases where Meta-B ends with a final cost > 1.1 x (resp. < 1/1.1 x) classical Adagrad (same dynamics)
  closure     (E4-B only) gap-closure ratio = median_i(fc_E3,i - fc_E4,i) / median_i(fc_E3,i - fc_E1,i)
              (per-case E4 = seed mean; 0 = no gain over emulator 4D-Var, 1 = recovers true-model 4D-Var);
              per-seed closures give the sd
  P1          (E4-B) median true-model cost J_M1(analysis) ratio E4 / E1
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
LEAD = 100
CONFIGS = [("l63", c, p) for c in ("strong", "weak") for p in ("full", "x_only")] + \
          [(e, c, p) for e in ("rho275", "xz095", "xz090", "clean") for c in ("strong", "weak") for p in ("full", "x_only")]


def one(folder: str, cfgname: str, method: str, split: str):
    hits = sorted((RES / folder).glob(f"{cfgname}_{method}_lr*_{split}.nc"))
    return xr.open_dataset(hits[0]) if hits else None


def evals_to(j, j_star, tol=0.01):
    hit = j <= (1 + tol) * j_star[:, None]
    return np.where(hit.any(1), hit.argmax(1), np.nan).astype(float)


def fc(ds, emu: bool):
    return (ds.rmse_fc_m1 if emu else ds.rmse_fc).sel(lead=LEAD).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["test", "test_ood"])
    ap.add_argument("--variant", default="delta", choices=["delta", "supervised"])
    a = ap.parse_args()
    sup = a.variant == "supervised"
    tag = "sup" if sup else "C1"
    for split in a.splits:
        rows = []
        for dyn, con, pat in CONFIGS:
            emu = dyn != "l63"
            name = f"{con}_{pat}"
            cls_folder = f"e3_{dyn}" if emu else "e1"
            meta_folder = f"e4b_{dyn}" if emu else "e2b"
            metas = []
            for p in sorted((RES / meta_folder).glob(f"{a.variant}_{name}_adagrad_lr*_{tag}_s[0-9]_{split}.nc")):
                try:                                       # unreadable (interrupted / failed write) -> skipped, warned
                    metas.append(xr.open_dataset(p).load())
                except OSError:
                    print(f"WARNING: unreadable record skipped: {p.name}")
            ada, gd = one(cls_folder, name, "adagrad", split), one(cls_folder, name, "gd", split)
            if not metas or ada is None:
                continue
            j_star = np.nanmin(np.stack([d.J_iter.values[:, -1] for d in (ada, gd) if d is not None]), axis=0)
            e_ada = evals_to(ada.J_iter.values, j_star)
            fc_ada = fc(ada, emu)
            per = []
            for m in metas:
                e_m = evals_to(m.J_iter.values, j_star)
                jm, ja = m.J_iter.values[:, -1], ada.J_iter.values[:, -1]
                per.append({"evals": np.nanmedian(e_m), "reached": np.isfinite(e_m).mean(),
                            "fc": np.nanmedian(fc(m, emu)), "ic": np.nanmedian(m.rmse_x0.values),
                            "basin_worse": int((jm > 1.1 * ja).sum()), "basin_better": int((ja > 1.1 * jm).sum()),
                            "diverged": int(m.diverged.sum()), "fc_case": fc(m, emu),
                            "cos_ghat": float(np.nanmedian(m.cos_ghat_grad.values)) if "cos_ghat_grad" in m
                            else np.nan})
            mean = lambda k: float(np.mean([p[k] for p in per]))  # noqa: E731
            sd = lambda k: float(np.std([p[k] for p in per], ddof=1)) if len(per) > 1 else np.nan  # noqa: E731
            r = {"dynamics": "M1 (E2-B)" if not emu else f"{dyn} (E4-B)", "constraint": con, "obs": pat,
                 "seeds": len(per), "lr": json.loads(metas[0].attrs["config"])["lr"],
                 "evals cls": float(np.nanmedian(e_ada)), "evals meta": f"{mean('evals'):.1f} ± {sd('evals'):.1f}",
                 "speedup": float(np.nanmedian(e_ada)) / mean("evals"),
                 "reached cls/meta": f"{np.isfinite(e_ada).mean():.2f} / {mean('reached'):.2f}",
                 "fc cls": float(np.nanmedian(fc_ada)), "fc meta": f"{mean('fc'):.3f} ± {sd('fc'):.3f}",
                 "fc rel": mean("fc") / float(np.nanmedian(fc_ada)) - 1,
                 "IC err cls/meta": f"{np.nanmedian(ada.rmse_x0.values):.3f} / {mean('ic'):.3f}",
                 "basin -/+": f"{mean('basin_worse'):.0f} / {mean('basin_better'):.0f}",
                 "diverged": sum(p["diverged"] for p in per)}
            # supervised: an iteration costs one FORWARD run (no adjoint); classical: one forward + adjoint run
            r["meta cost unit"] = "forward run" if sup else "gradient eval"
            if sup:
                r["cos(g_hat, grad)"] = mean("cos_ghat")
            r["gate"] = bool(r["speedup"] >= 10 and r["fc rel"] <= 0.05)
            if emu:
                e1 = one("e1", name, "adagrad", split)
                fc_e1, fc_e3 = fc(e1, False), fc_ada
                fc_e4 = np.mean([p["fc_case"] for p in per], 0)
                den = np.nanmedian(fc_e3 - fc_e1)
                r["E3-E1 gap"] = float(den)
                r["closure"] = float(np.nanmedian(fc_e3 - fc_e4) / den)
                r["closure sd"] = float(np.std([np.nanmedian(fc_e3 - p["fc_case"]) / den for p in per], ddof=1))
                r["P1 E4/E1"] = float(np.nanmedian(np.mean([m.J_true_final.values for m in metas], 0)
                                                   / e1.J_iter.values[:, -1]))
                r["P1 E3/E1"] = float(np.nanmedian(ada.J_true_final.values / e1.J_iter.values[:, -1]))
            rows.append(r)
        df = pd.DataFrame(rows)
        out = RES / (f"meta_b_sup_summary_{split}" if sup else f"meta_b_summary_{split}")
        df.to_csv(out.with_suffix(".csv"), index=False)
        title = "Meta-B supervised" if sup else "Meta-B (locked C1)"
        md = f"## {title} vs classical Adagrad — {split}\n\n" + to_markdown(df) + "\n"
        out.with_suffix(".md").write_text(md)
        print(md)


if __name__ == "__main__":
    main()
