"""
Meta-learners vs classical optimizers on the DENSE-obs WEAK-constraint problem (2026-11-16; F1, obs every step,
eta at every step -> control 333, sigma_q = 0.1 / sqrt 10; src/scripts/dense_meta_job.sh, src/run_dense_v2.py).

Same conventions as report_m5.py, two references:
    J* = best final J of the classical optimizers (Adagrad, GD, L-BFGS) on the same case; target J <= 1.01 J*
    cost = model runs to the target in FORWARD-RUN EQUIVALENTS of F1: 1 gradient (forward + adjoint) = r forward runs
           (r measured in this configuration: gradient time / forward time, tmp/results/dense/timing_*.json);
           cell 8a = 1 forward run per iteration; network evaluations are NOT counted (see the wall-clock note)
    speed-up vs Adagrad (exact adjoint, the M5 reference) and vs L-BFGS (the strong classical floor)
    reached, worse basins (final J > 1.1 J*), IC RMSE, forecast RMSE at 1 TU with F1 (operational) and the true model
Meta cells: mean (± sd) over the 3 training seeds of the per-seed medians.
Outputs: tmp/results/dense/meta_summary_<split>.{md,csv}, tmp/figs/dense_meta_curves_<split>.{png,pdf}

Usage (project root, env oceanai):  python src/report_dense_meta.py [--splits test test_ood]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import xarray as xr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_dense as RD  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES, FIGS = ROOT / "tmp/results/dense", ROOT / "tmp/figs"
TAG = "l63v2d_f1_weak_q0.0316228"
GROUP = ROOT / "tmp/results/m5dw_f1"
CLASSICAL = {"adagrad": "Adagrad (exact adjoint) — reference", "gd": "GD", "lbfgs": "L-BFGS (strong floor)"}
CELLS = {"c1": ("delta", "cell 1: learned optimizer, Δ inner loss (1 gradient / it)", "grad"),
         "c5": ("unroll", "cell 5: learned optimization, K 20 unroll (1 gradient / step)", "grad"),
         "c5k50": ("unroll", "cell 5: learned optimization, K 50 unroll", "grad"),
         "c8a": ("learned_grad", "cell 8a: adjoint-free learned gradient + Adagrad (1 forward / it)", "fwd")}
F = 500


def iters_to(j, jstar):
    hit = j <= 1.01 * jstar[:, None]
    return np.where(hit.any(1), hit.argmax(1), np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["test", "test_ood"])
    a = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    tim = json.loads((RES / f"timing_{TAG}_test_with_val_sweep.json").read_text())
    r = tim["gradient_s"] / tim["forward_s"]                       # forward-run equivalents of one gradient
    for split in a.splits:
        ds = xr.load_dataset(ROOT / f"tmp/data/l63v2d_{split}_seed42.nc")
        xt, L = ds.x_true.values, int(ds.attrs["window"])
        cl = {o: xr.load_dataset(RES / f"{TAG}_{o}_{split}.nc") for o in CLASSICAL}
        jfin = np.stack([cl[o].J_hist.values[:, -1] for o in CLASSICAL])
        jstar = np.nanmin(np.where(np.isfinite(jfin), jfin, np.nan), 0)
        rows, curves = [], {}

        def summary(j, ic, fc_est, fc_true, unit):
            k = iters_to(j, jstar) * unit
            return {"cost": float(np.nanmedian(k)), "reached": float(np.isfinite(k).mean()),
                    "worse": int((j[:, -1] > 1.1 * jstar).sum()), "ic": float(np.nanmedian(ic)),
                    "fc_est": float(np.nanmedian(fc_est)), "fc_true": float(np.nanmedian(fc_true))}

        ref = {}
        for o, lab in CLASSICAL.items():
            d = cl[o]
            x0, eta = RD.split_control(RD.control(d), ds, True)
            fc = RD.forecasts(x0, eta, "f1", ds)
            truth_fc = xt[:, L:L + F + 1]
            s = summary(d.J_hist.values, RD.rmse(x0.numpy(), xt[:, 0]), RD.rmse(fc["f1"], truth_fc)[:, 100],
                        RD.rmse(fc["truth"], truth_fc)[:, 100], r)
            ref[o] = s
            rows.append({"method": lab, "seeds": 1, **s})
            curves[o] = (np.arange(d.sizes["k"]) * r, np.nanmedian(d.J_hist.values / jstar[:, None] - 1, 0))
        for c, (var, lab, unit) in CELLS.items():
            per = []
            for s_ in range(3):
                p = GROUP / f"{var}_weak_full_adagrad_lr0.03_{c}_s{s_}_{split}.nc"
                if not p.exists():
                    continue
                d = xr.load_dataset(p)
                u = r if unit == "grad" else 1.0
                j = d.J_iter.values
                per.append(summary(j, d.rmse_x0.values, d.rmse_fc.sel(lead=100).values,
                                   d.rmse_fc_m1.sel(lead=100).values, u))
                if s_ == 0:
                    curves[c] = (np.arange(j.shape[1]) * u, np.nanmedian(j / jstar[:, None] - 1, 0))
            if not per:
                continue
            m = {k: float(np.mean([p_[k] for p_ in per])) for k in per[0]}
            m["cost_sd"] = float(np.std([p_["cost"] for p_ in per]))
            rows.append({"method": lab, "seeds": len(per), **m})
        df = pd.DataFrame(rows)
        df["speed-up vs Adagrad"] = ref["adagrad"]["cost"] / df["cost"]
        df["speed-up vs L-BFGS"] = ref["lbfgs"]["cost"] / df["cost"]
        df.to_csv(RES / f"meta_summary_{split}.csv", index=False)
        cols = ["method", "seeds", "cost", "speed-up vs Adagrad", "speed-up vs L-BFGS", "reached", "worse", "ic",
                "fc_est", "fc_true"]
        head = ["method", "seeds", "fwd-eq to 1 %", "× vs Adagrad", "× vs L-BFGS", "reached", "worse basins /200",
                "IC RMSE", "fc@1TU F1", "fc@1TU true"]
        lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for _, row in df.iterrows():
            cost = f"{row['cost']:.0f}" + (f" ± {row['cost_sd']:.0f}" if "cost_sd" in row and np.isfinite(
                row.get("cost_sd", np.nan)) else "")
            vals = [row["method"], int(row["seeds"]), cost] + [f"{row[k]:.2f}" for k in cols[3:6]] + \
                [int(row["worse"])] + [f"{row[k]:.3f}" for k in cols[7:]]
            lines.append("| " + " | ".join(str(v) for v in vals) + " |")
        md = (f"Dense obs, weak constraint, F1, {split}: 1 gradient = {r:.2f} forward runs; J* = best classical final "
              f"J; meta = mean over seeds (cost ± sd).\n\n" + "\n".join(lines) + "\n")
        (RES / f"meta_summary_{split}.md").write_text(md)
        print(md)
        fig, ax = plt.subplots(figsize=(8, 5))
        sty = {"adagrad": ("tab:blue", "-"), "gd": ("tab:gray", "-"), "lbfgs": ("tab:green", "-"),
               "c1": ("tab:orange", "--"), "c5": ("tab:purple", "--"), "c5k50": ("tab:pink", "--"),
               "c8a": ("tab:red", "--")}
        for k_, (x, y) in curves.items():
            lab = CLASSICAL.get(k_, CELLS.get(k_, ("", k_))[1])
            ax.semilogy(x, np.maximum(y, 1e-8), color=sty[k_][0], ls=sty[k_][1], label=lab)
        ax.axhline(0.01, color="k", lw=0.6, ls=":")
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xlim(0, 4000)
        ax.set_ylim(1e-5, 1e2)
        ax.set_xlabel("F1 forward-run equivalents (symlog)")
        ax.set_ylabel("median J / J* - 1")
        ax.set_title(f"Dense obs, weak constraint (control 333), F1, {split} — meta: seed 0", fontsize=10)
        ax.legend(fontsize=7)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(FIGS / f"dense_meta_curves_{split}.{ext}", dpi=140, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
