"""
E2-B (Meta-B delta, M1) val-sweep report: cost-to-target and skill per hyper-parameter configuration vs E1.

Usage (project root):  python src/report_e2b.py [--group e2b_sweep] [--constraint strong] [--pattern full]
Reads tmp/results/<group>/*_<split>.nc (--split, default val) and the E1 records of that split and the same
constraint/pattern; writes tmp/results/<group>/sweep_summary_<split>.{csv,md} (+ sweep_seed_summary_<split>.csv).

Cost-to-target (as report_e1.evals_to_target): per case J* = best FINAL J of the E1 classical runs (GD, Adagrad) on
the same problem — a FIXED reference so meta configs do not move their own target; evals to reach
J <= (1 + tol) J* (index of the first hit in J_iter = gradient evaluations), median over cases that reach it and
fraction reached.
Pre-registered selection rule (decided before the sweep finished, 2026-11-11):
    eligible  = reached_1% >= E1 Adagrad reached_1%  AND  median fc RMSE @1TU <= 1.05 x E1 Adagrad
    selected  = eligible config with the smallest median evals to the 1% target (ties: smaller fc @1TU)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from report_e1 import to_markdown  # noqa: E402  (tabulate not installed in env oceanai)

ROOT = Path(__file__).resolve().parent.parent


def cost_to_target(j_iter: np.ndarray, j_star: np.ndarray, tol: float):
    hit = j_iter <= (1 + tol) * j_star[:, None]
    k = np.where(hit.any(1), hit.argmax(1), np.nan)
    return float(np.nanmedian(k)) if np.isfinite(k).any() else np.nan, float(np.isfinite(k).mean())


def row(name: str, ds: xr.Dataset, j_star: np.ndarray, extra: dict) -> dict:
    r = {"config": name, **extra}
    for tol in (0.01, 0.05):
        r[f"evals_{int(tol * 100)}pct"], r[f"reached_{int(tol * 100)}pct"] = cost_to_target(ds.J_iter.values, j_star, tol)
    r.update({
        "diverged": int(ds.diverged.sum()), "converged": float(ds.converged.mean()),
        "evals_stop_med": float(np.median(ds.n_grad_evals)), "J_final_med": float(np.nanmedian(ds.J_iter[:, -1])),
        "rmse_x0_med": float(np.nanmedian(ds.rmse_x0)), "fc_1TU": float(np.nanmedian(ds.rmse_fc.sel(lead=100))),
        "fc_2TU": float(np.nanmedian(ds.rmse_fc.sel(lead=200))),
    })
    if "step_ratio" in ds:
        r["step_ratio_k<20"] = float(np.nanmedian(ds.step_ratio.values[:, :20]))
    if "train_info" in ds.attrs:
        r["train_wall_s"] = json.loads(ds.attrs["train_info"]).get("train_wall_s", np.nan)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="e2b_sweep")
    ap.add_argument("--constraint", default="strong")
    ap.add_argument("--pattern", default="full")
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    e1 = {m: xr.open_dataset(next((ROOT / "tmp/results/e1").glob(f"{a.constraint}_{a.pattern}_{m}_lr*_{a.split}.nc")))
          for m in ("adagrad", "gd")}
    j_star = np.nanmin(np.stack([d.J_iter.values[:, -1] for d in e1.values()]), axis=0)
    rows = [row(f"E1 {m}", d, j_star, {}) for m, d in e1.items()]
    for p in sorted((ROOT / "tmp/results" / a.group).glob(f"delta_{a.constraint}_{a.pattern}_*_{a.split}.nc")):
        ds = xr.open_dataset(p)
        cfg = json.loads(ds.attrs["config"])
        dl = cfg["delta"]
        rows.append(row(p.stem[: -len(a.split) - 1], ds, j_star,
                        {"episode_iters": dl["episode_iters"], "w_perf_delta": dl["w_perf_delta"],
                         "output_scale": cfg["net"]["output_scale"], "align": dl["grad_align_steps"],
                         "trans": dl["grad_trans_steps"], "meta_lr": dl["meta_lr"], "K": dl["K"],
                         "epochs": dl["n_epochs"], "seed": dl["train_seed"],
                         "w_align_perf": dl.get("w_align_perf", 0.0), "select": dl.get("select_epoch", False)}))
    df = pd.DataFrame(rows)
    ref = df[df.config == "E1 adagrad"].iloc[0]
    meta = df[~df.config.str.startswith("E1")].copy()
    meta["eligible"] = (meta["reached_1pct"] >= ref["reached_1pct"]) & (meta["fc_1TU"] <= 1.05 * ref["fc_1TU"])
    df = pd.concat([df[df.config.str.startswith("E1")], meta])
    sel = meta[meta.eligible].sort_values(["evals_1pct", "fc_1TU"]).head(1)
    out = ROOT / "tmp/results" / a.group
    df.to_csv(out / f"sweep_summary_{a.split}.csv", index=False)
    md = to_markdown(df)
    md += f"\n\nE1 Adagrad: evals to 1% target {ref['evals_1pct']:.0f} (reached {ref['reached_1pct']:.2f}); "
    md += f"gate (<= 1/10) = {ref['evals_1pct'] / 10:.1f} evals.\n"
    md += f"Selected (single run): {sel.config.iloc[0] if len(sel) else 'NONE eligible'}\n"
    if meta["seed"].nunique() > 1:
        # Seed aggregation: same configuration except train_seed -> mean +- std over seeds; selection on the mean
        keys = ["episode_iters", "w_perf_delta", "output_scale", "align", "trans", "meta_lr", "K", "epochs",
                "w_align_perf", "select"]
        agg = meta.groupby(keys).agg(n_seeds=("seed", "nunique"), evals_1pct_mean=("evals_1pct", "mean"),
                                     evals_1pct_std=("evals_1pct", "std"), reached_1pct_mean=("reached_1pct", "mean"),
                                     fc_1TU_mean=("fc_1TU", "mean"), fc_1TU_std=("fc_1TU", "std"),
                                     step_ratio=("step_ratio_k<20", "mean"), diverged=("diverged", "sum")).reset_index()
        agg = agg[agg.n_seeds > 1]
        agg["eligible"] = (agg.reached_1pct_mean >= ref["reached_1pct"]) & (agg.fc_1TU_mean <= 1.05 * ref["fc_1TU"])
        agg.to_csv(out / f"sweep_seed_summary_{a.split}.csv", index=False)
        md += "\n## Seed-aggregated (selection on mean)\n\n" + to_markdown(agg)
        best = agg[agg.eligible].sort_values(["evals_1pct_mean", "fc_1TU_mean"]).head(1)
        md += "\n\nSelected (seed mean): " + (best[keys].to_dict("records")[0].__repr__() if len(best) else "NONE") + "\n"
    (out / f"sweep_summary_{a.split}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
