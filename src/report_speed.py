"""
Meta-speed check report (2026-11-16): gradients / time to the 1 % cost target for the classical reference, the
classical floor (L-BFGS, Gauss-Newton; src/run_floor_v2.py) and the cell-1 / cell-5 learners (original M5 runs and
the speed-retuning sweep tmp/results/m5spd_<model>), strong / full, D1 v2.

Usage (project root):  python src/report_speed.py --model f1 --split val
Writes tmp/results/speed_<model>_<split>.md (+ .csv).
Selection rule (val only, pre-registered with the pilot request 2026-11-16, as M4): among learners with
reached >= reference reached - 0.05 and true-model fc@1TU <= 1.05 x reference, fewest median gradients to target.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "tmp/results"


def iters_to(j, js, tol=0.01):
    hit = j <= (1 + tol) * js[:, None]
    return np.where(hit.any(1), hit.argmax(1), np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="f1")
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    m, sp = a.model, a.split
    ref = xr.load_dataset(next((RES / f"v2_{m}").glob(f"bice_strong_full_adagrad_lr0.3_{sp}.nc")))
    gd = xr.load_dataset(next((RES / f"v2_{m}").glob(f"bice_strong_full_gd_lr*_{sp}.nc")))
    js = np.nanmin(np.stack([ref.J_iter.values[:, -1], gd.J_iter.values[:, -1]]), 0)
    t_grad = json.loads((RES / "grad_cost_ratio.json").read_text())[f"{'f1' if m == 'f1' else 'emulator'}/strong"][
        "gradient_s"]

    def row(name, ds, seeds=1):
        it = iters_to(ds.J_iter.values, js)
        jm, jr = ds.J_iter.values[:, -1], ref.J_iter.values[:, -1]
        return {"method": name, "seeds": seeds, "grads_to_target": float(np.nanmedian(it)),
                "reached": float(np.isfinite(it).mean()),
                "time_per_200_s": float(np.nanmedian(it)) * t_grad,
                "ic_rmse": float(np.nanmedian(ds.rmse_x0)),
                "fc_true_1TU": float(np.nanmedian(ds.rmse_fc_m1.sel(lead=100))),
                "worse_basins": int((jm > 1.1 * jr).sum())}

    rows = [row("reference: Adagrad lr 0.3, exact adjoint", ref), row("GD lr 0.001", gd)]
    groups = {}
    for p in sorted((RES / f"m5_{m}").glob(f"*_c[15]_s?_{sp}.nc")):
        groups.setdefault("M5 " + p.name[: -len(f"_{sp}.nc")].split("_")[-2], []).append(p)
    for p in sorted((RES / f"m5spd_{m}").glob(f"*_{sp}.nc")):
        name = p.name.rsplit("_s", 1)[0].split("lr", 1)[1].split("_", 1)[1]     # e.g. c1_os20_a2
        groups.setdefault(name, []).append(p)
    for name, paths in groups.items():
        rs = [row(name, xr.load_dataset(p)) for p in paths]
        r = {k: (float(np.mean([x[k] for x in rs])) if isinstance(rs[0][k], (int, float)) else rs[0][k])
             for k in rs[0]}
        r["seeds"] = len(rs)
        rows.append(r)
    df = pd.DataFrame(rows)
    R = df.iloc[0]
    df["speed-up"] = R.grads_to_target / df.grads_to_target
    df["eligible"] = (df.reached >= R.reached - 0.05) & (df.fc_true_1TU <= 1.05 * R.fc_true_1TU)
    fl = RES / f"floor_v2_{sp}.json"
    if fl.exists():
        f = json.loads(fl.read_text()).get(m)
        if f:
            for k, lab, key in (("lbfgs", "floor: L-BFGS (strong Wolfe)", "grads_to_target_median"),
                                ("gauss_newton", "floor: Gauss-Newton (+3 TL runs / iter)", "iters_to_target_median")):
                df.loc[len(df)] = {"method": lab, "seeds": 1, "grads_to_target": f[k][key], "reached": f[k]["reached"],
                                   "time_per_200_s": f["time_to_target_s"][k], "ic_rmse": f[k]["rmse_x0_median"],
                                   "fc_true_1TU": np.nan, "worse_basins": f[k]["worse_basin_vs_ref"],
                                   "speed-up": f["time_to_target_s"]["adagrad_ref"] / f["time_to_target_s"][k],
                                   "eligible": np.nan}
    df.to_csv(RES / f"speed_{m}_{sp}.csv", index=False)
    with open(RES / f"speed_{m}_{sp}.md", "w") as fh:
        fh.write(f"## Meta-speed check — {m}, {sp} (strong / full; target J <= 1.01 J*)\n\n")
        fh.write(df.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
