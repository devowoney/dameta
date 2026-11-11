"""
E3 report: 4D-Var through an emulator vs E1 (same cases, same configuration, physical model M1).

Usage:  python src/report_e3.py clean        -> tmp/results/e3_<name>/compare_e1.{csv,md}, tmp/figs/m3_e3_<name>_*.{png,pdf}

Per (split, constraint, pattern, method), medians over the 200 cases:
    J_true_E3   P1: 4D-Var cost of the E3 analysis evaluated with M1        vs  J_E1 (E1 final cost)
    rmse_x0     IC error of the analysis (E3 vs E1)
    fc_m1_1TU   forecast RMSE at 1 TU beyond the window, analysis integrated with M1 (IC quality)
    fc_emu_1TU  same, integrated with the emulator (operational use)
    vpt         valid prediction time (0.4 x clim std) of the M1 forecast from the analysis
    gap_fc      median over cases of (fc_m1_1TU[E3] - fc_1TU[E1]): denominator of the gap-closure ratio (milestone 4)
    evals_1pct  gradient evaluations for E3 to reach J_emu <= 1.01 J*_emu (its own cost)
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DT, LEAD = 0.01, 100


def to_markdown(df: pd.DataFrame) -> str:
    fmt = lambda v: f"{v:.3g}" if isinstance(v, float) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(lines + ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)])


def vpt(fc: np.ndarray, thr: float) -> np.ndarray:
    above = fc > thr
    out = np.where(above.any(1), above.argmax(1), fc.shape[1] - 1).astype(float) * DT
    out[~np.isfinite(fc).all(1)] = np.nan
    return out


def main(name: str) -> None:
    e3_dir, e1_dir = ROOT / f"tmp/results/e3_{name}", ROOT / "tmp/results/e1"
    thr = 0.4 * float(xr.open_dataset(ROOT / "tmp/data/l63_train_seed42.nc").x_true.std())
    rows, curves = [], {}
    for p in sorted(e3_dir.glob("*.nc")):
        e3 = xr.open_dataset(p)
        cfg = json.loads(e3.attrs["config"])
        # E1 counterpart = same (constraint, pattern, method, split) with E1's own selected lr (may differ from E3's)
        (e1_path,) = e1_dir.glob(f"{cfg['constraint']}_{cfg['pattern']}_{cfg['method']}_lr*_{cfg['split']}.nc")
        e1 = xr.open_dataset(e1_path)
        j_emu = e3.J_iter.values
        hit = j_emu <= 1.01 * j_emu[:, -1:]
        rows.append({
            "split": cfg["split"], "constraint": cfg["constraint"], "pattern": cfg["pattern"], "method": cfg["method"],
            "diverged": float(e3.diverged.mean()),
            "J_true_E3": float(np.nanmedian(e3.J_true_final)), "J_E1": float(np.nanmedian(e1.J_iter.values[:, -1])),
            "rmse_x0_E3": float(np.nanmedian(e3.rmse_x0)), "rmse_x0_E1": float(np.nanmedian(e1.rmse_x0)),
            "fc_m1_1TU_E3": float(np.nanmedian(e3.rmse_fc_m1.values[:, LEAD])),
            "fc_1TU_E1": float(np.nanmedian(e1.rmse_fc.values[:, LEAD])),
            "fc_emu_1TU_E3": float(np.nanmedian(e3.rmse_fc.values[:, LEAD])),
            "vpt_E3": float(np.nanmedian(vpt(e3.rmse_fc_m1.values, thr))),
            "vpt_E1": float(np.nanmedian(vpt(e1.rmse_fc.values, thr))),
            "gap_fc_1TU": float(np.nanmedian(e3.rmse_fc_m1.values[:, LEAD] - e1.rmse_fc.values[:, LEAD])),
            "evals_1pct_E3": float(np.median(hit.argmax(1))),
        })
        curves[(cfg["split"], cfg["constraint"], cfg["pattern"], cfg["method"])] = (
            np.nanmedian(e1.rmse_fc.values, 0), np.nanmedian(e3.rmse_fc_m1.values, 0), np.nanmedian(e3.rmse_fc.values, 0))
    df = pd.DataFrame(rows).sort_values(["split", "constraint", "pattern", "method"])
    df.to_csv(e3_dir / "compare_e1.csv", index=False)
    md = to_markdown(df)
    (e3_dir / "compare_e1.md").write_text(md + "\n")
    print(md)

    # Figure: forecast RMSE vs lead, Adagrad, full obs — E1 vs E3 (M1 forecast) vs E3 (emulator forecast)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True, sharey=True)
    for i, constraint in enumerate(["strong", "weak"]):
        for j, split in enumerate(["test", "test_ood"]):
            ax, key = axes[i, j], (split, constraint, "full", "adagrad")
            if key not in curves:
                continue
            e1c, e3m, e3e = curves[key]
            lead = np.arange(len(e1c)) * DT
            ax.plot(lead, e1c, color="#1f4e79", lw=1.6, label="E1: 4D-Var with M1")
            ax.plot(lead, e3m, color="#c0392b", lw=1.6, label=f"E3: 4D-Var with emulator, forecast M1")
            ax.plot(lead, e3e, color="#e67e22", lw=1.2, ls="--", label="E3: forecast with emulator")
            ax.set_yscale("log")
            ax.grid(alpha=0.3)
            ax.set_title(f"{constraint}, full obs, Adagrad — {split}", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("lead time beyond window (time units)")
    for ax in axes[:, 0]:
        ax.set_ylabel("median forecast RMSE")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"E3 (emulator `{name}`) vs E1", fontsize=10)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(ROOT / f"tmp/figs/m3_e3_{name}_forecast.{e}", dpi=150)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "clean")
