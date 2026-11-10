"""
E1 report: results table (markdown + CSV) and figures from the final classical 4D-Var runs (tmp/results/e1/).

Usage:  python src/report_e1.py           -> tmp/figs/m2_*.{png,pdf}, tmp/results/e1/summary.{csv,md}

Table, per (split, constraint, pattern, method): diverged / converged fractions, median gradient evaluations,
median final J (and chi2 reference m/2), median IC RMSE (analysis vs background), forecast RMSE at leads
1, 2, 5 time units beyond the window (median over cases), valid prediction time (VPT: first lead where the forecast
RMSE exceeds 0.4 x the climatological std of the truth; median, in time units), wall-clock per case.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES, FIGS = ROOT / "tmp/results/e1", ROOT / "tmp/figs"
DT = 0.01
COLORS = {"gd": "#c0392b", "adagrad": "#1f4e79", "background": "#7f8c8d"}


def vpt(rmse_fc: np.ndarray, thr: float) -> np.ndarray:
    """Valid prediction time per case: first lead (time units) with rmse > thr; full range if never exceeded."""
    above = rmse_fc > thr
    first = np.where(above.any(1), above.argmax(1), rmse_fc.shape[1] - 1).astype(float)
    first[~np.isfinite(rmse_fc).all(1)] = np.nan
    return first * DT


def load_runs() -> list:
    runs = []
    for p in sorted(RES.glob("*.nc")):
        ds = xr.open_dataset(p)
        runs.append((json.loads(ds.attrs["config"]), ds))
    return runs


def summary_table(runs: list, clim_std: float) -> pd.DataFrame:
    rows = []
    for cfg, ds in runs:
        n_obs_used = int(xr.open_dataset(ROOT / f"tmp/data/l63_{cfg['split']}_seed42.nc").obs_mask
                         .sel(pattern=cfg["pattern"]).sum())
        fc = ds.rmse_fc.values
        rows.append({
            "split": cfg["split"], "constraint": cfg["constraint"], "pattern": cfg["pattern"], "method": cfg["method"],
            "lr": cfg["lr"], "n": ds.sizes["ic"], "diverged": float(ds.diverged.mean()),
            "converged": float(ds.converged.mean()), "evals_med": float(np.median(ds.n_grad_evals)),
            "J_med": float(np.nanmedian(ds.J_iter.values[:, -1])), "J_chi2_ref": n_obs_used / 2,
            "rmse_x0_med": float(np.nanmedian(ds.rmse_x0)), "rmse_x0_b_med": float(np.median(ds.rmse_x0_b)),
            "fc_1TU": float(np.nanmedian(fc[:, 100])), "fc_2TU": float(np.nanmedian(fc[:, 200])),
            "fc_5TU": float(np.nanmedian(fc[:, 500])),
            "fc_b_1TU": float(np.median(ds.rmse_fc_b.values[:, 100])),
            "vpt_med": float(np.nanmedian(vpt(fc, 0.4 * clim_std))),
            "vpt_b_med": float(np.median(vpt(ds.rmse_fc_b.values, 0.4 * clim_std))),
            "wall_per_case_s": float(ds.attrs["wall_s"]) / ds.sizes["ic"],
        })
    return pd.DataFrame(rows).sort_values(["split", "constraint", "pattern", "method"])


def fig_forecast(runs: list) -> None:
    """Median forecast RMSE vs lead, test vs test_ood, strong/full and weak/full."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True, sharey=True)
    for i, constraint in enumerate(["strong", "weak"]):
        for j, split in enumerate(["test", "test_ood"]):
            ax = axes[i, j]
            bg_done = False
            for cfg, ds in runs:
                if (cfg["constraint"], cfg["pattern"], cfg["split"]) != (constraint, "full", split):
                    continue
                lead = ds.lead.values * DT
                ax.plot(lead, np.nanmedian(ds.rmse_fc.values, 0), color=COLORS[cfg["method"]], lw=1.6,
                        label=f"{cfg['method']} (lr={cfg['lr']:g})")
                if not bg_done:
                    ax.plot(lead, np.median(ds.rmse_fc_b.values, 0), color=COLORS["background"], lw=1.2, ls="--",
                            label="background (no DA)")
                    bg_done = True
            ax.set_title(f"{constraint} constraint, full obs — {split}", fontsize=9)
            ax.set_yscale("log")
            ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("lead time beyond window (time units)")
    for ax in axes[:, 0]:
        ax.set_ylabel("median forecast RMSE")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("E1 classical 4D-Var (M1): forecast skill beyond the assimilation window", fontsize=10)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(FIGS / f"m2_e1_forecast_rmse.{e}", dpi=150)


def fig_convergence(runs: list) -> None:
    """Cost decrease vs gradient evaluations (median over cases of J_k / J_0), val split, full obs."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, constraint in zip(axes, ["strong", "weak"]):
        for cfg, ds in runs:
            if (cfg["constraint"], cfg["pattern"], cfg["split"]) != (constraint, "full", "test"):
                continue
            j = ds.J_iter.values
            rel = j / j[:, :1]
            ax.plot(ds.k.values, np.nanmedian(rel, 0), color=COLORS[cfg["method"]], lw=1.6,
                    label=f"{cfg['method']} (lr={cfg['lr']:g})")
            ax.fill_between(ds.k.values, np.nanpercentile(rel, 25, 0), np.nanpercentile(rel, 75, 0),
                            color=COLORS[cfg["method"]], alpha=0.15)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("gradient evaluations (model + adjoint runs)")
        ax.set_title(f"{constraint} constraint, full obs — test", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$J_k / J_0$  (median, IQR band)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(FIGS / f"m2_e1_convergence.{e}", dpi=150)


def evals_to_target(runs: list) -> pd.DataFrame:
    """
    Cost to reach a target: per case, J* = best final J over the methods run on the same problem; report the median
    number of gradient evaluations each method needs to reach J <= (1 + tol) J*, and the fraction of cases that reach
    it within the budget. More faithful than "evaluations until the stopping rule fires" (curves plateau early).
    """
    groups = {}
    for cfg, ds in runs:
        groups.setdefault((cfg["split"], cfg["constraint"], cfg["pattern"]), {})[cfg["method"]] = ds.J_iter.values
    rows = []
    for (split, constraint, pattern), js in sorted(groups.items()):
        j_star = np.nanmin(np.stack([j[:, -1] for j in js.values()]), axis=0)
        for method, j in sorted(js.items()):
            for tol in (0.01, 0.05):
                hit = j <= (1 + tol) * j_star[:, None]
                k = np.where(hit.any(1), hit.argmax(1), np.nan)
                rows.append({"split": split, "constraint": constraint, "pattern": pattern, "method": method,
                             "tol": tol, "evals_med": float(np.nanmedian(k)), "reached": float(np.isfinite(k).mean())})
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    """Minimal markdown table (pandas.to_markdown needs `tabulate`, not installed in env oceanai)."""
    fmt = lambda v: f"{v:.3g}" if isinstance(v, float) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    clim_std = float(xr.open_dataset(ROOT / "tmp/data/l63_train_seed42.nc").x_true.std())  # climatological std
    df = summary_table(runs, clim_std)
    df.to_csv(RES / "summary.csv", index=False)
    md = to_markdown(df)
    (RES / "summary.md").write_text(md + f"\n\nVPT threshold = 0.4 x climatological std = {0.4 * clim_std:.2f}\n")
    print(md)
    print(f"\nVPT threshold = 0.4 x clim std = {0.4 * clim_std:.2f}")
    et = evals_to_target(runs)
    et.to_csv(RES / "evals_to_target.csv", index=False)
    print("\n" + to_markdown(et))
    fig_forecast(runs)
    fig_convergence(runs)
    print("figures -> tmp/figs/m2_e1_forecast_rmse.*, m2_e1_convergence.*")


if __name__ == "__main__":
    main()
