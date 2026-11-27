"""
README figure: the 4D-Var scheme on the model rectangle, one case (truth L1, full observations, L-BFGS).

One row per estimation model (L1 reference, F1, F2, E1, E2), x component:
    truth, observations in the window, free forecast of the model from the shared first guess x_b (no assimilation),
    and the 4D-Var analysis trajectory (model run from x0_a) continued as a forecast.
Trajectories are re-run with each model from the stored x0_b / x0_a (strong constraint: the window trajectory is the
model run from x0_a, as in run_4dvar_v2).

Usage (project root):  python src/fig_readme_4dvar.py --data-root <m8 worktree> [--case 4] [--scheme strong]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dameta.dynamics.models import build_model, load_registry  # noqa: E402

torch.set_default_dtype(torch.float64)
MODELS = ["l1", "f1", "f2", "e1", "e2"]
LABEL = {"l1": "L1 (perfect-model reference)", "f1": "F1 (mild misspecification)",
         "f2": "F2 (strong misspecification)", "e1": "E1 (emulator of F1)", "e2": "E2 (emulator of F2)"}
COL = {"l1": "#9a9994", "f1": "#5598e7", "f2": "#184f95", "e1": "#eb6834", "e2": "#a8431a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


@torch.no_grad()
def run(step, x, n):
    out = [x]
    for _ in range(n):
        x = step(x)
        out.append(x)
    return torch.stack(out, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--case", type=int, default=4)
    ap.add_argument("--scheme", default="strong")
    ap.add_argument("--var", type=int, default=0)
    ap.add_argument("--out", default="docs/figures/m8r_scheme_case4")
    a = ap.parse_args()
    root = Path(a.data_root)
    reg = load_registry()
    d = xr.load_dataset(root / "tmp/data/l63v2_val_seed42.nc")
    dt, L = float(d.attrs["dt"]), int(d.attrs["window"])
    xt = d.x_true.values[a.case]                                   # [T + 1, 3]
    n = xt.shape[0] - 1
    t = np.arange(n + 1) * dt
    t_obs = d.t_obs.values * dt
    y = d.y_obs.values[a.case]
    xb = torch.tensor(xr.load_dataset(root / "tmp/data/l63v2_val_seed42_bg_l1.nc").x0_b.values[a.case])[None]
    i1 = L + int(round(1.0 / dt))                                  # 1 TU after the window

    rows = {}
    for m in MODELS:
        step = build_model(m, reg, str(root / "tmp/emulators"))[0]
        r = xr.load_dataset(root / f"tmp/results/m8r/smoke/L1/{m}/bice_{a.scheme}_full_lbfgs_wolfe_lr1_val.nc")
        xa = torch.tensor(r.x0_a.values[a.case])[None]
        free = run(step, xb, n)[0].numpy()
        ana = run(step, xa, n)[0].numpy()
        rows[m] = dict(free=free, ana=ana, ic=np.linalg.norm(ana[0] - xt[0]), icb=np.linalg.norm(free[0] - xt[0]),
                       fa=np.linalg.norm(ana[i1] - xt[i1]), fb=np.linalg.norm(free[i1] - xt[i1]))
        stored = float(r.rmse_fc.values[a.case, int(round(1.0 / dt))]) * np.sqrt(3)   # RMSE over xyz -> norm
        print(m, {k: round(float(v), 3) for k, v in rows[m].items() if np.ndim(v) == 0},
              f"check: stored fc norm @+1TU {stored:.3f}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.linewidth": 0.6, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "mathtext.default": "regular"})
    fig, axs = plt.subplots(len(MODELS), 1, figsize=(7.48, 7.2), sharex=True, sharey=True,
                            gridspec_kw={"hspace": 0.42, "top": 0.93})
    v = a.var
    for ax, m in zip(axs, MODELS):
        rw = rows[m]
        ax.axvspan(0, L * dt, color=GRID, alpha=0.7, lw=0, zorder=0)
        ax.axvline(L * dt + 1.0, color=MUTED, lw=0.6, ls=":", zorder=1)
        ax.plot(t, xt[:, v], color=INK, lw=1.0, label="truth L1", zorder=4)
        ax.plot(t, rw["free"][:, v], color=COL[m], lw=1.0, ls=(0, (2, 1.5)), alpha=0.8,
                label="free forecast from first guess", zorder=2)
        ax.plot(t, rw["ana"][:, v], color=COL[m], lw=1.6, label="4D-Var analysis → forecast", zorder=3)
        ax.plot(t_obs, y[:, v], "o", ms=3.5, mfc="white", mec=INK, mew=0.8, label="observations", zorder=5)
        ax.set_title(LABEL[m], loc="left", fontsize=8, color=COL[m], fontweight="bold", pad=2)
        ax.text(1.0, 1.02, f"error at t$_0$: {rw['icb']:.2f} → {rw['ic']:.2f}    "
                f"forecast error at +1 TU: free {rw['fb']:.1f}, 4D-Var {rw['fa']:.1f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color=MUTED)
        ax.set_ylabel("xyz"[v])
    axs[0].text(0.02, 0.06, "assimilation window", transform=axs[0].transAxes, fontsize=6.5, color=MUTED)
    axs[0].text(L * dt + 1.03, 0.06, "+1 TU", transform=axs[0].get_xaxis_transform(), fontsize=6.5, color=MUTED)
    axs[-1].set(xlabel="time [TU]", xlim=(0, t[-1]))
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.01), fontsize=7)
    fig.suptitle(f"4D-Var ({a.scheme}, L-BFGS, full obs), truth L1, validation case {a.case}: "
                 "observations, analysis and forecast vs free forecast", fontsize=8, y=0.995)
    out = Path(a.out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print("saved", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
