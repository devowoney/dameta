"""
README figure: behaviour of the seven rectangle models (L1, F1, F2, E1, E2, E1b, E2b).

    (a) same initial condition: x(t) of every model from the true state of test case 0
    (b) two initial conditions 1e-3 apart: |x(t) - x'(t)| per model, median and 25-75 % over the test cases,
        with the fitted exponential growth rate (log-linear fit over 0.5-6 TU)

Usage (project root):  python src/fig_readme_models.py --data-root <worktree with tmp/data and tmp/emulators> \
                           --out docs/figures/models_behaviour
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
MODELS = ["l1", "f1", "f2", "e1", "e2", "e1b", "e2b"]
NAMES = {"l1": "L1 (truth)", "f1": "F1", "f2": "F2", "e1": "E1", "e2": "E2", "e1b": "E1b", "e2b": "E2b"}
COL = {"l1": "#52514e", "f1": "#5598e7", "f2": "#184f95", "e1": "#eb6834", "e2": "#a8431a", "e1b": "#d6a21e",
       "e2b": "#7a5a10"}
LS = {"l1": "-", "f1": "-", "f2": "-", "e1": "--", "e2": "--", "e1b": ":", "e2b": ":"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
WINDOW_TU = 1.6


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
    ap.add_argument("--out", default="docs/figures/models_behaviour")
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--n-pairs", type=int, default=200)
    ap.add_argument("--delta", type=float, default=1e-3)
    ap.add_argument("--t-same", type=float, default=5.0)
    ap.add_argument("--t-pair", type=float, default=10.0)
    a = ap.parse_args()
    root = Path(a.data_root)
    reg = load_registry()
    steps = {m: build_model(m, reg, str(root / "tmp/emulators"))[0] for m in MODELS}
    ds = xr.load_dataset(root / "tmp/data/l63v2_test_seed42.nc")
    dt = float(ds.attrs["dt"])
    x0_all = torch.tensor(ds.x_true.values[:, 0])
    n_same, n_pair = int(round(a.t_same / dt)), int(round(a.t_pair / dt))

    same = {m: run(steps[m], x0_all[a.case][None], n_same)[0].numpy() for m in MODELS}
    x0 = x0_all[: a.n_pairs]
    d = torch.ones(3) / np.sqrt(3) * a.delta
    sep, rate = {}, {}
    t_pair = np.arange(n_pair + 1) * dt
    fit = (t_pair >= 0.5) & (t_pair <= 6.0)
    for m in MODELS:
        s = (run(steps[m], x0, n_pair) - run(steps[m], x0 + d, n_pair)).norm(dim=-1).numpy()
        q = np.percentile(s, [25, 50, 75], 0)
        sep[m] = q
        rate[m] = np.polyfit(t_pair[fit], np.log(q[1][fit]), 1)[0]
        print(f"{m}: growth rate {rate[m]:.3f} /TU", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.linewidth": 0.6, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "lines.linewidth": 1.2, "mathtext.default": "regular"})
    fig, axs = plt.subplots(1, 2, figsize=(7.48, 2.6), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.25})
    t_same = np.arange(n_same + 1) * dt
    ax = axs[0]
    ax.axvspan(0, WINDOW_TU, color=GRID, alpha=0.7, lw=0, zorder=0)
    ax.text(WINDOW_TU + 0.05, 0.03, "DA window", transform=ax.get_xaxis_transform(), fontsize=6.5, color=MUTED)
    for m in MODELS:
        ax.plot(t_same, same[m][:, 0], color=COL[m], ls=LS[m], lw=1.6 if m == "l1" else 1.1, label=NAMES[m],
                zorder=3 if m == "l1" else 2)
    ax.set(xlabel="time [TU]", ylabel="x", xlim=(0, a.t_same),
           title=f"(a) same initial condition (test case {a.case})")
    ax.legend(ncol=7, loc="upper left", bbox_to_anchor=(0, -0.2), fontsize=7, handlelength=1.8, columnspacing=1.0)
    ax = axs[1]
    for m in MODELS:
        q = sep[m]
        if m == "l1":
            ax.fill_between(t_pair, q[0], q[2], color=GRID, lw=0, zorder=0)
        ax.semilogy(t_pair, q[1], color=COL[m], ls=LS[m], label=f"{NAMES[m].split()[0]}  λ ≈ {rate[m]:.2f}/TU")
    ax.set(xlabel="time [TU]", ylabel="|x(t) − x'(t)|", xlim=(0, a.t_pair),
           title=f"(b) two initial conditions {a.delta:g} apart")
    ax.legend(loc="lower right", fontsize=6.5)
    ax.text(0.02, 0.97, f"median of {a.n_pairs} test cases; grey: L1 25–75 %", transform=ax.transAxes,
            fontsize=6.5, color=MUTED, va="top")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print("saved", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
