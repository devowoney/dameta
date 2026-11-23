"""
Ensemble error growth of the model rectangle without using the truth (pilot 2026-11-20).

Members are built as in a real system (no truth inside the forecast):
    member_i(t0)  = M^tau( x_true(t0 - tau) + eps_i ),   eps_i ~ N(0, s2 I)       short model forecast of a noised state
    control(t0)   = M^tau( x_true(t0 - tau) )                                     the model's own control
Each model M in {L1, F1, F2, E1, E2} then propagates its members and its control for `horizon` steps. The error is
always measured against the model's OWN control:  d_i(t) = x_i^M(t) - x_c^M(t)  (the truth is never used).
The same noise draws eps_i are used for every model (common random numbers), so curves differ only by the model.

Two studies (D1 v2 test split; x_true_past gives the state before t0):
    windows  100 different windows (test cases 0..99), one member per window   -> flow-averaged error growth
    fixed    one fixed window (test case `case`), 100 members                  -> flow-dependent error growth

Per member:  e_i(t)   = RMS over (x, y, z) of d_i(t)
             rho_i(t) = 1 - |d_i(t)|^2 / (2 sum_v sigma_v^2)   (sigma_v^2 = L1 climatological variance; rho = correlation
                        between member and control for stationary statistics: 1 = identical, 0 = independent)
             T_dec,i  = first lead where rho_i < rho_thr (decorrelation time)

Usage (project root):  python src/ensemble_models.py [--tau 50] [--s2 2.0] [--horizon 500] [--case 0] [--rho-thr 0.5]
Outputs: tmp/results/models/ensemble_{windows,fixed}.npz,
         tmp/figs/ens_{windows,fixed}_{growth,decorr,summary,dist}.{png,pdf}
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
ROOT = Path(__file__).resolve().parent.parent
MODELS = ["l1", "f1", "f2", "e1", "e2"]
# validated palette (dataviz validator, light surface): F = blue family, E = orange family; level 2 = darker step.
# L1 = neutral reference ink. Secondary encoding: F solid, E dashed, plus direct labels.
COL = {"l1": "#52514e", "f1": "#5598e7", "f2": "#184f95", "e1": "#eb6834", "e2": "#a8431a"}
LS = {"l1": "-", "f1": "-", "f2": "-", "e1": "--", "e2": "--"}
WINDOW_TU = 1.1


@torch.no_grad()
def propagate(step, x, n):
    out = [x]
    for _ in range(n):
        x = step(x)
        out.append(x)
    return torch.stack(out, 1)                                   # [N, n + 1, 3]


@torch.no_grad()
def run_study(steps, base, noise, tau, horizon, var_clim):
    """base [N, 3] states at t0 - tau, noise [N, 3]. Returns {model: dict(e, rho, tdec)}."""
    out = {}
    for m, step in steps.items():
        xm, xc = base + noise, base
        for _ in range(tau):                                     # short model forecast to t0 (member and control)
            xm, xc = step(xm), step(xc)
        d = propagate(step, xm, horizon) - propagate(step, xc, horizon)       # [N, T + 1, 3]
        e = (d ** 2).mean(-1).sqrt().numpy()
        rho = 1 - (d ** 2).sum(-1).numpy() / (2 * float(var_clim.sum()))
        out[m] = {"e": e, "rho": rho, "d0": d[:, 0].numpy()}
    return out


def tdec(rho, thr, dt):
    below = rho < thr
    return np.where(below.any(1), below.argmax(1), np.nan) * dt


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True,
                         "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False, "figure.facecolor": "white",
                         "savefig.bbox": "tight", "lines.linewidth": 2})
    return plt


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"tmp/figs/{name}.{ext}", dpi=200)


def window_mark(ax):
    ax.axvline(WINDOW_TU, color="#52514e", lw=0.8, ls=":")
    ax.text(WINDOW_TU, 1.0, " window 1.1 TU", transform=ax.get_xaxis_transform(), fontsize=7, color="#52514e",
            va="top", ha="left")


def figures(res, t, tag, title, thr, dt):
    plt = style()
    names = {m: m.upper() for m in MODELS}
    # ---- 1 / 2. per model: every run as a faint line, mean (line) ± 1 std (band); error growth and decorrelation ---
    for key, fname, ylab, what in (("e", "growth", "|member − own control| (RMS over x, y, z)", "error growth"),
                                   ("rho", "decorr", "ρ(t) = 1 − |d|² / (2 Σ σ²_clim)", "decorrelation member ↔ own control")):
        fig, axs = plt.subplots(1, 5, figsize=(15, 3.1), sharey=True)
        for ax, m in zip(axs, MODELS):
            a = res[m][key]
            ax.plot(t, a.T, color=COL[m], lw=0.4, alpha=0.08)
            mu, sd = a.mean(0), a.std(0)
            lo = np.maximum(mu - sd, 0) if key == "e" else mu - sd
            ax.fill_between(t, lo, mu + sd, color=COL[m], alpha=0.22, lw=0)
            ax.plot(t, mu, color=COL[m], ls=LS[m], lw=2)
            ax.set_title(names[m], fontsize=10, color="#0b0b0b")
            ax.set_xlabel("lead time [TU]")
            if key == "e":
                ax.set_ylim(0, max(np.nanpercentile(r["e"], 99.5) for r in res.values()))
            else:
                ax.set_ylim(-1.0, 1.05)
                ax.axhline(thr, color="#52514e", lw=0.8, ls="--")
            window_mark(ax)
        axs[0].set_ylabel(ylab)
        fig.suptitle(f"{title}\n{what} (faint: each run; line: mean; band: ± 1 std)", fontsize=10, y=1.06)
        save(fig, f"ens_{tag}_{fname}")
        plt.close(fig)

    # ---- summary: model means only (spread is in the per-model figures); decorrelation-time distribution --------
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.6), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    for m in MODELS:
        axs[0].plot(t, res[m]["e"].mean(0), color=COL[m], ls=LS[m], label=names[m])
        axs[1].plot(t, res[m]["rho"].mean(0), color=COL[m], ls=LS[m], label=names[m])
    axs[0].set(xlabel="lead time [TU]", ylabel="mean |member − own control|", title="error growth (mean)", ylim=(0, None))
    axs[1].axhline(thr, color="#52514e", lw=0.8, ls="--")
    axs[1].text(t[-1], thr, f"ρ = {thr:g}", fontsize=7, ha="right", va="bottom", color="#52514e")
    axs[1].set(xlabel="lead time [TU]", ylabel="mean ρ(t)", title="decorrelation member ↔ own control (mean)")
    axs[0].legend(fontsize=8, loc="upper left")
    for ax in axs[:2]:
        window_mark(ax)
    td = [res[m]["tdec"] for m in MODELS]
    vp = axs[2].violinplot([x[np.isfinite(x)] for x in td], showmedians=False, showextrema=False, widths=0.8)
    for body, m in zip(vp["bodies"], MODELS):
        body.set_facecolor(COL[m])
        body.set_edgecolor("none")
        body.set_alpha(0.35)
    rng = np.random.default_rng(0)
    for k, (m, x) in enumerate(zip(MODELS, td)):
        x = x[np.isfinite(x)]
        axs[2].scatter(k + 1 + rng.uniform(-0.12, 0.12, len(x)), x, s=4, color=COL[m], alpha=0.5, lw=0)
        axs[2].errorbar(k + 1.32, x.mean(), yerr=x.std(), fmt="o", color="#0b0b0b", ms=4, lw=1, capsize=2)
    axs[2].axhline(WINDOW_TU, color="#52514e", lw=0.8, ls=":")
    axs[2].set_xticks(range(1, 6), [names[m] for m in MODELS])
    axs[2].set(ylabel=f"decorrelation time T(ρ < {thr:g}) [TU]",
               title="decorrelation time per run (dot: mean ± std)")
    fig.suptitle(title, fontsize=10, y=1.02)
    save(fig, f"ens_{tag}_summary")
    plt.close(fig)

    # ---- 3. error distribution at selected leads ---------------------------------------------------------------
    leads = [0.0, 0.5, 1.1, 2.0, 3.0, 5.0]
    leads = [L for L in leads if L <= t[-1] + 1e-9]
    fig, axs = plt.subplots(1, len(leads), figsize=(2.6 * len(leads), 3.2), sharey=False)
    for ax, L in zip(axs, leads):
        i = int(round(L / dt))
        data = [res[m]["e"][:, i] for m in MODELS]
        vp = ax.violinplot(data, showmedians=False, showextrema=False, widths=0.85)
        for body, m in zip(vp["bodies"], MODELS):
            body.set_facecolor(COL[m])
            body.set_edgecolor("none")
            body.set_alpha(0.45)
        for k, x in enumerate(data):
            ax.errorbar(k + 1, x.mean(), yerr=x.std(), fmt="o", color="#0b0b0b", ms=3, lw=1, capsize=2)
        ax.set_xticks(range(1, 6), [names[m] for m in MODELS], fontsize=7)
        ax.set_title(f"lead {L:g} TU", fontsize=9)
    axs[0].set_ylabel("|member − own control|")
    fig.suptitle(f"{title}: error distribution (violin: density; dot: mean ± std)", fontsize=10, y=1.03)
    save(fig, f"ens_{tag}_dist")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=int, default=50, help="short forecast length before t0 [steps] (pilot: 0.5 TU)")
    ap.add_argument("--s2", type=float, default=2.0, help="variance of the Gaussian noise on the state at t0 - tau")
    ap.add_argument("--horizon", type=int, default=500, help="forecast length after t0 [steps]")
    ap.add_argument("--case", type=int, default=0, help="test case defining the fixed window")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--rho-thr", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    reg = load_registry()
    steps = {m: build_model(m, reg, str(ROOT / "tmp/emulators"))[0] for m in MODELS}
    ds = xr.load_dataset(ROOT / "tmp/data/l63v2_test_seed42.nc")
    past = torch.tensor(ds.x_true_past.values)                  # [n_case, history + 1, 3], lag 0 = t0
    dt = float(ds.attrs["dt"])
    var_clim = torch.tensor(xr.load_dataset(ROOT / "tmp/data/l63v2_train_seed42.nc").x_true.values
                            ).reshape(-1, 3).var(0)
    g = torch.Generator().manual_seed(a.seed)
    noise = torch.randn(a.n, 3, generator=g) * np.sqrt(a.s2)     # shared by all models and both studies
    studies = {"windows": (past[:a.n, a.tau], f"{a.n} windows (test cases 0–{a.n - 1}), 1 run each"),
               "fixed": (past[a.case, a.tau].expand(a.n, 3), f"fixed window (test case {a.case}), {a.n} runs")}
    t = np.arange(a.horizon + 1) * dt
    (ROOT / "tmp/figs").mkdir(parents=True, exist_ok=True)
    (ROOT / "tmp/results/models").mkdir(parents=True, exist_ok=True)
    for tag, (base, title) in studies.items():
        res = run_study(steps, base, noise, a.tau, a.horizon, var_clim)
        for m in MODELS:
            res[m]["tdec"] = tdec(res[m]["rho"], a.rho_thr, dt)
        np.savez_compressed(ROOT / f"tmp/results/models/ensemble_{tag}.npz", t=t,
                            **{f"{m}_{k}": v for m in MODELS for k, v in res[m].items()},
                            config=np.array(str(vars(a))))
        title = (f"{title}; IC = model forecast over {a.tau * dt:g} TU from a state + N(0, {a.s2:g} I)")
        figures(res, t, tag, title, a.rho_thr, dt)
        print(f"{tag}: done", flush=True)


if __name__ == "__main__":
    main()
