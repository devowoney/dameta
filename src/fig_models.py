"""
Paper figures of the model rectangle (pilot 2026-11-24): L1 (truth), F1 / F2 (false L63), E1 / E2 (emulators).

    fig_models_attractor   long free run of every model from the same state (x-z projection, the "butterfly");
                           F / E panels laid out as the rectangle (F1 E1 / F2 E2), L1 attractor underlaid in grey
    fig_models_growth      truth-free ensemble error growth (log scale), (a) many windows, (b) one fixed window
    fig_models_decorr      temporal decorrelation member <-> own control: (a) rho(t), (b) decorrelation time vs the
                           assimilation window

Ensemble members follow src/ensemble_models.py (pilot-approved definition, truth never used in the forecast):
    member(t0) = M^tau( x_true(t0 - tau) + eps ),  eps ~ N(0, s2 I);   control(t0) = M^tau( x_true(t0 - tau) )
    d(t) = member(t) - control(t) propagated by the SAME model; e = RMS over (x, y, z) of d;
    T_dec = first lead with rho < rho_thr.

Two decorrelation measures (pilot 2026-11-24):
  1. distance-based rho(t) = 1 - |d(t)|^2 / (2 sum_v sigma_v^2), one value per run.
     sigma_v^2 = L1 climatological variance of variable v (D1 v2 train truth, pooled over cases and times).
     Origin: for two states a, b of the same climate (mean mu, variance sigma_v^2),
         E|a - b|^2 = sum_v (Var a_v + Var b_v - 2 Cov(a_v, b_v)) = 2 sum_v sigma_v^2 (1 - rho),
     rho = sum_v Cov_v / sum_v sigma_v^2 (variance-weighted correlation). Replacing E|d|^2 by one run's |d|^2 gives
     a per-run estimate: 1 = member equals control, 0 = |d|^2 at the independent-state level (saturation),
     < 0 possible (member and control on opposite wings). rho = 0.5 <=> RMS error = saturation / sqrt(2).
     Common yardstick: L1 sigma_v^2 for every model.
  2. Pearson rho_P(t) across runs (windows study only; in the fixed study the control is one state, so a
     correlation over members is undefined):
         rho_P,v(t) = Cov_n(member_v, control_v) / sqrt(Var_n member_v Var_n control_v)   (n = all runs)
         rho_P(t)   = sum_v Cov_v / sqrt(sum_v Var member_v * sum_v Var control_v)
     Uses each model's own variances, bounded in [-1, 1]. 95 % CI: bootstrap over windows (all members of a
     resampled window kept together), `--n-boot` resamples; T_dec,P = first lead with rho_P < rho_thr.
Same noise draws for all models (common random numbers).
    windows study: every D1 v2 test window (200) x `--members` draws each
    fixed study:   test case `--case`, `--n-fixed` draws

Usage (project root):
    python src/fig_models.py                    # compute (tmp/results/models/fig_models.npz) + figures
    python src/fig_models.py --plot-only        # figures from the saved npz
Outputs: tmp/figs/fig_models_{attractor,growth,decorr}.{pdf,png}
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dameta.dynamics.models import build_model, load_registry, model_params  # noqa: E402

torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parent.parent
NPZ = ROOT / "tmp/results/models/fig_models.npz"
MODELS = ["l1", "f1", "f2", "e1", "e2"]
NAMES = {"l1": "L1", "f1": "F1", "f2": "F2", "e1": "E1", "e2": "E2"}
# same validated palette as src/ensemble_models.py: F = blue family, E = orange family, level 2 darker; L1 neutral ink
COL = {"l1": "#52514e", "f1": "#5598e7", "f2": "#184f95", "e1": "#eb6834", "e2": "#a8431a"}
LS = {"l1": "-", "f1": "-", "f2": "-", "e1": "--", "e2": "--"}
# bad emulators (pilot 2026-11-27): yellow-brown family, same level-2-darker rule; not in MODELS (rectangle figures)
NAMES.update({"e1b": "E1b", "e2b": "E2b"})
COL.update({"e1b": "#d6a21e", "e2b": "#7a5a10"})
LS.update({"e1b": ":", "e2b": ":"})
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
WINDOW_TU = 0.6                                                  # Pearson decorrelation (pilot 2026-11-24); --window-tu
SIGMA_O = np.sqrt(2.0)
W1, W2 = 3.74, 7.48                                              # AGU one / two column widths [in] (95 / 190 mm)


# ---------------------------------------------------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def run(step, x, n, keep_every=1):
    out = [x]
    for i in range(1, n + 1):
        x = step(x)
        if i % keep_every == 0:
            out.append(x)
    return torch.stack(out, 1)


def pearson(xm, xc):
    """xm, xc [N, T, 3] -> (rho_P [T], rho_P,v [T, 3]) across the N runs."""
    am, ac = xm - xm.mean(0), xc - xc.mean(0)
    cov, vm, vc = (am * ac).mean(0), (am ** 2).mean(0), (ac ** 2).mean(0)
    return cov.sum(-1) / np.sqrt(vm.sum(-1) * vc.sum(-1)), cov / np.sqrt(vm * vc)


def first_below(r, thr, dt):
    below = r < thr
    return np.where(below.any(-1), below.argmax(-1), np.nan) * dt


@torch.no_grad()
def ensemble(step, base, noise, tau, horizon, var_clim, boot=None):
    """boot = (members per window, n_boot, rng, thr, dt): also Pearson across runs with a window bootstrap."""
    xm, xc = base + noise, base
    for _ in range(tau):
        xm, xc = step(xm), step(xc)
    tm, tc = run(step, xm, horizon), run(step, xc, horizon)      # [N, T + 1, 3]
    d = tm - tc
    e = (d ** 2).mean(-1).sqrt()
    rho = 1 - (d ** 2).sum(-1) / (2 * var_clim.sum())
    out = {"e": e.numpy().astype(np.float32), "rho": rho.numpy().astype(np.float32)}
    if boot is not None:
        k, n_boot, rng, thr, dt = boot
        tm, tc = tm.numpy().astype(np.float32), tc.numpy().astype(np.float32)
        out["pr"], out["pr_var"] = pearson(tm, tc)
        n_win = tm.shape[0] // k
        curves = np.empty((n_boot, tm.shape[1]), np.float32)
        for b in range(n_boot):
            idx = (rng.integers(0, n_win, n_win)[:, None] * k + np.arange(k)).ravel()
            curves[b] = pearson(tm[idx], tc[idx])[0]
        out["pr_lo"], out["pr_hi"] = np.percentile(curves, [2.5, 97.5], 0)
        out["pr_tdec_boot"] = first_below(curves, thr, dt)
    return out


def compute(a):
    reg = load_registry()
    steps = {m: build_model(m, reg, str(ROOT / "tmp/emulators"))[0] for m in MODELS}
    ds = xr.load_dataset(ROOT / "tmp/data/l63v2_test_seed42.nc")
    dt = float(ds.attrs["dt"])
    past = torch.tensor(ds.x_true_past.values)                  # lag index = steps before t0
    var_clim = torch.tensor(xr.load_dataset(ROOT / "tmp/data/l63v2_train_seed42.nc").x_true.values).reshape(-1, 3).var(0)
    n_win = past.shape[0]
    g = torch.Generator().manual_seed(a.seed)
    noise_w = torch.randn(n_win * a.members, 3, generator=g) * np.sqrt(a.s2)
    noise_f = torch.randn(a.n_fixed, 3, generator=g) * np.sqrt(a.s2)
    base_w = past[:, a.tau].repeat_interleave(a.members, 0)
    base_f = past[a.case, a.tau].expand(a.n_fixed, 3)
    out = {"dt": dt, "var_clim": var_clim.numpy(), "config": np.array(str(vars(a)))}
    x0 = torch.tensor(ds.x_true.values[a.case, 0])[None]
    for m in MODELS:
        rng = np.random.default_rng(a.seed)                     # same bootstrap resamples for every model
        w = ensemble(steps[m], base_w, noise_w, a.tau, a.horizon, var_clim, (a.members, a.n_boot, rng, a.rho_thr, dt))
        f = ensemble(steps[m], base_f, noise_f, a.tau, a.horizon, var_clim)
        out.update({f"{m}_{k}_win" if k in ("e", "rho") else f"{m}_{k}": v for k, v in w.items()})
        out[f"{m}_e_fix"], out[f"{m}_rho_fix"] = f["e"], f["rho"]
        traj = run(steps[m], x0, a.spinup + a.attr_steps)[0, a.spinup:]
        out[f"{m}_attr"] = traj.numpy().astype(np.float32)
        print(f"{m}: done", flush=True)
    for m in ("l1", "f1", "f2"):
        p = model_params(m, reg)
        out[f"{m}_theta"] = np.array([p.sigma, p.rho, p.beta])
    NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(NPZ, **out)


# ---------------------------------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------------------------------
def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
        "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "figure.facecolor": "white",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "lines.linewidth": 1.4, "pdf.fonttype": 42,
        "mathtext.default": "regular"})
    return plt


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / f"tmp/figs/{name}.{ext}", dpi=300)


def panel_label(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right", color=INK)


def window_band(ax):
    ax.axvspan(0, WINDOW_TU, color=GRID, alpha=0.7, lw=0, zorder=0)
    ax.text(WINDOW_TU + 0.08, 0.02, "DA window", transform=ax.get_xaxis_transform(), fontsize=6, color=MUTED,
            ha="left", va="bottom")


def fig_attractor(r, plt):
    fig = plt.figure(figsize=(W2, 3.3))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1, 1], wspace=0.12, hspace=0.22)
    axes = {"l1": fig.add_subplot(gs[:, 0]), "f1": fig.add_subplot(gs[0, 1]), "e1": fig.add_subplot(gs[0, 2]),
            "f2": fig.add_subplot(gs[1, 1]), "e2": fig.add_subplot(gs[1, 2])}
    ref = r["l1_attr"]
    allz = np.concatenate([r[f"{m}_attr"][:, 2] for m in MODELS])
    allx = np.concatenate([r[f"{m}_attr"][:, 0] for m in MODELS])
    xl, zl = (allx.min() - 1, allx.max() + 1), (allz.min() - 1, allz.max() + 1)
    th = {m: r[f"{m}_theta"] for m in ("l1", "f1", "f2")}
    sub = {"l1": "truth, θ = (10, 28, 8/3)",
           "f1": "θ = ({:.2f}, {:.2f}, {:.3f})".format(*th["f1"]),
           "f2": "θ = ({:.2f}, {:.2f}, {:.3f})".format(*th["f2"]),
           "e1": "emulator of F1", "e2": "emulator of F2"}
    for m, ax in axes.items():
        a = r[f"{m}_attr"]
        if m != "l1":
            ax.plot(ref[:, 0], ref[:, 2], color="#c9c8c3", lw=0.25, alpha=0.9, rasterized=True)
        ax.plot(a[:, 0], a[:, 2], color=COL[m], lw=0.25, alpha=0.9, rasterized=True)
        ax.set(xlim=xl, ylim=zl)
        ax.grid(False)
        ax.set_title(f"$\\bf{{{NAMES[m]}}}$  {sub[m]}", loc="left", color=INK, fontsize=7, pad=2)
        if m in ("l1", "f2", "e2"):
            ax.set_xlabel("x", labelpad=1)
        else:
            ax.tick_params(labelbottom=False)
        if m == "l1":
            ax.set_ylabel("z", labelpad=1)
        else:
            ax.tick_params(labelleft=m in ("f1", "f2"))
    # rectangle axes: horizontal = emulation (Problem 2), vertical = physical misspecification (Problem 1)
    b1, b2 = axes["f1"].get_position(), axes["e1"].get_position()
    fig.text((b1.x0 + b2.x1) / 2, b1.y1 + 0.075, "model → emulator  (emulation error)", fontsize=7, color=MUTED,
             ha="center")
    b3 = axes["e2"].get_position()
    fig.text(b3.x1 + 0.012, (b3.y0 + b2.y1) / 2, "mild → strong  (parameter misspecification)", fontsize=7,
             color=MUTED, ha="left", va="center", rotation=-90)
    fig.text(0.5, -0.03, f"x–z projection of a {r['l1_attr'].shape[0] * float(r['dt']) - float(r['dt']):g}-TU free run "
             "from the same initial state; grey: L1", fontsize=6, color=MUTED, ha="center")
    save(fig, "fig_models_attractor")
    plt.close(fig)


def stats_band(ax, t, a, m, label=True):
    """median line for every model; the 25-75 % band only for the L1 reference (bands of 5 models overlap)."""
    q25, q50, q75 = np.nanpercentile(a, [25, 50, 75], axis=0)
    if m == "l1":
        ax.fill_between(t, q25, q75, color=COL[m], alpha=0.15, lw=0, label="L1 25–75 %")
    ax.plot(t, q50, color=COL[m], ls=LS[m], lw=1.2, label=NAMES[m] if label else None)


def fig_growth(r, plt, cfg):
    t = np.arange(r["l1_e_win"].shape[1]) * float(r["dt"])
    sat = np.sqrt(2 * r["var_clim"].mean())                     # RMS distance of two independent climate states
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.7), sharey=True)
    for ax, key, title in ((axs[0], "win", f"(a) {cfg['n_win']} windows × {cfg['members']} members"),
                           (axs[1], "fix", f"(b) one window, {cfg['n_fixed']} members")):
        window_band(ax)
        for m in MODELS:
            stats_band(ax, t, r[f"{m}_e_{key}"], m)
        ax.axhline(sat, color=MUTED, lw=0.7, ls="-.")
        ax.text(WINDOW_TU + 0.1, sat * 1.08, "saturation", fontsize=6, color=MUTED, ha="left", va="bottom")
        ax.axhline(SIGMA_O, color=MUTED, lw=0.7, ls=":")
        ax.text(t[-1], SIGMA_O * 1.08, "σ$_o$", fontsize=7, color=MUTED, ha="right", va="bottom")
        ax.set(yscale="log", xlim=(0, t[-1]), xlabel="lead time [TU]")
        ax.set_title(title, loc="left")
    axs[0].set_ylabel("ensemble error |member − own control| (RMS)")
    axs[0].set_ylim(0.5, sat * 2.5)
    axs[0].legend(loc="lower right", ncol=2, handlelength=2.2)
    fig.text(0.5, -0.04, "line: median; grey band: L1 25–75 %; error measured against each model's own unperturbed forecast "
             "(no truth)", fontsize=6, color=MUTED, ha="center")
    fig.tight_layout()
    save(fig, "fig_models_growth")
    plt.close(fig)


def tdec(rho, thr, dt):
    below = rho < thr
    return np.where(below.any(1), below.argmax(1), np.nan) * dt


def fig_decorr(r, plt, cfg):
    dt, thr = float(r["dt"]), cfg["rho_thr"]
    t = np.arange(r["l1_rho_win"].shape[1]) * dt
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.6), gridspec_kw={"width_ratios": [1, 1, 1.05]})

    def common(ax, ylab, ylim):
        window_band(ax)
        ax.axhline(thr, color=MUTED, lw=0.7, ls="--")
        ax.axhline(0, color=MUTED, lw=0.5)
        ax.text(t[-1], thr + 0.02, f"{thr:g}", fontsize=6, color=MUTED, ha="right", va="bottom")
        ax.set(xlim=(0, t[-1]), ylim=ylim, xlabel="lead time [TU]", ylabel=ylab)

    ax = axs[0]
    for m in MODELS:
        stats_band(ax, t, r[f"{m}_rho_win"], m)
    common(ax, "ρ(t) = 1 − |d|² / (2 Σ σ$_v^2$)", (-0.3, 1.03))
    ax.set_title("(a) distance-based, per run", loc="left")

    ax = axs[1]
    for m in MODELS:
        ax.fill_between(t, r[f"{m}_pr_lo"], r[f"{m}_pr_hi"], color=COL[m], alpha=0.25, lw=0)
        ax.plot(t, r[f"{m}_pr"], color=COL[m], ls=LS[m], lw=1.2, label=NAMES[m])
    common(ax, "Pearson ρ$_P$(t) across runs", (-0.3, 1.03))
    ax.set_title("(b) Pearson, member vs control", loc="left")
    ax.legend(loc="upper right", ncol=2, handlelength=2.2)

    ax = axs[2]
    pos = np.arange(len(MODELS))
    for off, key, alpha in ((-0.22, "win", 0.55), (0.0, "fix", 0.25)):
        data = []
        for m in MODELS:
            x = tdec(r[f"{m}_rho_{key}"], thr, dt)
            data.append(x[np.isfinite(x)])
        bp = ax.boxplot(data, positions=pos + off, widths=0.2, whis=(5, 95), showfliers=False, patch_artist=True,
                        medianprops={"color": INK, "lw": 1}, whiskerprops={"color": MUTED, "lw": 0.7},
                        capprops={"color": MUTED, "lw": 0.7}, boxprops={"lw": 0})
        for bx, m in zip(bp["boxes"], MODELS):
            bx.set_facecolor(COL[m])
            bx.set_alpha(alpha)
    for k, m in enumerate(MODELS):
        tp = first_below(r[f"{m}_pr"], thr, dt)
        lo, hi = np.nanpercentile(r[f"{m}_pr_tdec_boot"], [2.5, 97.5])
        ax.errorbar(k + 0.24, tp, yerr=[[tp - lo], [hi - tp]], fmt="D", color=COL[m], mec=INK, mew=0.5, ms=4,
                    ecolor=INK, elinewidth=0.8, capsize=2)
    ax.axhline(WINDOW_TU, color=MUTED, lw=0.8, ls=":")
    ax.text(1.01, WINDOW_TU, f"DA window\n{WINDOW_TU:g} TU", transform=ax.get_yaxis_transform(), fontsize=6,
            color=MUTED, ha="left", va="center")
    ax.set_xticks(pos, [NAMES[m] for m in MODELS])
    ax.set(ylabel=f"decorrelation time (< {thr:g}) [TU]", ylim=(0, None))
    ax.set_title("(c) decorrelation time", loc="left")
    fig.text(0.5, -0.06,
             f"(a) median over {cfg['n_win'] * cfg['members']} runs, grey band: L1 25–75 %;  (b) across the same runs, "
             "band: 95 % window-bootstrap CI;  (c) boxes: (a) per run, dark = 200 windows, light = one window "
             "(25–75 %, whiskers 5–95 %); diamond: (b) with 95 % CI", fontsize=6, color=MUTED, ha="center", wrap=True)
    fig.tight_layout()
    save(fig, "fig_models_decorr")
    plt.close(fig)


def summary(r, cfg):
    dt, thr = float(r["dt"]), cfg["rho_thr"]
    rows = ["| model | median e @0 / 0.8 / 1.1 / 3 / 5 / 10 TU (windows) | T_dec median [5 %, 95 %] windows | fixed "
            "| Pearson ρ_P @0.8 / 1.1 TU [95 % CI] | Pearson T_dec [95 % CI] |", "|---|---|---|---|---|---|"]
    for m in MODELS:
        e = np.median(r[f"{m}_e_win"], 0)
        idx = [int(round(L / dt)) for L in (0, 0.8, 1.1, 3, 5, 10) if L / dt < len(e)]
        cells = []
        for key in ("win", "fix"):
            x = tdec(r[f"{m}_rho_{key}"], thr, dt)
            x = x[np.isfinite(x)]
            cells.append(f"{np.median(x):.2f} [{np.percentile(x, 5):.2f}, {np.percentile(x, 95):.2f}] (n {len(x)})")
        pr = " / ".join(f"{r[f'{m}_pr'][i]:.3f} [{r[f'{m}_pr_lo'][i]:.3f}, {r[f'{m}_pr_hi'][i]:.3f}]"
                        for i in (int(round(0.8 / dt)), int(round(1.1 / dt))))
        tb = r[f"{m}_pr_tdec_boot"]
        tp = f"{first_below(r[f'{m}_pr'], thr, dt):.2f} [{np.nanpercentile(tb, 2.5):.2f}, {np.nanpercentile(tb, 97.5):.2f}]"
        rows.append(f"| {NAMES[m]} | {' / '.join(f'{e[i]:.3g}' for i in idx)} | {cells[0]} | {cells[1]} | {pr} | {tp} |")
    txt = "\n".join(rows)
    (ROOT / "tmp/results/models/fig_models_summary.md").write_text(txt + "\n")
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=int, default=50, help="short forecast before t0 [steps] (0.5 TU, pilot)")
    ap.add_argument("--s2", type=float, default=2.0)
    ap.add_argument("--horizon", type=int, default=1000, help="lead steps after t0 (10 TU)")
    ap.add_argument("--members", type=int, default=10, help="noise draws per test window (windows study)")
    ap.add_argument("--n-fixed", type=int, default=1000, help="noise draws in the fixed-window study")
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--spinup", type=int, default=1000, help="attractor run: discarded steps")
    ap.add_argument("--attr-steps", type=int, default=10000, help="attractor run: plotted steps (100 TU)")
    ap.add_argument("--rho-thr", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000, help="window bootstrap resamples for the Pearson CI")
    ap.add_argument("--window-tu", type=float, default=0.6, help="assimilation window drawn in the figures [TU]")
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    global WINDOW_TU
    WINDOW_TU = a.window_tu
    if not a.plot_only:
        compute(a)
    r = dict(np.load(NPZ))
    cfg = eval(str(r["config"]))                                 # noqa: S307 (own file)
    cfg["n_win"] = r["l1_e_win"].shape[0] // cfg["members"]
    (ROOT / "tmp/figs").mkdir(parents=True, exist_ok=True)
    plt = style()
    fig_attractor(r, plt)
    fig_growth(r, plt, cfg)
    fig_decorr(r, plt, cfg)
    summary(r, cfg)


if __name__ == "__main__":
    main()
