"""
M8 assimilation-window figures (branch m8-classic-4dv, pilot 2026-11-25): impact of the window length on the 4D-Var
optimization geometry and on the initial-condition (IC) correction.

    python src/fig_window_4dv.py      reads tmp/results/m8/feas/w<W>/<model>/*.nc (10-case val feasibility runs,
                                      B = 4 x hybrid B, obs every 20 steps, sigma_o2 = 4) -> tmp/figs/m8win_*.{png,pdf}

The ICs, observations at common times and first guesses are identical across windows (same D1 v3 draw; only the window
length and therefore the number of observation times change), so every difference between rows is a window effect.

Figures
    m8win_profile_<p>           strong: J along the segment x_b + s (x_true - x_b), s in [-0.5, 1.5], each case
                                min-max normalised on the segment; rows = window, columns = model (10 thin lines)
    m8win_geom_<p>_case<i>      strong: J on the plane (x_b, x_true, L-BFGS analysis) and the projected iterate paths of
                                every optimizer; rows = window, columns = model; one case (median / q90 first-guess error)
    m8win_nonpd                 share of non-positive-definite Hessians: over all iterates (bars, all optimizers pooled)
                                and at the analysis (markers); panels = scheme x obs; groups = model, bars = window
    m8win_lammin_a              lambda_min of the Hessian at the analysis per case (symlog), all optimizers pooled
    m8win_icerr_xyz_<o>_<c>_<p> per-variable IC error |x0_v^k - x_true,v(0)| vs iteration (median + IQR over cases);
                                rows = model, columns = x, y, z, colour = window (dotted: first guess)
    m8win_icerr_a_xyz_<c>_<p>   per-variable IC error at the analysis per case: first guess vs each window, every
                                optimizer (marker); rows = model, columns = x, y, z
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fig_models import INK, MUTED, NAMES, save, style  # noqa: E402
from fig_classic_4dv import FG, LABEL, ORDER, SLOTS, DASH, TRUTH, plane_cost  # noqa: E402
from run_4dvar_v2 import setup_problem  # noqa: E402

torch.set_default_dtype(torch.float64)
RES = ROOT / "tmp/results/m8/feas"
MODELS = ("f1", "f2", "e1", "e2")
CONSTRAINTS = ("strong", "weak")
PATTERNS = ("full", "x_only")
VARS = ("x", "y", "z")
DT = 0.01
# window = ordered quantity -> one sequential hue (light -> dark) + line style as second encoding
WCOL = ["#7fb0ea", "#2a78d6", "#0d3a73"]
WLS = [(0, (1, 1.2)), "--", "-"]
MARK = {"gd": "o", "adagrad": "s", "adam": "^", "lbfgs_wolfe": "D"}


def collect():
    runs, windows = {}, set()
    for p in sorted(RES.glob("w*/*/*.nc")):
        ds = xr.load_dataset(p)
        cfg = json.loads(ds.attrs["config"])
        w = int(p.parent.parent.name[1:])
        windows.add(w)
        runs[(w, cfg["model"], cfg["constraint"], cfg["pattern"], cfg["method"])] = (ds, cfg)
    return runs, sorted(windows)


def wlab(w):
    return f"{w * DT:.1f} TU"


_PROB = {}


def problem(cfg):
    """setup_problem cached per (window data dir, model, scheme, obs)."""
    # data moved 2026-11-26 (1.6 TU chosen): tmp/data (0.6 TU) -> tmp/data_w60, tmp/data_w160 -> tmp/data
    d = cfg["data_dir"].rstrip("/")
    cfg = {**cfg, "data_dir": d[:-len("_w160")] if d.endswith("tmp/data_w160") else
           (d + "_w60" if d.endswith("tmp/data") else d)}
    key = (cfg["data_dir"], cfg["model"], cfg["constraint"], cfg["pattern"])
    if key not in _PROB:
        _PROB[key] = setup_problem(OmegaConf.create(cfg), cfg["split"], cfg.get("n_cases"))
    return _PROB[key]


def legend_top(fig, axes, ncol=None, y=1.0):
    h, lab = [], []
    for ax in np.ravel(axes):
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in lab:
                h.append(hh)
                lab.append(ll)
    fig.legend(h, lab, loc="upper center", ncol=ncol or len(lab), bbox_to_anchor=(0.5, y), handlelength=2.6)


# ------------------------------------------------------------------------------------------------------------ geometry
@torch.no_grad()
def fig_profile(runs, windows, plt, p):
    fig, axes = plt.subplots(len(windows), len(MODELS), figsize=(7.2, 1.7 * len(windows) + 0.5), sharex=True,
                             sharey=True)
    s = np.linspace(-0.5, 1.5, 201)
    for r, w in enumerate(windows):
        for c, m in enumerate(MODELS):
            ax = axes[r, c]
            key = (w, m, "strong", p, "lbfgs_wolfe")
            if key not in runs:
                continue
            P = problem(runs[key][1])
            xb = P["x0_b"].numpy()
            xt = P["ds"].x_true.values[:, 0]
            for i in range(xb.shape[0]):
                pts = torch.tensor(xb[i][None] + s[:, None] * (xt[i] - xb[i])[None])
                cost = type(P["cost"])(P["cost"].step, P["y"][i:i + 1].expand(len(s), -1, -1), P["mask"],
                                       P["x0_b"][i:i + 1].expand(len(s), -1), P["ccfg"], weak=False,
                                       b_inv=P["b_inv"][i:i + 1].expand(len(s), -1, -1))
                j = cost(pts).numpy()
                jn = (j - j.min()) / (j.max() - j.min())
                ax.plot(s, jn, color=WCOL[r], lw=0.7, alpha=0.8)
            ax.axvline(0, color=FG, lw=0.6, ls=(0, (2, 2)))
            ax.axvline(1, color=TRUTH, lw=0.6, ls=(0, (2, 2)))
            if r == 0:
                ax.set_title(NAMES[m], loc="left", color=INK)
            if c == 0:
                ax.set_ylabel(f"window {wlab(w)}\nnormalised J")
            if r == len(windows) - 1:
                ax.set_xlabel(r"s  ($x_b$ at 0, $x_{true}(0)$ at 1)")
    fig.suptitle(f"Strong constraint, {p.replace('_', '-')} obs — cost along the segment first guess → truth, "
                 "every case (min–max normalised)", x=0.01, ha="left", y=1.0, color=MUTED, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, f"m8win_profile_{p}")
    plt.close(fig)


def fig_geom(runs, windows, plt, p, i, tag):
    meths = [me for me in ORDER if any(k[4] == me for k in runs)]
    fig, axes = plt.subplots(len(windows), len(MODELS), figsize=(7.4, 1.95 * len(windows) + 0.6))
    for r, w in enumerate(windows):
        for c, m in enumerate(MODELS):
            ax = axes[r, c]
            ref = (w, m, "strong", p, "lbfgs_wolfe")
            if ref not in runs:
                ax.set_visible(False)
                continue
            P = problem(runs[ref][1])
            xb = P["x0_b"][i].numpy()
            xt = P["ds"].x_true.values[i, 0]
            xa = runs[ref][0].x0_a.values[i]
            e1 = (xt - xb) / np.linalg.norm(xt - xb)
            d = xa - xb - np.dot(xa - xb, e1) * e1
            if np.linalg.norm(d) < 1e-6 * np.linalg.norm(xt - xb):
                d = np.cross(e1, [0.0, 0.0, 1.0])
            e2 = d / np.linalg.norm(d)
            paths = {}
            for me in meths:
                if (w, m, "strong", p, me) in runs:
                    ds = runs[(w, m, "strong", p, me)][0]
                    xs = ds.x0_iter.values[i, : int(ds.n_grad_evals.values[i]) + 1] - xb
                    xs = xs[np.isfinite(xs).all(1)]
                    paths[me] = np.stack([xs @ e1, xs @ e2], 1)
            allp = np.concatenate([np.zeros((1, 2)), [[np.linalg.norm(xt - xb), 0.0]]] + list(paths.values()))
            lo, hi = allp.min(0), allp.max(0)
            mid, half = (lo + hi) / 2, max(hi - lo) * 0.65 + 0.5          # square extent: panels align
            ext = (mid[0] - half, mid[0] + half, mid[1] - half, mid[1] + half)
            a, b, j = plane_cost(P, i, xb, e1, e2, ext, n=101)
            lj = np.log10(j)
            lev = np.linspace(lj.min(), np.quantile(lj, 0.95), 14)
            cs = ax.contourf(a, b, lj, levels=lev, cmap="Blues_r", extend="max", alpha=0.85)
            ax.contour(a, b, lj, levels=lev, colors="white", linewidths=0.3)
            for k, (me, pth) in enumerate(paths.items()):
                col, ls = SLOTS[ORDER.index(me)], DASH[ORDER.index(me)]
                ax.plot(pth[:, 0], pth[:, 1], color=col, ls=ls, lw=0.9, label=LABEL[me])
                ax.plot(pth[-1, 0], pth[-1, 1], "o", ms=4, mfc=col, mec="white", mew=0.6)
            ax.plot(0, 0, "s", ms=5, mfc=FG, mec="white", mew=0.7, label="first guess $x_b$")
            ax.plot(np.linalg.norm(xt - xb), 0, "*", ms=8.5, mfc=TRUTH, mec="white", mew=0.5,
                    label="truth $x_{true}(0)$")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(False)
            ax.tick_params(labelsize=5.5)
            cb = fig.colorbar(cs, ax=ax, shrink=0.8, pad=0.02, format="%.1f")
            cb.ax.tick_params(labelsize=5)
            if r == 0:
                ax.set_title(NAMES[m], loc="left", color=INK)
            if c == 0:
                ax.set_ylabel(f"window {wlab(w)}\n" + r"$\perp$ (toward analysis)", fontsize=6.5)
            if r == len(windows) - 1:
                ax.set_xlabel(r"along $x_{true} - x_b$", fontsize=6.5)
            if c == len(MODELS) - 1:
                cb.set_label(r"$\log_{10} J$", fontsize=6)
    fig.suptitle(f"Strong constraint, {p.replace('_', '-')} obs, case {i} ({tag}) — cost on the plane (x_b, x_true, "
                 "L-BFGS analysis) and projected iterate paths (● analysis)", x=0.01, ha="left", y=1.0, color=MUTED,
                 fontsize=7)
    legend_top(fig, axes, y=0.975)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, f"m8win_geom_{p}_case{i}")
    plt.close(fig)


def fig_nonpd(runs, windows, plt):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.4), sharey=True)
    width = 0.8 / len(windows)
    for a_i, (c, p) in enumerate([(c, p) for c in CONSTRAINTS for p in PATTERNS]):
        ax = axes.ravel()[a_i]
        for r, w in enumerate(windows):
            it_frac, a_frac = [], []
            for m in MODELS:
                lam_it, lam_a = [], []
                for me in ORDER:
                    if (w, m, c, p, me) in runs:
                        ds = runs[(w, m, c, p, me)][0]
                        v = ds.lam_min.values
                        lam_it.append(v[np.isfinite(v)])
                        lam_a.append(ds.lam_min_a.values)
                lam_it, lam_a = np.concatenate(lam_it), np.concatenate(lam_a)
                it_frac.append((lam_it <= 0).mean())
                a_frac.append((lam_a <= 0).mean())
            x = np.arange(len(MODELS)) + (r - (len(windows) - 1) / 2) * width
            ax.bar(x, it_frac, width * 0.92, color=WCOL[r], label=f"window {wlab(w)} — iterates")
            ax.plot(x, a_frac, "D", ms=4.5, mfc="white", mec=INK, mew=0.9, zorder=3,
                    label=f"window {wlab(w)} — analysis")
        ax.set_xticks(np.arange(len(MODELS)), [NAMES[m] for m in MODELS])
        ax.set_title(f"{c} · {p.replace('_', '-')} obs", loc="left", color=INK)
        if a_i % 2 == 0:
            ax.set_ylabel("share non-PD Hessian")
    fig.suptitle("Non-positive-definite Hessian of J (λ_min ≤ 0): bars = all iterates, open diamonds = analyses; "
                 "4 optimizers pooled, 10 val cases", x=0.01, ha="left", y=1.0, color=MUTED, fontsize=7)
    legend_top(fig, axes, ncol=3, y=0.975)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "m8win_nonpd")
    plt.close(fig)


def fig_lammin_a(runs, windows, plt):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.4), sharey=True)
    width = 0.8 / len(windows)
    rng = np.random.default_rng(0)
    for a_i, (c, p) in enumerate([(c, p) for c in CONSTRAINTS for p in PATTERNS]):
        ax = axes.ravel()[a_i]
        for r, w in enumerate(windows):
            for mi, m in enumerate(MODELS):
                x0 = mi + (r - (len(windows) - 1) / 2) * width
                for me in ORDER:
                    if (w, m, c, p, me) in runs:
                        v = runs[(w, m, c, p, me)][0].lam_min_a.values
                        ax.plot(x0 + rng.uniform(-0.3, 0.3, len(v)) * width, v, MARK[me], ms=2.6, mfc="none",
                                mec=WCOL[r], mew=0.6, alpha=0.9,
                                label=f"window {wlab(w)}" if (mi == 0 and me == "lbfgs_wolfe") else None)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_xticks(np.arange(len(MODELS)), [NAMES[m] for m in MODELS])
        ax.set_title(f"{c} · {p.replace('_', '-')} obs", loc="left", color=INK)
        if a_i % 2 == 0:
            ax.set_ylabel(r"$\lambda_{min}(H)$ at the analysis")
    for me in ORDER:                                                     # optimizer marker key
        axes[0, 0].plot([], [], MARK[me], ms=3.5, mfc="none", mec=MUTED, label=LABEL[me])
    fig.suptitle("Smallest Hessian eigenvalue at the analysis per case (symlog; < 0 = saddle / non-convex), "
                 "10 val cases × 4 optimizers", x=0.01, ha="left", y=1.0, color=MUTED, fontsize=7)
    legend_top(fig, axes, y=0.975)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "m8win_lammin_a")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------- IC correction
def fig_icerr_xyz(runs, windows, plt, me, c, p):
    fig, axes = plt.subplots(len(MODELS), 3, figsize=(7.0, 1.55 * len(MODELS) + 0.5), sharex=True, sharey=True)
    for mi, m in enumerate(MODELS):
        for r, w in enumerate(windows):
            key = (w, m, c, p, me)
            if key not in runs:
                continue
            ds, cfg = runs[key]
            xt = problem(cfg)["ds"].x_true.values[:, 0]
            ks = int(min(ds.sizes["k"] - 1, np.nanmax(ds.n_grad_evals.values)))
            e = np.abs(ds.x0_iter.values[:, : ks + 1] - xt[:, None])           # [ic, k, var]
            k = np.arange(ks + 1) + 1
            for v in range(3):
                ax = axes[mi, v]
                with np.errstate(all="ignore"):
                    q = np.nanpercentile(e[..., v], [25, 50, 75], axis=0)
                ax.fill_between(k, q[0], q[2], color=WCOL[r], alpha=0.15, lw=0)
                ax.plot(k, q[1], color=WCOL[r], ls=WLS[r], lw=1.2, label=f"window {wlab(w)}")
                if r == 0:
                    ax.axhline(np.median(e[:, 0, v]), color=FG, lw=0.8, ls=(0, (1, 2)), label="first guess (median)")
        for v in range(3):
            ax = axes[mi, v]
            ax.set_xscale("log")
            if mi == 0:
                ax.set_title(f"{VARS[v]}", loc="left", color=INK)
            if v == 0:
                ax.set_ylabel(f"{NAMES[m]}\n" + r"$|x_{0}^k - x_{true}(0)|$")
            if mi == len(MODELS) - 1:
                ax.set_xlabel("iteration k + 1")
    fig.suptitle(f"{LABEL[me]}, {c} · {p.replace('_', '-')} obs — per-variable IC error vs iteration "
                 "(median + IQR over 10 val cases)", x=0.01, ha="left", y=1.0, color=MUTED, fontsize=7)
    legend_top(fig, axes, y=0.975)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, f"m8win_icerr_xyz_{me}_{c}_{p}")
    plt.close(fig)


def fig_icerr_a_xyz(runs, windows, plt, c, p):
    fig, axes = plt.subplots(len(MODELS), 3, figsize=(7.0, 1.55 * len(MODELS) + 0.5), sharey=True)
    pos = {"fg": 0}
    pos.update({w: r + 1 for r, w in enumerate(windows)})
    offs = {me: (j - 1.5) * 0.12 for j, me in enumerate(ORDER)}
    for mi, m in enumerate(MODELS):
        for r, w in enumerate(windows):
            for me in ORDER:
                key = (w, m, c, p, me)
                if key not in runs:
                    continue
                ds, cfg = runs[key]
                xt = problem(cfg)["ds"].x_true.values[:, 0]
                ea = np.abs(ds.x0_a.values - xt)
                eb = np.abs(problem(cfg)["x0_b"].numpy() - xt)
                for v in range(3):
                    ax = axes[mi, v]
                    ax.plot(np.full(len(ea), pos[w] + offs[me]), ea[:, v], MARK[me], ms=2.8, mfc="none",
                            mec=WCOL[r], mew=0.7, label=LABEL[me] if (mi == 0 and v == 0 and r == 2) else None)
                    ax.plot(pos[w] + offs[me], np.median(ea[:, v]), "_", ms=7, mec=WCOL[r], mew=1.4)
                    if r == 0 and me == "lbfgs_wolfe":
                        ax.plot(np.full(len(eb), pos["fg"]), eb[:, v], "o", ms=2.8, mfc="none", mec=FG, mew=0.7)
                        ax.plot(pos["fg"], np.median(eb[:, v]), "_", ms=9, mec=FG, mew=1.4)
        for v in range(3):
            ax = axes[mi, v]
            ax.set_yscale("log")
            ax.set_xticks([0] + [r + 1 for r in range(len(windows))], ["first\nguess"] + [wlab(w) for w in windows],
                          fontsize=6)
            if mi == 0:
                ax.set_title(f"{VARS[v]}", loc="left", color=INK)
            if v == 0:
                ax.set_ylabel(f"{NAMES[m]}\n" + r"$|x_{0,a} - x_{true}(0)|$")
    fig.suptitle(f"{c} · {p.replace('_', '-')} obs — per-variable IC error at the analysis per case vs window length "
                 "(— median; markers = optimizer)", x=0.01, ha="left", y=1.0, color=MUTED, fontsize=7)
    legend_top(fig, axes, y=0.975)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, f"m8win_icerr_a_xyz_{c}_{p}")
    plt.close(fig)


def main():
    plt = style()
    runs, windows = collect()
    assert len(windows) <= len(WCOL), windows
    (ROOT / "tmp/figs").mkdir(parents=True, exist_ok=True)
    fig_nonpd(runs, windows, plt)
    fig_lammin_a(runs, windows, plt)
    for p in PATTERNS:
        fig_profile(runs, windows, plt, p)
        # cases: median / q90 first-guess error of F1 (first guess identical across windows)
        P = problem(runs[(windows[0], "f1", "strong", p, "lbfgs_wolfe")][1])
        fge = np.linalg.norm(P["x0_b"].numpy() - P["ds"].x_true.values[:, 0], axis=1)
        for q in (0.5, 0.9):
            fig_geom(runs, windows, plt, p, int(np.argmin(np.abs(fge - np.quantile(fge, q)))),
                     f"F1 first-guess error q{int(q * 100)}")
    for c in CONSTRAINTS:
        for p in PATTERNS:
            fig_icerr_a_xyz(runs, windows, plt, c, p)
            for me in ORDER:
                fig_icerr_xyz(runs, windows, plt, me, c, p)
    print(f"{len(runs)} runs, windows {windows} -> tmp/figs/m8win_*")


if __name__ == "__main__":
    main()
