"""
M8 comparison figures (branch m8-classic-4dv): every optimizer superposed, per estimation model F1, F2, E1, E2.

    python src/fig_classic_4dv.py            reads tmp/results/m8/test/<model>/*.nc, writes tmp/figs/m8cmp_*.{png,pdf}

Figures (one per scheme x obs unless stated; panels = models; median line + interquartile band over the test cases):
    m8cmp_cost_<c>_<p>        normalized cost  (J^k - J*) / (J^0 - J*)  vs iteration, J* = lowest J reached by ANY method
                              on that case (all iterates), J^0 = J(first guess) (same for every method)
    m8cmp_icerr_xyz_<c>_<p>   per-variable IC error |x0_v^k - x_true,v(0)| vs iteration (dashed: first guess)
    m8cmp_icerr_a_xyz         per-variable IC error at the analysis, box per method, rows scheme x obs
    m8cmp_fcerr_xyz_<c>_<p>   per-variable trajectory error |x_v(t) - x_true,v(t)| over the window + 5-TU forecast,
                              trajectory run by the estimation model (weak: with the analysed eta in the window)
                              (per-variable figures: one common y range inside each figure, pilot 2026-11-25)
    m8cmp_traj_<c>_<p>        example case: x, y, z of the truth, first guess and every analysis trajectory
    m8cmp_geom_<model>        strong constraint only (control = x0, 3-D): cost J on the plane through the first guess
                              x_b, the truth x_true and the L-BFGS analysis; iterate paths of every method projected on
                              that plane (a slice: the paths may leave it); rows = obs pattern, columns = 2 cases
    m8cmp_geom_xyz_<model>    strong only: J on the coordinate planes x-y, x-z, y-z through the L-BFGS analysis (third
                              coordinate fixed at its analysis value); paths projected on each plane; rows = obs x case
Curves stop at the last iteration any case of that method is still active (frozen cases keep their last value).
New methods (e.g. meta runs written to the same tree) are picked up automatically, next colour slot.
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
from run_4dvar_v2 import setup_problem  # noqa: E402
from dameta.da import FourDVarCost, forecast_from_analysis  # noqa: E402
from dameta.da.cost import window_trajectory  # noqa: E402

torch.set_default_dtype(torch.float64)
RES = ROOT / "tmp/results/m8"
MODELS = ("f1", "f2", "e1", "e2")
CONSTRAINTS = ("strong", "weak")
PATTERNS = ("full", "x_only")
ORDER = ("gd", "adagrad", "adam", "lbfgs_wolfe")
LABEL = {"gd": "GD", "adagrad": "Adagrad", "adam": "Adam", "lbfgs_wolfe": "L-BFGS"}
# categorical slots (validated: src dataviz validator, light surface; CVD ΔE >= 9.2) + line style as 2nd encoding
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4"]
DASH = ["-", "--", "-.", (0, (1, 1.2)), (0, (5, 1, 1, 1))]
FG, TRUTH = "#8a8984", INK
DT = 0.01


def method_keys(runs):
    found = sorted({k[3] for k in runs}, key=lambda m: (ORDER.index(m) if m in ORDER else 99, m))
    return {m: (LABEL.get(m, m), SLOTS[i % len(SLOTS)], DASH[i % len(DASH)]) for i, m in enumerate(found)}


def collect():
    runs = {}
    for p in sorted((RES / "test").glob("*/*.nc")):
        ds = xr.load_dataset(p)
        cfg = json.loads(ds.attrs["config"])
        runs[(cfg["model"], cfg["constraint"], cfg["pattern"], cfg["method"])] = (ds, cfg)
    return runs


def band(ax, x, a, color, ls, label, lw=1.2):
    with np.errstate(all="ignore"):
        q = np.nanpercentile(a, [25, 50, 75], axis=0)
    ax.fill_between(x, q[0], q[2], color=color, alpha=0.14, lw=0)
    ax.plot(x, q[1], color=color, ls=ls, lw=lw, label=label)


def win(ax, window, label=True):
    """Assimilation window [0, window * DT] TU: shaded band, dashed end line, length label at the top."""
    ax.axvspan(0, window * DT, color="#e6e5e0", alpha=0.9, lw=0, zorder=0)
    ax.axvline(window * DT, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=0.5)
    if label:
        ax.text(window * DT, 1.0, f" window {window * DT:.1f} TU", transform=ax.get_xaxis_transform(), fontsize=5.5,
                color=MUTED, ha="left", va="top")


def k_stop(ds):
    return int(min(ds.sizes["k"] - 1, np.nanmax(ds.n_grad_evals.values)))


def grid_fig(plt, title):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8), sharex=True)
    fig.suptitle(title, x=0.01, ha="left", y=1.035, color=MUTED, fontsize=7)
    return fig, axes.ravel()


def finish(fig, axes, name, legend_extra=()):
    h, lab = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:                                              # union of handles over panels
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in lab:
                h.append(hh)
                lab.append(ll)
    fig.legend(h, lab, loc="upper center", ncol=len(lab), bbox_to_anchor=(0.5, 1.0), handlelength=2.6)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, name)


# ------------------------------------------------------------------------------------------------------- cost / IC error
def fig_cost(runs, meths, plt, c, p):
    fig, axes = grid_fig(plt, f"{c} · {p.replace('_', '-')} obs — normalized cost $(J^k - J^*)/(J^0 - J^*)$, "
                              "J* = best J of any method per case")
    for ax, m in zip(axes, MODELS):
        keys = [k for k in runs if k[:3] == (m, c, p)]
        if not keys:
            continue
        j_star = np.nanmin(np.stack([np.nanmin(runs[k][0].J_iter.values, 1) for k in keys]), 0)
        for k in sorted(keys, key=lambda k: list(meths).index(k[3])):
            ds = runs[k][0]
            ks = k_stop(ds)
            j = ds.J_iter.values[:, : ks + 1]
            jn = (j - j_star[:, None]) / (j[:, :1] - j_star[:, None])
            jn = np.clip(jn, 1e-12, None)
            lab, col, ls = meths[k[3]]
            band(ax, np.arange(ks + 1) + 1, jn, col, ls, lab)
        ax.set(xscale="log", yscale="log", ylim=(1e-8, 2))
        ax.set_title(NAMES[m], loc="left", color=INK)
    for ax in axes[2:]:
        ax.set_xlabel("iteration k + 1")
    for ax in axes[::2]:
        ax.set_ylabel("normalized cost")
    finish(fig, axes, f"m8cmp_cost_{c}_{p}")
    plt.close(fig)


# --------------------------------------------------------------------------------------------------- trajectories
@torch.no_grad()
def trajectories(runs):
    """Analysis and first-guess trajectories [n, L + F + 1, 3] per run (estimation model), cached in an npz."""
    cache = RES / "traj_cmp.npz"
    if cache.exists():
        z = np.load(cache)
        if set(z.files) >= {"|".join(k) for k in runs}:
            return {tuple(f.split("|")): z[f] for f in z.files}
    out, probs = {}, {}
    for k, (ds, cfg) in runs.items():
        mk = (k[0], k[1], k[2])
        if mk not in probs:
            probs[mk] = setup_problem(OmegaConf.create(cfg), cfg["split"], cfg.get("n_cases"))
        P = probs[mk]
        step, ccfg, F = P["cost"].step, P["ccfg"], int(cfg["forecast"])
        x0 = torch.tensor(ds.x0_a.values)
        eta = torch.nan_to_num(torch.tensor(ds.eta_final.values)) if P["weak"] else None
        tr = torch.cat([window_trajectory(step, x0, ccfg, eta)[:, : ccfg.window],
                        forecast_from_analysis(step, x0, ccfg, F, eta)], 1).numpy()
        tr[ds.diverged.values] = np.nan
        out[k] = tr
        fg = k[:3] + ("first_guess",)
        if fg not in out:
            out[fg] = torch.cat([window_trajectory(step, P["x0_b"], ccfg)[:, : ccfg.window],
                                 forecast_from_analysis(step, P["x0_b"], ccfg, F)], 1).numpy()
    np.savez_compressed(cache, **{"|".join(k): v for k, v in out.items()})
    return out


def fig_traj(runs, meths, traj, xt, window, case, obs, plt, c, p):
    fig, axes = plt.subplots(len(MODELS), 3, figsize=(7.2, 7.6), sharex=True)
    t = np.arange(xt.shape[1]) * DT
    for i, m in enumerate(MODELS):
        for v in range(3):
            ax = axes[i, v]
            win(ax, window, label=v == 0)
            ax.plot(t, xt[case, :, v], color=TRUTH, lw=1.6, label="truth (L1)")
            if obs["mask"][0, v]:                                        # observed component: noisy obs y_i
                ax.plot(obs["t"], obs["y"][case, :, v], "o", ms=3.2, mfc="white", mec=TRUTH, mew=0.8,
                        label=r"observation ($\sigma_o^2$ = 2)", zorder=5)
            fg = (m, c, p, "first_guess")
            if fg in traj:
                ax.plot(t, traj[fg][case, :, v], color=FG, lw=0.9, ls=(0, (4, 2)), label="first guess")
            for me in meths:
                if (m, c, p, me) in traj:
                    lab, col, ls = meths[me]
                    ax.plot(t, traj[(m, c, p, me)][case, :, v], color=col, ls=ls, lw=0.9, label=lab)
            if i == 0:
                ax.set_title("xyz"[v], color=INK)
            if v == 0:
                ax.set_ylabel(NAMES[m], color=INK, fontweight="bold")
            if i == len(MODELS) - 1:
                ax.set_xlabel("time [TU]")
    fig.suptitle(f"{c} · {p.replace('_', '-')} obs — test case {case}: analysis trajectories (estimation model) "
                 "and truth; shaded: assimilation window", x=0.01, ha="left", y=1.02, color=MUTED, fontsize=7)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), handlelength=2.6)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    save(fig, f"m8cmp_traj_{c}_{p}")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------- per-variable (x, y, z)
VARS = ("x", "y", "z")


def xyz_fig(plt, title):
    fig, axes = plt.subplots(len(MODELS), 3, figsize=(7.2, 7.6), sharex=True, sharey=True)   # one y range per figure
    fig.suptitle(title, x=0.01, ha="left", y=1.02, color=MUTED, fontsize=7)
    for v in range(3):
        axes[0, v].set_title(VARS[v], color=INK)
    for i, m in enumerate(MODELS):
        axes[i, 0].set_ylabel(NAMES[m], color=INK, fontweight="bold")
    return fig, axes


def xyz_finish(fig, axes, name):
    axes.flat[0].set_ylim(bottom=0)            # shared y: set once, after every panel is drawn (common range)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=len(lab), bbox_to_anchor=(0.5, 1.0), handlelength=2.6)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save(fig, name)


def fig_icerr_xyz(runs, meths, plt, c, p):
    fig, axes = xyz_fig(plt, f"{c} · {p.replace('_', '-')} obs — per-variable IC error $|x_{{0,v}}^k - "
                             "x_{true,v}(0)|$ vs iteration (dashed: first guess, median)")
    for i, m in enumerate(MODELS):
        keys = sorted([k for k in runs if k[:3] == (m, c, p)], key=lambda k: list(meths).index(k[3]))
        if not keys:
            continue
        x0t = runs[keys[0]][0].x_true0.values
        for v in range(3):
            ax = axes[i, v]
            for k in keys:
                ds = runs[k][0]
                ks = k_stop(ds)
                lab, col, ls = meths[k[3]]
                band(ax, np.arange(ks + 1) + 1, np.abs(ds.x0_iter.values[:, : ks + 1, v] - x0t[:, None, v]), col, ls,
                     lab, lw=1.0)
            e0 = np.median(np.abs(runs[keys[0]][0].x0_iter.values[:, 0, v] - x0t[:, v]))
            ax.axhline(e0, color=FG, lw=0.9, ls=(0, (4, 2)), label="first guess (median)")
            ax.set_xscale("log")
            if i == len(MODELS) - 1:
                ax.set_xlabel("iteration k + 1")
    xyz_finish(fig, axes, f"m8cmp_icerr_xyz_{c}_{p}")
    plt.close(fig)


def fig_fcerr_xyz(runs, meths, traj, xt, window, plt, c, p):
    fig, axes = xyz_fig(plt, f"{c} · {p.replace('_', '-')} obs — per-variable trajectory error $|x_v(t) - "
                             "x_{true,v}(t)|$ (estimation model)")
    t = np.arange(xt.shape[1]) * DT
    for i, m in enumerate(MODELS):
        for v in range(3):
            ax = axes[i, v]
            win(ax, window, label=(i == 0 and v == 0))
            fg = (m, c, p, "first_guess")
            if fg in traj:
                ax.plot(t, np.nanmedian(np.abs(traj[fg][..., v] - xt[..., v]), 0), color=FG, lw=0.9,
                        ls=(0, (4, 2)), label="first guess (median)")
            for me in meths:
                if (m, c, p, me) in traj:
                    lab, col, ls = meths[me]
                    band(ax, t, np.abs(traj[(m, c, p, me)][..., v] - xt[..., v]), col, ls, lab, lw=0.9)
            if i == len(MODELS) - 1:
                ax.set_xlabel("time from window start [TU]")
    xyz_finish(fig, axes, f"m8cmp_fcerr_xyz_{c}_{p}")
    plt.close(fig)


def fig_icerr_a_xyz(runs, meths, plt):
    combos = [(c, p) for c in CONSTRAINTS for p in PATTERNS]
    fig, axes = plt.subplots(len(combos), 3, figsize=(7.2, 7.6), sharey=True)
    w = 0.8 / (len(meths) + 1)
    for r, (c, p) in enumerate(combos):
        for v in range(3):
            ax = axes[r, v]
            for i, m in enumerate(MODELS):
                keys = [k for k in runs if k[:3] == (m, c, p)]
                if not keys:
                    continue
                ds0 = runs[keys[0]][0]
                x0t = ds0.x_true0.values[:, v]
                series = [("first guess", FG, np.abs(ds0.x0_iter.values[:, 0, v] - x0t))] + \
                         [(meths[me][0], meths[me][1], np.abs(runs[(m, c, p, me)][0].x0_a.values[:, v] - x0t))
                          for me in meths if (m, c, p, me) in runs]
                for j, (lab, col, val) in enumerate(series):
                    val = val[np.isfinite(val)]
                    bp = ax.boxplot(val, positions=[i + (j - len(series) / 2 + 0.5) * w], widths=w * 0.8,
                                    patch_artist=True, showfliers=False, medianprops={"color": INK, "lw": 0.8},
                                    whiskerprops={"color": MUTED, "lw": 0.5}, capprops={"color": MUTED, "lw": 0.5},
                                    boxprops={"lw": 0})
                    bp["boxes"][0].set_facecolor(col)
                    bp["boxes"][0].set_alpha(0.75)
                    if r == 0 and v == 0 and i == 0:
                        bp["boxes"][0].set_label(lab)
            ax.set_xticks(range(len(MODELS)), [NAMES[m] for m in MODELS])
            ax.grid(axis="x", visible=False)
            if r == 0:
                ax.set_title(VARS[v], color=INK)
            if v == 0:
                ax.set_ylabel(f"{c} · {p.replace('_', '-')}", color=INK)
    fig.suptitle("per-variable IC error at the analysis $|x_{a,v} - x_{true,v}(0)|$ (box: IQR, whiskers 1.5 IQR, "
                 "outliers hidden)", x=0.01, ha="left", y=1.02, color=MUTED, fontsize=7)
    xyz_finish(fig, axes, "m8cmp_icerr_a_xyz")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------------ geometry
@torch.no_grad()
def plane_cost(P, i, origin, e1, e2, extent, n=121):
    """J of case i on origin + a e1 + b e2, (a, b) in extent (a0, a1, b0, b1). Returns a, b, J [n, n]."""
    a = np.linspace(extent[0], extent[1], n)
    b = np.linspace(extent[2], extent[3], n)
    aa, bb = np.meshgrid(a, b)
    pts = torch.tensor(origin[None] + aa.ravel()[:, None] * e1[None] + bb.ravel()[:, None] * e2[None])
    g = pts.shape[0]
    base = P["cost"]
    cost = FourDVarCost(base.step, P["y"][i:i + 1].expand(g, -1, -1), P["mask"], P["x0_b"][i:i + 1].expand(g, -1),
                        P["ccfg"], weak=False, b_inv=P["b_inv"][i:i + 1].expand(g, -1, -1))
    return a, b, cost(pts).numpy().reshape(n, n)


def fig_geom(runs, meths, plt, m):
    keys = {p: [k for k in runs if k[:3] == (m, "strong", p)] for p in PATTERNS}
    if not any(keys.values()):
        return
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.6))
    for r, p in enumerate(PATTERNS):
        if not keys[p]:
            continue
        ds0, cfg0 = runs[keys[p][0]]
        P = setup_problem(OmegaConf.create(cfg0), cfg0["split"], cfg0.get("n_cases"))
        fge = ds0.sol_deg_b.values
        cases = [int(np.argmin(np.abs(fge - np.quantile(fge, q)))) for q in (0.5, 0.9)]
        ref = "lbfgs_wolfe" if (m, "strong", p, "lbfgs_wolfe") in runs else keys[p][0][3]
        for col_i, i in enumerate(cases):
            ax = axes[r, col_i]
            xb = P["x0_b"][i].numpy()
            xt = ds0.x_true0.values[i]
            xa = runs[(m, "strong", p, ref)][0].x0_a.values[i]
            e1 = (xt - xb) / np.linalg.norm(xt - xb)
            d = xa - xb - np.dot(xa - xb, e1) * e1
            if np.linalg.norm(d) < 1e-6 * np.linalg.norm(xt - xb):    # degenerate: any direction orthogonal to e1
                d = np.cross(e1, [0.0, 0.0, 1.0])
            e2 = d / np.linalg.norm(d)
            paths = {}
            for me in meths:
                if (m, "strong", p, me) in runs:
                    ds = runs[(m, "strong", p, me)][0]
                    xs = ds.x0_iter.values[i, : int(ds.n_grad_evals.values[i]) + 1] - xb
                    paths[me] = np.stack([xs @ e1, xs @ e2], 1)
            allp = np.concatenate([np.zeros((1, 2)), [[np.linalg.norm(xt - xb), 0.0]]] + list(paths.values()))
            lo, hi = allp.min(0), allp.max(0)
            span = max(hi - lo) * 0.25 + 0.3
            ext = (lo[0] - span, hi[0] + span, lo[1] - span, hi[1] + span)
            a, b, j = plane_cost(P, i, xb, e1, e2, ext)
            lev = np.linspace(np.log10(j.min()), np.quantile(np.log10(j), 0.95), 14)
            cs = ax.contourf(a, b, np.log10(j), levels=lev, cmap="Blues_r", extend="max", alpha=0.85)
            ax.contour(a, b, np.log10(j), levels=lev, colors="white", linewidths=0.3)
            for me, pth in paths.items():
                lab, col, ls = meths[me]
                ax.plot(pth[:, 0], pth[:, 1], color=col, ls=ls, lw=1.1, label=lab)
                ax.plot(pth[-1, 0], pth[-1, 1], "o", ms=5, mfc=col, mec="white", mew=0.8)
            ax.plot(0, 0, "s", ms=6, mfc=FG, mec="white", mew=0.8, label="first guess $x_b$")
            ax.plot(np.linalg.norm(xt - xb), 0, "*", ms=10, mfc=TRUTH, mec="white", mew=0.6, label="truth $x_{true}(0)$")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(False)
            ax.set_title(f"{p.replace('_', '-')} obs · case {i} (first-guess error q{50 if col_i == 0 else 90})",
                         loc="left", color=INK)
            ax.set_xlabel(r"along $x_{true} - x_b$")
            ax.set_ylabel(r"$\perp$ (toward the L-BFGS analysis)")
            cb = fig.colorbar(cs, ax=ax, shrink=0.8, pad=0.02, format="%.1f")
            cb.set_label(r"$\log_{10} J$", fontsize=7)
            cb.ax.tick_params(labelsize=6)
    fig.suptitle(f"{NAMES[m]}, strong constraint — cost on the plane (x_b, x_true, L-BFGS analysis) and projected "
                 "iterate paths (● analysis)", x=0.01, ha="left", y=1.02, color=MUTED, fontsize=7)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=len(lab), bbox_to_anchor=(0.5, 0.995), handlelength=2.6)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, f"m8cmp_geom_{m}")
    plt.close(fig)


def fig_geom_xyz(runs, meths, plt, m):
    keys = {p: [k for k in runs if k[:3] == (m, "strong", p)] for p in PATTERNS}
    if not any(keys.values()):
        return
    planes = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]                       # (horizontal, vertical, fixed) variable index
    rows = []
    for p in PATTERNS:
        if keys[p]:
            fge = runs[keys[p][0]][0].sol_deg_b.values
            rows += [(p, q, int(np.argmin(np.abs(fge - np.quantile(fge, q))))) for q in (0.5, 0.9)]
    fig, axes = plt.subplots(len(rows), 3, figsize=(7.2, 2.25 * len(rows)))
    axes = np.atleast_2d(axes)
    eye = np.eye(3)
    for r, (p, q, i) in enumerate(rows):
        ds0, cfg0 = runs[keys[p][0]]
        P = setup_problem(OmegaConf.create(cfg0), cfg0["split"], cfg0.get("n_cases"))
        xb, xt = P["x0_b"][i].numpy(), ds0.x_true0.values[i]
        ref = "lbfgs_wolfe" if (m, "strong", p, "lbfgs_wolfe") in runs else keys[p][0][3]
        xa = runs[(m, "strong", p, ref)][0].x0_a.values[i]
        paths = {}
        for me in meths:
            if (m, "strong", p, me) in runs:
                ds = runs[(m, "strong", p, me)][0]
                paths[me] = ds.x0_iter.values[i, : int(ds.n_grad_evals.values[i]) + 1]
        pts = np.concatenate([xb[None], xt[None]] + list(paths.values()))
        for col, (h, v, f) in enumerate(planes):
            ax = axes[r, col]
            lo, hi = pts[:, [h, v]].min(0), pts[:, [h, v]].max(0)
            span = max(hi - lo) * 0.25 + 0.3
            ext = (lo[0] - span, hi[0] + span, lo[1] - span, hi[1] + span)
            origin = xa * eye[f]                                         # fixed coordinate = analysis value
            a, b, j = plane_cost(P, i, origin, eye[h], eye[v], ext)
            lj = np.log10(j)
            lev = np.linspace(lj.min(), np.quantile(lj, 0.95), 14)
            cs = ax.contourf(a, b, lj, levels=lev, cmap="Blues_r", extend="max", alpha=0.85)
            ax.contour(a, b, lj, levels=lev, colors="white", linewidths=0.3)
            for me, pth in paths.items():
                lab, colr, ls = meths[me]
                ax.plot(pth[:, h], pth[:, v], color=colr, ls=ls, lw=1.0, label=lab)
                ax.plot(pth[-1, h], pth[-1, v], "o", ms=4.5, mfc=colr, mec="white", mew=0.7)
            ax.plot(xb[h], xb[v], "s", ms=5.5, mfc=FG, mec="white", mew=0.7, label="first guess $x_b$")
            ax.plot(xt[h], xt[v], "*", ms=9, mfc=TRUTH, mec="white", mew=0.6, label="truth $x_{true}(0)$")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(False)
            ax.set_xlabel(VARS[h], labelpad=1)
            ax.set_ylabel(VARS[v], labelpad=1)
            ax.set_title(f"{VARS[h]}–{VARS[v]} ({VARS[f]} = {xa[f]:.2f})", loc="left", color=INK, fontsize=7)
            cb = fig.colorbar(cs, ax=ax, shrink=0.85, pad=0.02, format="%.1f")
            cb.ax.tick_params(labelsize=5.5)
            if col == 2:
                cb.set_label(r"$\log_{10} J$", fontsize=6)
        axes[r, 0].text(-0.42, 0.5, f"{p.replace('_', '-')} obs\ncase {i} (fg err q{int(q * 100)})",
                        transform=axes[r, 0].transAxes, rotation=90, ha="center", va="center", fontsize=7,
                        color=INK, fontweight="bold")
    fig.suptitle(f"{NAMES[m]}, strong constraint — cost on the coordinate planes through the L-BFGS analysis (third "
                 "variable fixed) and projected iterate paths (● analysis)", x=0.01, ha="left", y=1.01, color=MUTED,
                 fontsize=7)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=len(lab), bbox_to_anchor=(0.5, 0.995), handlelength=2.6)
    fig.tight_layout(rect=(0.02, 0, 1, 0.975))
    save(fig, f"m8cmp_geom_xyz_{m}")
    plt.close(fig)


def main():
    plt = style()
    runs = collect()
    meths = method_keys(runs)
    xt_ds = xr.load_dataset(ROOT / "tmp/data/l63v2_test_seed42.nc")
    window = int(xt_ds.attrs["window"])
    for k, (ds, cfg) in runs.items():                                 # truth / first-guess IC on every record
        ds["x_true0"] = (("ic", "var"), xt_ds.x_true.values[: ds.sizes["ic"], 0])
        ds["sol_deg_b"] = (("ic",), np.sqrt(3) * ds.rmse_x0_b.values)
    traj = trajectories(runs)
    n = next(iter(runs.values()))[0].sizes["ic"]
    xt = xt_ds.x_true.values[:n, : next(iter(traj.values())).shape[1]]
    # example case: the median first-guess IC error of F1 (common to every figure)
    fge = runs[next(k for k in runs if k[0] == "f1")][0].sol_deg_b.values
    case = int(np.argmin(np.abs(fge - np.median(fge))))
    for c in CONSTRAINTS:
        for p in PATTERNS:
            fig_cost(runs, meths, plt, c, p)
            obs = {"t": xt_ds.t_obs.values * DT if "t_obs" in xt_ds.coords else np.arange(0, window + 1, 10) * DT,
                   "y": xt_ds.y_obs.values[:n], "mask": xt_ds.obs_mask.sel(pattern=p).values}
            fig_traj(runs, meths, traj, xt, window, case, obs, plt, c, p)
            fig_icerr_xyz(runs, meths, plt, c, p)
            fig_fcerr_xyz(runs, meths, traj, xt, window, plt, c, p)
    fig_icerr_a_xyz(runs, meths, plt)
    for m in MODELS:
        fig_geom(runs, meths, plt, m)
        fig_geom_xyz(runs, meths, plt, m)
    print(f"{len(runs)} runs, methods {list(meths)} -> tmp/figs/m8cmp_*")


if __name__ == "__main__":
    main()
