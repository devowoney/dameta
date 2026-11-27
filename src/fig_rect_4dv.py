"""
M8-R figures (pilot 2026-11-26): strong / weak 4D-Var landscape on the model rectangle (shared truth-model first
guess / B per truth, chi2-tuned scalar Q, perfect-model reference strong only).

    python src/fig_rect_4dv.py [smoke|test]   reads tmp/results/m8r/<sub>/<truth>/<model>/*.nc -> tmp/figs/m8r_*

    m8r_land_<truth>_<pattern>   L-BFGS: landscape at the analysis per model (box over cases), strong vs weak side by
                                 side: lambda_min(H), kappa_H (PD analyses only, log), share of iterates with a non-PD
                                 Hessian, mean cos(x_true - x0^k, -grad) over the iterates
    m8r_geom_<truth>_<scheme>    J of one case (median first-guess error, full obs) on the coordinate planes x-y, x-z,
                                 y-z through each model's L-BFGS analysis (third variable fixed; weak: eta fixed at the
                                 analysed eta_a); rows = models of that truth (same truth, obs, x_b, B)
Truth / obs are read from each result's config (data_dir), so the F-truth stages use their own data.
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
sys.path.insert(0, str(ROOT / "src/scripts"))

from fig_models import COL, INK, MUTED, NAMES, save, style  # noqa: E402
from m8_lists import PATTERNS, STAGES  # noqa: E402
from run_4dvar_v2 import setup_problem  # noqa: E402
from dameta.da import FourDVarCost  # noqa: E402

torch.set_default_dtype(torch.float64)
VARS = ("x", "y", "z")
SCHEME_ALPHA = {"strong": 0.9, "weak": 0.35}


def collect(sub):
    runs = {}
    for tag, (_, _, ref, models) in STAGES.items():
        for m in (ref,) + models:
            for f in sorted((ROOT / "tmp/results/m8r" / sub / tag / m).glob("bice_*lbfgs_wolfe*.nc")):
                ds = xr.load_dataset(f)
                cfg = json.loads(ds.attrs["config"])
                runs[(tag, m, cfg["constraint"], cfg["pattern"])] = (ds, cfg)
    return runs


def finite(a):
    a = np.asarray(a, float)
    return a[np.isfinite(a)]


def fig_land(runs, plt, tag, p):
    models = [k[1] for k in runs if k[0] == tag and k[3] == p]
    models = list(dict.fromkeys(models))
    if not models:
        return
    panels = [("lam_min_a", r"$\lambda_{\min}(H)$ at the analysis", False),
              ("kappa_H_a", r"$\kappa_H$ at the analysis (PD only)", True),
              ("nonpd_it", "share of non-PD iterates", False),
              ("cos_it", r"mean $\cos(x_{true}-x_0^k, -\nabla J)$", False)]
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.4))
    for ax, (key, title, logy) in zip(axes, panels):
        for i, m in enumerate(models):
            for j, c in enumerate(("strong", "weak")):
                if (tag, m, c, p) not in runs:
                    continue
                ds = runs[(tag, m, c, p)][0]
                ok = ~ds.diverged.values
                if key == "nonpd_it":
                    nn = ds.n_neg.values[ok]
                    rec = np.isfinite(nn)
                    v = ((nn > 0) & rec).sum(1) / np.maximum(rec.sum(1), 1)
                elif key == "cos_it":
                    v = np.nanmean(ds.cos_truth.values[ok], 1)
                else:
                    v = ds[key].values[ok]
                v = finite(v)
                if logy:
                    v = v[v > 0]
                if v.size == 0:
                    continue
                pos = i + (j - 0.5) * 0.36
                bp = ax.boxplot(v, positions=[pos], widths=0.3, showfliers=False, patch_artist=True,
                                medianprops={"color": INK, "lw": 0.8}, whiskerprops={"lw": 0.6},
                                capprops={"lw": 0.6})
                bp["boxes"][0].set(facecolor=COL[m], alpha=SCHEME_ALPHA[c], lw=0.5)
        ax.set_xticks(range(len(models)), [NAMES[m] for m in models])
        if logy:
            ax.set_yscale("log")
        ax.set_title(title, loc="left", color=INK, fontsize=7)
        ax.grid(axis="x", visible=False)
    fig.suptitle(f"truth {tag}, {p.replace('_', '-')} obs, L-BFGS — left box strong, right box weak (light); "
                 f"{NAMES[STAGES[tag][2]]} = perfect-model reference (strong only)", x=0.01, ha="left", y=1.02,
                 color=MUTED, fontsize=7)
    fig.tight_layout()
    save(fig, f"m8r_land_{tag}_{p}")
    plt.close(fig)


@torch.no_grad()
def plane_cost(P, i, origin, e1, e2, ext, eta=None, n=101):
    """J of case i on origin + a e1 + b e2 (x0 plane); weak: eta fixed (= analysed eta of that run)."""
    a, b = np.linspace(ext[0], ext[1], n), np.linspace(ext[2], ext[3], n)
    aa, bb = np.meshgrid(a, b)
    pts = torch.tensor(origin[None] + aa.ravel()[:, None] * e1[None] + bb.ravel()[:, None] * e2[None])
    g = pts.shape[0]
    base = P["cost"]
    weak = eta is not None
    cost = FourDVarCost(base.step, P["y"][i:i + 1].expand(g, -1, -1), P["mask"], P["x0_b"][i:i + 1].expand(g, -1),
                        P["ccfg"], weak=weak, b_inv=P["b_inv"][i:i + 1].expand(g, -1, -1))
    j = cost(pts, torch.tensor(eta)[None].expand(g, -1, -1)) if weak else cost(pts)
    return a, b, j.numpy().reshape(n, n)


def fig_geom(runs, plt, tag, c, p="full"):
    keys = [k for k in runs if k[0] == tag and k[2] == c and k[3] == p]
    if not keys:
        return
    ds0, _ = runs[keys[0]]
    fge = ds0.rmse_x0_b.values
    i = int(np.argmin(np.abs(fge - np.median(fge))))                  # median first-guess error (shared x_b)
    eye = np.eye(3)
    planes = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
    fig, axes = plt.subplots(len(keys), 3, figsize=(7.2, 2.25 * len(keys)))
    axes = np.atleast_2d(axes)
    for r, k in enumerate(keys):
        ds, cfg = runs[k]
        P = setup_problem(OmegaConf.create(cfg), cfg["split"], cfg.get("n_cases"))
        xb = P["x0_b"][i].numpy()
        xt = P["ds"].x_true.values[i, 0]
        xa = ds.x0_a.values[i]
        eta = ds.eta_final.values[i] if c == "weak" else None
        path = ds.x0_iter.values[i, : int(ds.n_grad_evals.values[i]) + 1]
        path = path[np.isfinite(path).all(1)]
        pts = np.concatenate([xb[None], xt[None], path])
        for col, (h, v, f) in enumerate(planes):
            ax = axes[r, col]
            lo, hi = pts[:, [h, v]].min(0), pts[:, [h, v]].max(0)
            span = max(hi - lo) * 0.25 + 0.3
            ext = (lo[0] - span, hi[0] + span, lo[1] - span, hi[1] + span)
            a, b, j = plane_cost(P, i, xa * eye[f], eye[h], eye[v], ext, eta)
            lj = np.log10(j)
            lev = np.linspace(lj.min(), np.quantile(lj, 0.95), 14)
            cs = ax.contourf(a, b, lj, levels=lev, cmap="Blues_r", extend="max", alpha=0.85)
            ax.contour(a, b, lj, levels=lev, colors="white", linewidths=0.3)
            ax.plot(path[:, h], path[:, v], color=COL[k[1]], lw=1.0, label="L-BFGS path")
            ax.plot(xa[h], xa[v], "o", ms=4.5, mfc=COL[k[1]], mec="white", mew=0.7, label="analysis")
            ax.plot(xb[h], xb[v], "s", ms=5.5, mfc="#8a8984", mec="white", mew=0.7, label="first guess $x_b$")
            ax.plot(xt[h], xt[v], "*", ms=9, mfc=INK, mec="white", mew=0.6, label="truth $x_{true}(0)$")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(False)
            ax.set_xlabel(VARS[h], labelpad=1)
            ax.set_ylabel(VARS[v], labelpad=1)
            ax.set_title(f"{VARS[h]}–{VARS[v]} ({VARS[f]} = {xa[f]:.2f})", loc="left", color=INK, fontsize=7)
            cb = fig.colorbar(cs, ax=ax, shrink=0.85, pad=0.02, format="%.1f")
            cb.ax.tick_params(labelsize=5.5)
            if col == 2:
                cb.set_label(r"$\log_{10} J$", fontsize=6)
        axes[r, 0].text(-0.42, 0.5, NAMES[k[1]] + (" (ref.)" if k[1] == STAGES[tag][2] else ""),
                        transform=axes[r, 0].transAxes, rotation=90, ha="center", va="center", fontsize=8,
                        color=INK, fontweight="bold")
    fig.suptitle(f"truth {tag}, {c} constraint, {p.replace('_', '-')} obs, case {i} (median first-guess error) — J "
                 "on the coordinate planes through each model's L-BFGS analysis"
                 + (" (eta fixed at the analysed eta)" if c == "weak" else ""), x=0.01, ha="left", y=1.01,
                 color=MUTED, fontsize=7)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=len(lab), bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0.02, 0, 1, 0.975))
    save(fig, f"m8r_geom_{tag}_{c}")
    plt.close(fig)


def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else "test"
    plt = style()
    (ROOT / "tmp/figs").mkdir(parents=True, exist_ok=True)
    runs = collect(sub)
    for tag in STAGES:
        for p in PATTERNS:
            fig_land(runs, plt, tag, p)
        for c in ("strong", "weak"):
            fig_geom(runs, plt, tag, c)
    print(f"{len(runs)} L-BFGS runs -> tmp/figs/m8r_*")


if __name__ == "__main__":
    main()
