"""
M8 report (branch m8-classic-4dv): classical 4D-Var on the model rectangle F1, F2, E1, E2 (L1 = truth, metrics only).

    python src/report_classic_4dv.py select     val lr sweep tmp/results/m8/sweep/<model>/*.nc -> selected_lr.json,
                                                lr_sweep.md (pre-registered rule of src/select_lr.py, per model);
                                                lists the groups whose pick sits on a grid edge (-> extend the grid)
    python src/report_classic_4dv.py report     test runs tmp/results/m8/test/<model>/*.nc -> classic_4dv.{md,csv},
                                                figures tmp/figs/m8_*.{png,pdf}

Metrics (mean ± std over the test cases; per-iterate curves: median and interquartile range over active cases):
    sol_deg     |x0_a - x_true(0)|_2
    traj_deg    1/K sum_k |x_a^k - x_true^k|_2, k = 0 .. 560 (0.6-TU window + 5-TU forecast, estimation model)
    cos_truth   cos(x_true(0) - x0^k, -grad_x0 J(c^k))       (k = 0: first guess; _a: at the analysis)
    lam_min, lam_max, kappa_H = lam_max / lam_min (positive-definite cases only), n_neg = # negative eigenvalues of the
    Hessian of J w.r.t. the control (strong d = 3, weak d = 21), at the analysis.
"""

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fig_models import COL, INK, MUTED, NAMES, save, style  # noqa: E402
from select_lr import select  # noqa: E402

RES = ROOT / "tmp/results/m8"
MODELS = ("f1", "f2", "e1", "e2")
CONSTRAINTS = ("strong", "weak")
PATTERNS = ("full", "x_only")
METHODS = ("gd", "adagrad", "adam", "lbfgs_wolfe")
MNAME = {"gd": "GD", "adagrad": "Adagrad", "adam": "Adam", "lbfgs_wolfe": "L-BFGS"}


def load(path: Path):
    ds = xr.open_dataset(path)
    return ds, json.loads(ds.attrs["config"])


# ---------------------------------------------------------------------------------------------------------------- select
def cmd_select():
    rows = []
    for p in sorted((RES / "sweep").glob("*/*.nc")):
        ds, cfg = load(p)
        j = ds.J_iter.values[:, -1]
        rows.append({"model": cfg["model"], "constraint": cfg["constraint"], "pattern": cfg["pattern"],
                     "method": cfg["method"], "lr": float(cfg["lr"]), "diverged_frac": float(ds.diverged.values.mean()),
                     "conv_frac": float(ds.converged.values.mean()),
                     "J_med": float(np.nanmedian(np.where(np.isfinite(j), j, np.nan))),
                     "evals_med": float(np.median(ds.n_grad_evals.values)),
                     "sol_deg_med": float(np.nanmedian(ds.sol_deg.values))})
    groups = {}
    for r in rows:
        groups.setdefault(f"{r['model']}/{r['constraint']}/{r['pattern']}/{r['method']}", []).append(r)
    chosen, edges, md = {}, [], ["# M8 lr sweep (val, 200 cases)", ""]
    for key in sorted(groups):
        g = sorted(groups[key], key=lambda r: r["lr"])
        best = select(g)
        chosen[key] = best["lr"] if best else None
        if best and best["lr"] in (g[0]["lr"], g[-1]["lr"]):
            edges.append(f"{key}: lr {best['lr']:g} at the {'lower' if best['lr'] == g[0]['lr'] else 'upper'} edge")
        md += [f"### {key} -> lr = {best['lr'] if best else 'NONE'}", "",
               "| lr | diverged | converged | median J | median evals | median sol_deg |", "|---|---|---|---|---|---|"]
        for r in g:
            mark = " **←**" if best and r["lr"] == best["lr"] else ""
            md.append(f"| {r['lr']:g}{mark} | {r['diverged_frac']:.0%} | {r['conv_frac']:.0%} | {r['J_med']:.3f} | "
                      f"{r['evals_med']:.0f} | {r['sol_deg_med']:.3f} |")
        md.append("")
    md += ["## Grid edges", ""] + ([f"- {e}" for e in edges] or ["- none"])
    (RES / "selected_lr.json").write_text(json.dumps(chosen, indent=2))
    (RES / "lr_sweep.md").write_text("\n".join(md) + "\n")
    print("\n".join(edges) if edges else "no grid edge")
    print(f"{len(chosen)} groups -> {RES / 'selected_lr.json'}")


# ---------------------------------------------------------------------------------------------------------------- report
def ms(a, fmt="{:.3f}"):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return "–" if a.size == 0 else f"{fmt.format(a.mean())} ± {fmt.format(a.std())}"


def collect():
    runs = {}
    for p in sorted((RES / "test").glob("*/*.nc")):
        ds, cfg = load(p)
        runs[(cfg["model"], cfg["constraint"], cfg["pattern"], cfg["method"])] = (ds, cfg)
    return runs


def table(runs):
    head = ["model", "scheme", "obs", "optimizer", "lr", "n", "diverged", "grad evals", "sol_deg", "traj_deg",
            "cos_truth k=0", "cos_truth a", "lam_min a", "lam_max a", "kappa_H a (PD)", "non-PD frac", "n_neg a"]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    csv = [",".join(head)]
    for m in MODELS:
        for c in CONSTRAINTS:
            for p in PATTERNS:
                for meth in METHODS:
                    if (m, c, p, meth) not in runs:
                        continue
                    ds, cfg = runs[(m, c, p, meth)]
                    pd = ds.lam_min_a.values > 0
                    cells = [NAMES[m], c, p, MNAME[meth], f"{cfg['lr']:g}", str(ds.sizes["ic"]),
                             str(int(ds.diverged.sum())), ms(ds.n_grad_evals.values, "{:.0f}"),
                             ms(ds.sol_deg.values), ms(ds.traj_deg.values), ms(ds.cos_truth.values[:, 0]),
                             ms(ds.cos_truth_a.values), ms(ds.lam_min_a.values, "{:.3g}"),
                             ms(ds.lam_max_a.values, "{:.3g}"), ms(ds.kappa_H_a.values, "{:.3g}"),
                             f"{1 - pd.mean():.2f}", ms(ds.n_neg_a.values, "{:.2f}")]
                    md.append("| " + " | ".join(cells) + " |")
                    csv.append(",".join(x.replace(" ± ", ";") for x in cells))
    return md, csv


def curve(ax, k, a, color, label):
    """Median and IQR over cases per iterate, drawn while >= 10 % of the cases are active."""
    n_act = np.isfinite(a).sum(0)
    keep = n_act >= max(1, 0.1 * a.shape[0])
    if not keep.any():
        return
    with np.errstate(all="ignore"):
        q = np.nanpercentile(a[:, keep], [25, 50, 75], axis=0)
    kk = k[keep] + 1                                                 # log x axis: iterate k plotted at k + 1
    ax.fill_between(kk, q[0], q[2], color=color, alpha=0.18, lw=0)
    ax.plot(kk, q[1], color=color, lw=1.2, label=label)


def figures(runs, plt):
    quantities = [("cos_truth", r"cos$\theta$ (truth correction, $-\nabla_{x_0}J$)", "linear"),
                  ("lam_min", r"$\lambda_{min}(H)$", "symlog"), ("lam_max", r"$\lambda_{max}(H)$", "log"),
                  ("kappa_H", r"$\kappa_H$ (PD cases)", "log"), ("n_neg", "# negative eigenvalues", "linear")]
    for meth in METHODS:
        for var, ylab, scale in quantities:
            fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex=True, sharey=var in ("cos_truth", "n_neg"))
            for i, c in enumerate(CONSTRAINTS):
                for j, p in enumerate(PATTERNS):
                    ax = axes[i, j]
                    for m in MODELS:
                        if (m, c, p, meth) not in runs:
                            continue
                        ds = runs[(m, c, p, meth)][0]
                        a = ds[var].values
                        if var == "kappa_H":
                            a = np.where(ds.lam_min.values > 0, a, np.nan)
                        curve(ax, ds.k.values, a, COL[m], NAMES[m])
                    ax.set_xscale("log")
                    ax.set_yscale(scale) if scale != "symlog" else ax.set_yscale("symlog", linthresh=1e-2)
                    ax.set_title(f"{c} · {p.replace('_', '-')} obs", loc="left", color=INK)
                    if i == 1:
                        ax.set_xlabel("iteration k + 1")
                    if j == 0:
                        ax.set_ylabel(ylab)
            h, lab = axes[0, 0].get_legend_handles_labels()
            fig.legend(h, lab, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03))
            fig.suptitle(f"{MNAME[meth]}: median and IQR over test cases", x=0.01, ha="left", y=1.03, color=MUTED,
                         fontsize=7)
            fig.tight_layout()
            save(fig, f"m8_{var}_{meth}")
            plt.close(fig)
    # eigenvalue distribution at the analysis (all eigenvalues of all cases), per optimizer
    for meth in METHODS:
        fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharey=True)
        for i, c in enumerate(CONSTRAINTS):
            for j, p in enumerate(PATTERNS):
                ax = axes[i, j]
                data, labels, cols = [], [], []
                for m in MODELS:
                    if (m, c, p, meth) in runs:
                        ev = runs[(m, c, p, meth)][0].eig_a.values.ravel()
                        ev = ev[np.isfinite(ev)]
                        data.append(np.sign(ev) * np.log10(1 + np.abs(ev)))
                        labels.append(NAMES[m])
                        cols.append(COL[m])
                if not data:
                    continue
                parts = ax.violinplot(data, showmedians=True, widths=0.8)
                for body, col in zip(parts["bodies"], cols):
                    body.set_facecolor(col)
                    body.set_alpha(0.5)
                for key in ("cmedians", "cbars", "cmins", "cmaxes"):
                    parts[key].set_color(MUTED)
                    parts[key].set_linewidth(0.8)
                ax.axhline(0, color=MUTED, lw=0.6, ls=":")
                ax.set_xticks(range(1, len(labels) + 1), labels)
                ax.set_title(f"{c} · {p.replace('_', '-')} obs", loc="left", color=INK)
                if j == 0:
                    ax.set_ylabel(r"sign($\lambda$) log$_{10}$(1 + |$\lambda$|)")
        fig.suptitle(f"{MNAME[meth]}: Hessian eigenvalues at the analysis", x=0.01, ha="left", color=MUTED, fontsize=7)
        fig.tight_layout()
        save(fig, f"m8_eig_a_{meth}")
        plt.close(fig)


def cmd_report():
    runs = collect()
    md, csv = table(runs)
    hdr = ["# M8 classical 4D-Var on the model rectangle (test, mean ± std over cases)", "",
           "Window 0.6 TU (60 steps, obs every 10), forecast 5 TU; L1 = truth (metrics only); "
           "`a` = at the analysis; κ_H over positive-definite cases only.", ""]
    (RES / "classic_4dv.md").write_text("\n".join(hdr + md) + "\n")
    (RES / "classic_4dv.csv").write_text("\n".join(csv) + "\n")
    (ROOT / "tmp/figs").mkdir(parents=True, exist_ok=True)
    figures(runs, style())
    print(f"{len(runs)} runs -> {RES / 'classic_4dv.md'}, tmp/figs/m8_*")


if __name__ == "__main__":
    {"select": cmd_select, "report": cmd_report}[sys.argv[1]]()
