"""
M8 weak- vs strong-constraint figures (branch m8-classic-4dv, pilot 2026-11-26): with the per-model diagonal Q
(src/estimate_q.py, cost.q=model), does weak 4D-Var solve a different problem than strong 4D-Var?

    python src/fig_weak_strong.py     reads tmp/results/m8/feasq/<model>/*.nc (10 val cases, 1.6-TU window, obs every
                                      20 steps, sigma_o2 = 4, B = 4 x hybrid B) -> tmp/figs/m8wq_*.{png,pdf}

Figures (one per obs pattern p unless stated)
    m8wq_traj_<p>        example case (median F1 first-guess error): x, y, z over the window of the truth, the
                         observations, the strong and the weak L-BFGS analysis trajectories (weak: jumps = eta);
                         rows = model, columns = variable
    m8wq_eta_<p>         weak L-BFGS: analysed eta_i,v vs the correction the model actually needs at that interval,
                         -e_i,v = x_true(t_i) - M(x_true(t_i - obs_every)); rows = model, columns = variable
    m8wq_icerr_a_xyz_<p> per-variable IC error at the analysis per case: first guess, strong, weak (marker = optimizer)
    m8wq_terms           cost terms at the L-BFGS analysis (Jb, Jo, Jq; median over cases), strong vs weak
    m8wq_conv_<p>        normalized cost (J^k - J_a) / (J^0 - J_a) vs iteration, strong (dashed) vs weak (solid),
                         one panel per model, colour = optimizer (J_a = best J of that scheme on the case)
    m8wq_spec            Hessian at the analysis: lambda_min, kappa_H (PD cases), share non-PD over iterates,
                         strong vs weak (all optimizers pooled)
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
from fig_classic_4dv import DASH, FG, LABEL, ORDER, SLOTS, TRUTH  # noqa: E402
from run_4dvar_v2 import setup_problem  # noqa: E402
from dameta.da.cost import window_trajectory  # noqa: E402
from dameta.dynamics.models import build_model, load_registry  # noqa: E402

torch.set_default_dtype(torch.float64)
RES = ROOT / "tmp/results/m8/feasq"
MODELS = ("f1", "f2", "e1", "e2")
PATTERNS = ("full", "x_only")
VARS = ("x", "y", "z")
DT = 0.01
SCOL = {"strong": "#2a78d6", "weak": "#eb6834"}          # scheme colours (validated slots 1, 2)
SLS = {"strong": "--", "weak": "-"}
MARK = {"gd": "o", "adagrad": "s", "adam": "^", "lbfgs_wolfe": "D"}


def collect():
    runs = {}
    for p in sorted(RES.glob("*/*.nc")):
        ds = xr.load_dataset(p)
        cfg = json.loads(ds.attrs["config"])
        runs[(cfg["model"], cfg["constraint"], cfg["pattern"], cfg["method"])] = (ds, cfg)
    return runs


_PROB = {}


def problem(cfg):
    key = (cfg["model"], cfg["constraint"], cfg["pattern"])
    if key not in _PROB:
        _PROB[key] = setup_problem(OmegaConf.create(cfg), cfg["split"], cfg.get("n_cases"))
    return _PROB[key]


def legend_top(fig, axes, y=0.975, ncol=None):
    h, lab = [], []
    for ax in np.ravel(axes):
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in lab:
                h.append(hh)
                lab.append(ll)
    fig.legend(h, lab, loc="upper center", ncol=ncol or len(lab), bbox_to_anchor=(0.5, y), handlelength=2.6)


def control(ds, weak):
    x0 = torch.tensor(np.nan_to_num(ds.x0_a.values))
    eta = torch.tensor(np.nan_to_num(ds.eta_final.values)) if weak else None
    return x0, eta


# ----------------------------------------------------------------------------------------------------------- trajectory
@torch.no_grad()
def fig_traj(runs, plt, p, case):
    fig, axes = plt.subplots(len(MODELS), 3, figsize=(7.2, 1.5 * len(MODELS) + 0.6), sharex=True)
    for mi, m in enumerate(MODELS):
        for c in ("strong", "weak"):
            key = (m, c, p, "lbfgs_wolfe")
            if key not in runs:
                continue
            ds, cfg = runs[key]
            P = problem(cfg)
            x0, eta = control(ds, c == "weak")
            tr = window_trajectory(P["cost"].step, x0[case:case + 1], P["ccfg"],
                                   eta[case:case + 1] if eta is not None else None)[0].numpy()
            t = np.arange(tr.shape[0]) * DT
            for v in range(3):
                axes[mi, v].plot(t, tr[:, v], color=SCOL[c], ls=SLS[c], lw=1.1, label=f"{c} analysis")
        P = problem(runs[(m, "strong", p, "lbfgs_wolfe")][1])
        L, every = P["ccfg"].window, P["ccfg"].obs_every
        xt = P["ds"].x_true.values[case, : L + 1]
        y, mask = P["y"][case].numpy(), P["mask"].numpy()
        t_obs = np.arange(0, L + 1, every) * DT
        for v in range(3):
            ax = axes[mi, v]
            ax.plot(np.arange(L + 1) * DT, xt[:, v], color=TRUTH, lw=1.0, label="truth")
            ob = mask[:, v] > 0
            if ob.any():
                ax.plot(t_obs[ob], y[ob, v], "o", ms=3.2, mfc="white", mec=INK, mew=0.8, label="observation")
            ax.plot(0, P["x0_b"][case, v].item(), "s", ms=4.5, mfc=FG, mec="white", mew=0.6, label="first guess")
            if mi == 0:
                ax.set_title(VARS[v], loc="left", color=INK)
            if v == 0:
                ax.set_ylabel(NAMES[m])
            if mi == len(MODELS) - 1:
                ax.set_xlabel("time [TU]")
    fig.suptitle(f"{p.replace('_', '-')} obs, case {case} — window trajectories of the L-BFGS analyses "
                 "(weak: jumps at the obs times = analysed η)", x=0.01, ha="left", y=1.0, color=MUTED, fontsize=7)
    legend_top(fig, axes)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, f"m8wq_traj_{p}")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------------------ eta
@torch.no_grad()
def fig_eta(runs, plt, p):
    reg = load_registry()
    fig, axes = plt.subplots(len(MODELS), 3, figsize=(7.0, 1.75 * len(MODELS) + 0.5))
    for mi, m in enumerate(MODELS):
        key = (m, "weak", p, "lbfgs_wolfe")
        if key not in runs:
            continue
        ds, cfg = runs[key]
        P = problem(cfg)
        L, every = P["ccfg"].window, P["ccfg"].obs_every
        step = build_model(m, reg, str(ROOT / "tmp/emulators"))[0]
        xt = torch.tensor(P["ds"].x_true.values[:, : L + 1])
        need = []
        for t0 in range(0, L, every):
            x = xt[:, t0]
            for _ in range(every):
                x = step(x)
            need.append((xt[:, t0 + every] - x).numpy())
        need = np.stack(need, 1)                                       # [n, n_obs - 1, 3]
        eta = ds.eta_final.values
        q = np.sqrt(np.asarray(P["ccfg"].q_var)) if P["ccfg"].q_var else np.full(3, P["ccfg"].sigma_q)
        for v in range(3):
            ax = axes[mi, v]
            a, b = need[..., v].ravel(), eta[..., v].ravel()
            lim = np.nanmax(np.abs(np.concatenate([a, b]))) * 1.05
            ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.6, ls=(0, (2, 2)), label="η = needed")
            ax.axhspan(-q[v], q[v], color="#e6e5e0", lw=0, zorder=0, label=r"$\pm\sigma_q$ (Q diag)")
            ax.plot(a, b, "o", ms=2.6, mfc="none", mec=SCOL["weak"], mew=0.6, label="interval × case")
            r = np.corrcoef(a, b)[0, 1] if np.nanstd(b) > 0 else np.nan
            ax.text(0.03, 0.95, f"r = {r:.2f}", transform=ax.transAxes, va="top", fontsize=6, color=INK)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal", adjustable="box")
            if mi == 0:
                ax.set_title(VARS[v], loc="left", color=INK)
            if v == 0:
                ax.set_ylabel(f"{NAMES[m]}\nanalysed η")
            if mi == len(MODELS) - 1:
                ax.set_xlabel(r"needed correction $x_{true}(t_i) - M(x_{true}(t_{i-1}))$")
    fig.suptitle(f"Weak L-BFGS, {p.replace('_', '-')} obs — analysed model-error increments vs the model error at "
                 "each interval (10 val cases × 8 intervals)", x=0.01, ha="left", y=1.0, color=MUTED, fontsize=7)
    legend_top(fig, axes)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, f"m8wq_eta_{p}")
    plt.close(fig)


# -------------------------------------------------------------------------------------------------------- IC error
def fig_icerr_a(runs, plt, p):
    fig, axes = plt.subplots(len(MODELS), 3, figsize=(7.0, 1.55 * len(MODELS) + 0.5), sharey=True)
    offs = {me: (j - 1.5) * 0.1 for j, me in enumerate(ORDER)}
    for mi, m in enumerate(MODELS):
        for ci, c in enumerate(("strong", "weak")):
            for me in ORDER:
                key = (m, c, p, me)
                if key not in runs:
                    continue
                ds, cfg = runs[key]
                P = problem(cfg)
                xt = P["ds"].x_true.values[:, 0]
                ea = np.abs(ds.x0_a.values - xt)
                eb = np.abs(P["x0_b"].numpy() - xt)
                for v in range(3):
                    ax = axes[mi, v]
                    ax.plot(np.full(len(ea), ci + 1 + offs[me]), ea[:, v], MARK[me], ms=2.8, mfc="none",
                            mec=SCOL[c], mew=0.7, label=LABEL[me] if c == "weak" else None)
                    ax.plot(ci + 1 + offs[me], np.median(ea[:, v]), "_", ms=6, mec=SCOL[c], mew=1.4)
                    if ci == 0 and me == "lbfgs_wolfe":
                        ax.plot(np.zeros(len(eb)), eb[:, v], "o", ms=2.8, mfc="none", mec=FG, mew=0.7)
                        ax.plot(0, np.median(eb[:, v]), "_", ms=9, mec=FG, mew=1.4)
        for v in range(3):
            ax = axes[mi, v]
            ax.set_yscale("log")
            ax.set_xticks([0, 1, 2], ["first\nguess", "strong", "weak"], fontsize=6)
            if mi == 0:
                ax.set_title(VARS[v], loc="left", color=INK)
            if v == 0:
                ax.set_ylabel(f"{NAMES[m]}\n" + r"$|x_{0,a} - x_{true}(0)|$")
    fig.suptitle(f"{p.replace('_', '-')} obs — per-variable IC error at the analysis per case, strong vs weak "
                 "(— median; markers = optimizer)", x=0.01, ha="left", y=1.0, color=MUTED, fontsize=7)
    legend_top(fig, axes)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, f"m8wq_icerr_a_xyz_{p}")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------- cost terms
@torch.no_grad()
def fig_terms(runs, plt):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    w = 0.38
    for a_i, p in enumerate(PATTERNS):
        ax = axes[a_i]
        for ci, c in enumerate(("strong", "weak")):
            meds = {"Jb": [], "Jo": [], "Jq": []}
            for m in MODELS:
                ds, cfg = runs[(m, c, p, "lbfgs_wolfe")]
                P = problem(cfg)
                tm = P["cost"].terms(*control(ds, c == "weak"))
                for k in meds:
                    meds[k].append(float(np.median(tm[k].numpy())))
            x = np.arange(len(MODELS)) + (ci - 0.5) * w
            bottom = np.zeros(len(MODELS))
            for k, shade in (("Jb", 0.35), ("Jo", 0.7), ("Jq", 1.0)):
                ax.bar(x, meds[k], w * 0.92, bottom=bottom, color=SCOL[c], alpha=shade,
                       label=f"{c}: {k}", edgecolor="white", lw=0.4)
                bottom += np.array(meds[k])
        ax.set_xticks(np.arange(len(MODELS)), [NAMES[m] for m in MODELS])
        ax.set_title(f"{p.replace('_', '-')} obs", loc="left", color=INK)
        if a_i == 0:
            ax.set_ylabel("cost term at the analysis (median)")
    fig.suptitle("L-BFGS analysis: J_b, J_o, J_q (stacked; left bar strong, right bar weak)", x=0.01, ha="left",
                 y=1.0, color=MUTED, fontsize=7)
    legend_top(fig, axes, y=0.95, ncol=6)
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    save(fig, "m8wq_terms")
    plt.close(fig)


# -------------------------------------------------------------------------------------------------------- convergence
def fig_conv(runs, plt, p):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex=True, sharey=True)
    for ax, m in zip(axes.ravel(), MODELS):
        for c in ("strong", "weak"):
            keys = [k for k in runs if k[:3] == (m, c, p)]
            if not keys:
                continue
            j_star = np.nanmin(np.stack([np.nanmin(runs[k][0].J_iter.values, 1) for k in keys]), 0)
            for k in sorted(keys, key=lambda k: ORDER.index(k[3])):
                ds = runs[k][0]
                ks = int(min(ds.sizes["k"] - 1, np.nanmax(ds.n_grad_evals.values)))
                j = ds.J_iter.values[:, : ks + 1]
                jn = (j - j_star[:, None]) / (j[:, :1] - j_star[:, None])
                with np.errstate(all="ignore"):
                    med = np.nanmedian(np.clip(jn, 1e-8, None), 0)
                me = k[3]
                ax.plot(np.arange(ks + 1) + 1, med, color=SLOTS[ORDER.index(me)], ls=SLS[c], lw=1.1,
                        label=f"{LABEL[me]} {c}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(NAMES[m], loc="left", color=INK)
    for ax in axes[1]:
        ax.set_xlabel("iteration k + 1")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$(J^k - J^*)/(J^0 - J^*)$ (median)")
    fig.suptitle(f"{p.replace('_', '-')} obs — normalized cost vs iteration, strong (dashed) vs weak (solid); "
                 "J* = best J of that scheme on the case", x=0.01, ha="left", y=1.0, color=MUTED, fontsize=7)
    legend_top(fig, axes, ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, f"m8wq_conv_{p}")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------- spectrum
def fig_spec(runs, plt):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2))
    w = 0.38
    for r, p in enumerate(PATTERNS):
        for ci, c in enumerate(("strong", "weak")):
            lmin, kap, nonpd = [], [], []
            for m in MODELS:
                a_l, a_k, it = [], [], []
                for me in ORDER:
                    if (m, c, p, me) in runs:
                        ds = runs[(m, c, p, me)][0]
                        a_l.append(ds.lam_min_a.values)
                        a_k.append(ds.kappa_H_a.values)
                        v = ds.lam_min.values
                        it.append(v[np.isfinite(v)])
                lmin.append(np.concatenate(a_l))
                kap.append(np.concatenate(a_k))
                it = np.concatenate(it)
                nonpd.append((it <= 0).mean())
            x = np.arange(len(MODELS)) + (ci - 0.5) * w
            for j in range(len(MODELS)):
                axes[r, 0].plot(np.full(len(lmin[j]), x[j]), lmin[j], "o", ms=2.4, mfc="none", mec=SCOL[c],
                                mew=0.6, label=c if j == 0 else None)
                kk = kap[j][np.isfinite(kap[j])]
                axes[r, 1].plot(np.full(len(kk), x[j]), kk, "o", ms=2.4, mfc="none", mec=SCOL[c], mew=0.6)
            axes[r, 2].bar(x, nonpd, w * 0.92, color=SCOL[c])
        axes[r, 0].set_yscale("symlog", linthresh=0.1)
        axes[r, 0].axhline(0, color=MUTED, lw=0.6)
        axes[r, 1].set_yscale("log")
        for a_i, ttl in enumerate((r"$\lambda_{min}$ at the analysis", r"$\kappa_H$ at the analysis (PD)",
                                   "share non-PD over iterates")):
            axes[r, a_i].set_xticks(np.arange(len(MODELS)), [NAMES[m] for m in MODELS])
            axes[r, a_i].set_title(f"{p.replace('_', '-')} obs · {ttl}", loc="left", color=INK, fontsize=6.5)
    fig.suptitle("Hessian of J w.r.t. the control (strong d = 3, weak d = 27), 4 optimizers pooled, 10 val cases",
                 x=0.01, ha="left", y=1.0, color=MUTED, fontsize=7)
    legend_top(fig, axes, y=0.96)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "m8wq_spec")
    plt.close(fig)


def main():
    plt = style()
    runs = collect()
    (ROOT / "tmp/figs").mkdir(parents=True, exist_ok=True)
    fig_terms(runs, plt)
    fig_spec(runs, plt)
    for p in PATTERNS:
        P = problem(runs[("f1", "strong", p, "lbfgs_wolfe")][1])
        fge = np.linalg.norm(P["x0_b"].numpy() - P["ds"].x_true.values[:, 0], axis=1)
        case = int(np.argmin(np.abs(fge - np.median(fge))))
        fig_traj(runs, plt, p, case)
        fig_eta(runs, plt, p)
        fig_icerr_a(runs, plt, p)
        fig_conv(runs, plt, p)
    print(f"{len(runs)} runs -> tmp/figs/m8wq_*")


if __name__ == "__main__":
    main()
