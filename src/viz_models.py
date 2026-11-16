"""
Model behaviour check (pilot request 2026-11-16): truth L63 (rho 28) vs biased knowledge model F1 (L63 rho 27.5) vs
emulator F2 (rho275, MLP tendency + RK4), all from the SAME initial conditions (D1 v2 test split, x_true at t0).

Figures (tmp/figs/, PNG + PDF):
  models_timeseries   x, y, z(t) over 10 TU from one test IC for the three models + |model - truth| (log scale)
                      + median |model - truth| over all test ICs
  models_phase        x-z phase portraits (one IC, 20 TU) of the three models
  models_sensitivity  each model run from x0 and from x0 + delta (|delta| = 1e-3): x(t) of both members, and the
                      separation |dx(t)| (log) for one IC and median over all test ICs, with the exp(lambda_1 t) slope
Numbers (stdout + tmp/results/models_behaviour.json): divergence time from truth (first t with |model - truth| > 1,
median over ICs), separation growth rate fitted on the linear phase (log-median separation, t in [1, 8] TU).

Usage (project root, env oceanai):  python src/viz_models.py [--ic 0] [--split test]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import xarray as xr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dameta.dynamics import L63Model, L63Params  # noqa: E402
from dameta.dynamics.emulator import L63Emulator  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "tmp/figs"
COL = {"truth": "k", "F1 (L63, rho 27.5)": "tab:blue", "F2 (emulator rho275)": "tab:red"}


def models():
    emu = L63Emulator.load(ROOT / "tmp/emulators/rho275.pt")
    for p in emu.parameters():
        p.requires_grad_(False)
    return {"truth": L63Model(L63Params()).step,
            "F1 (L63, rho 27.5)": L63Model(L63Params(rho=27.5)).step,
            "F2 (emulator rho275)": emu.step}


@torch.no_grad()
def roll(step, x0, n):
    out, x = [x0], x0
    for _ in range(n):
        x = step(x)
        out.append(x)
    return torch.stack(out, 1).numpy()                       # [n_ic, n + 1, 3]


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ic", type=int, default=0)
    ap.add_argument("--split", default="test")
    ap.add_argument("--delta", type=float, default=1e-3)
    a = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    FIGS.mkdir(parents=True, exist_ok=True)
    ds = xr.load_dataset(ROOT / f"tmp/data/l63v2_{a.split}_seed42.nc")
    x0 = torch.tensor(ds.x_true.values[:, 0])                  # [n, 3]
    n_ic, dt, nT = x0.shape[0], 0.01, 2000                     # 20 TU
    t = np.arange(nT + 1) * dt
    M = models()
    traj = {k: roll(s, x0, nT) for k, s in M.items()}
    # perturbed twin: fixed random unit direction per case, |delta| = a.delta
    g = torch.Generator().manual_seed(0)
    d = torch.randn(n_ic, 3, generator=g)
    d = a.delta * d / d.norm(dim=1, keepdim=True)
    twin = {k: roll(s, x0 + d, nT) for k, s in M.items()}
    i = a.ic
    stats = {}

    # ---- 1. time series ------------------------------------------------------------------------------------------
    nshow = 1000
    fig, ax = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
    for v, name in enumerate("xyz"):
        for k, tr in traj.items():
            ax[v].plot(t[:nshow + 1], tr[i, :nshow + 1, v], color=COL[k], lw=1.2 if k == "truth" else 1.0,
                       ls="-" if k == "truth" else "--", label=k)
        ax[v].set_ylabel(name)
    ax[0].legend(ncol=3, fontsize=8, loc="upper right")
    ax[0].set_title(f"Same initial condition ({a.split} case {i}): truth vs biased model F1 vs emulator F2")
    for k in list(M)[1:]:
        e1 = np.linalg.norm(traj[k][i] - traj["truth"][i], axis=-1)
        em = np.median(np.linalg.norm(traj[k] - traj["truth"], axis=-1), 0)
        ax[3].semilogy(t[:nshow + 1], e1[:nshow + 1], color=COL[k], label=k)
        ax[4].semilogy(t[:nshow + 1], em[:nshow + 1], color=COL[k], label=k)
        tdiv = np.array([t[np.argmax(r > 1.0)] if (r > 1.0).any() else np.nan
                         for r in np.linalg.norm(traj[k] - traj["truth"], axis=-1)])
        stats[k] = {"divergence_time_TU_median": float(np.nanmedian(tdiv)),
                    "divergence_time_TU_q25_q75": [float(np.nanquantile(tdiv, q)) for q in (0.25, 0.75)],
                    "err_vs_truth_median_at_1.1TU": float(em[110])}
    for axx, lab in ((ax[3], f"|model - truth|\ncase {i}"), (ax[4], f"|model - truth|\nmedian {n_ic} ICs")):
        axx.axhline(1.0, color="gray", lw=0.6, ls=":")
        axx.axvspan(0, 1.1, color="orange", alpha=0.12, lw=0)
        axx.set_ylabel(lab)
        axx.set_ylim(1e-4, 50)
    ax[3].text(0.02, 2e-4, "assimilation window (1.1 TU)", fontsize=7, color="darkorange")
    ax[4].legend(fontsize=8)
    ax[4].set_xlabel("time [TU]")
    save(fig, "models_timeseries")

    # ---- 2. phase portraits --------------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for axx, (k, tr) in zip(ax, traj.items()):
        axx.plot(tr[i, :, 0], tr[i, :, 2], color=COL[k], lw=0.5)
        axx.plot(tr[i, 0, 0], tr[i, 0, 2], "o", color="orange", ms=5)
        axx.set_title(k, fontsize=10)
        axx.set_xlabel("x")
    ax[0].set_ylabel("z")
    fig.suptitle(f"x-z phase portrait, 20 TU from the same IC ({a.split} case {i}, dot = IC)", fontsize=10)
    save(fig, "models_phase")

    # ---- 3. sensitivity to the IC --------------------------------------------------------------------------------
    nshow = 1000
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(4, 2)
    for r, k in enumerate(M):
        axx = fig.add_subplot(gs[r, :])
        axx.plot(t[:nshow + 1], traj[k][i, :nshow + 1, 0], color=COL[k], lw=1.1, label=f"{k}: x0")
        axx.plot(t[:nshow + 1], twin[k][i, :nshow + 1, 0], color=COL[k], lw=1.0, ls=":", label=f"{k}: x0 + delta")
        if k != "truth":
            axx.plot(t[:nshow + 1], traj["truth"][i, :nshow + 1, 0], color="gray", lw=0.6, alpha=0.6,
                     label="truth: x0 (reference)")
        axx.set_ylabel("x")
        axx.legend(fontsize=7, ncol=3, loc="upper right")
        if r == 0:
            axx.set_title(f"Two initial conditions |delta| = {a.delta:g} apart ({a.split} case {i})")
    ax1, ax2 = fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1])
    fit = (t >= 1.0) & (t <= 8.0)
    for k in M:
        s1 = np.linalg.norm(twin[k][i] - traj[k][i], axis=-1)
        sm = np.median(np.linalg.norm(twin[k] - traj[k], axis=-1), 0)
        ax1.semilogy(t[:nshow + 1], s1[:nshow + 1], color=COL[k], label=k)
        ax2.semilogy(t, sm, color=COL[k], label=k)
        lam = np.polyfit(t[fit], np.log(sm[fit]), 1)[0]
        stats.setdefault(k, {})["separation_growth_rate_fit_1_8TU"] = float(lam)
        sat = np.argmax(sm > 1.0)
        stats[k]["predictability_time_TU_sep_gt_1"] = float(t[sat]) if sat > 0 else None
    tt = t[t <= 12]
    ax2.semilogy(tt, a.delta * np.exp(0.905 * tt) * 0.5, color="gray", ls="--", lw=0.8, label="slope exp(0.905 t)")
    ax1.set_title(f"|separation|, case {i}", fontsize=9)
    ax2.set_title(f"median |separation| over {n_ic} ICs", fontsize=9)
    for axx in (ax1, ax2):
        axx.set_xlabel("time [TU]")
        axx.axhline(1.0, color="gray", lw=0.6, ls=":")
        axx.axvspan(0, 1.1, color="orange", alpha=0.12, lw=0)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "models_sensitivity")

    (ROOT / "tmp/results/models_behaviour.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
