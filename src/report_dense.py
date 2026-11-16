"""
Report of the dense-observation test (src/run_dense_v2.py): sparse (12 obs, D1 v2) vs dense (111 obs, every step)
over the same 1.1-TU window; estimation model F1 (L63 rho 27.5) and the true-model reference E1.

--constraint strong: configurations sparse / dense; optimizers Adagrad, GD, L-BFGS, Gauss-Newton.
--constraint weak:   configurations sparse (sigma_q 0.1) / dense (sigma_q 0.1) / dense (sigma_q 0.1 / sqrt 10);
                     optimizers Adagrad, GD, L-BFGS (no GN). The analysis trajectory carries the eta increments; the
                     forecast starts from the end-of-window analysis state. Extra column: RMS of eta per model step.

Metrics per (configuration, model, optimizer), test split, medians over the 200 cases:
    J* = best final cost over the optimizers (same configuration, model, case); target J <= 1.01 J*
    cost to target in gradient-equivalents (1 gradient = 1 forward + 1 adjoint run; GN unit converted by timing)
    and wall-clock (measured gradient time x gradient-equivalents, 200 cases batched, one core)
    reached = fraction of cases reaching the target; worse basins = final J > 1.1 J*
    IC RMSE vs truth; forecast RMSE at 1 TU after the window with the TRUE model (IC quality) and with the
    estimation model (operational)
Figures (tmp/figs/, suffix _weak for weak): dense_framework_case<i> (one case: truth, obs, first guess, analyses,
forecasts; + its optimization curves), dense_curves (median J/J* - 1 and IC RMSE vs gradient-equivalents),
dense_forecast (forecast RMSE vs lead). Table: tmp/results/dense/summary_<split>[_weak].{md,csv}

Usage (project root, env oceanai):  python src/report_dense.py [--split test] [--case 0] [--constraint weak]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import xarray as xr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dameta.da import forecast_from_analysis  # noqa: E402
from dameta.da.cost import CostConfig, window_trajectory  # noqa: E402
from dameta.dynamics import L63Model, L63Params  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES, FIGS = ROOT / "tmp/results/dense", ROOT / "tmp/figs"
Q2 = 0.0316227766
# configuration key -> (dataset file prefix, record-tag suffix, label)
CONFIGS = {
    "strong": {"sparse": ("l63v2", "", "sparse (12 obs)"), "dense": ("l63v2d", "", "dense (111 obs)")},
    "weak": {"sparse": ("l63v2", "_weak_q0.1", "sparse (12 obs), sq 0.1"),
             "dense": ("l63v2d", "_weak_q0.1", "dense (111 obs), sq 0.1"),
             "dense_qs": ("l63v2d", f"_weak_q{Q2:g}", "dense (111 obs), sq 0.1/sqrt10")},
}
LSTYLE = {"sparse": "--", "dense": "-", "dense_qs": ":"}
MODELS = {"f1": "F1 (rho 27.5)", "truth": "E1 true model"}
ALL_METHODS = {"adagrad": "Adagrad", "gd": "GD", "lbfgs": "L-BFGS", "gn": "Gauss-Newton"}
MCOL = {"adagrad": "tab:blue", "gd": "tab:gray", "lbfgs": "tab:green", "gn": "tab:purple"}
STEP = {"f1": L63Model(L63Params(rho=27.5)).step, "truth": L63Model(L63Params()).step}
F, LEAD1 = 500, 100


def rmse(a, b):
    return np.sqrt(((a - b) ** 2).mean(-1))


def ccfg_of(ds):
    return CostConfig(window=int(ds.attrs["window"]), obs_every=int(ds.attrs["obs_every"]))


def split_control(ctrl, ds, weak):
    """Packed control [n, d] -> x0 [n, 3], eta [n, n_obs - 1, 3] or None."""
    ctrl = torch.tensor(ctrl)
    if not weak:
        return ctrl[:, :3], None
    return ctrl[:, :3], ctrl[:, 3:].reshape(len(ctrl), ds.sizes["t_obs"] - 1, 3)


def control(r):
    """Final packed control of a record (strong records written before 2026-11-16 weak runs hold only x0_a)."""
    return r.control_a.values if "control_a" in r else r.x0_a.values


def load(split, cons):
    data, rec, timing = {}, {}, {}
    for c, (d, suf, _) in CONFIGS[cons].items():
        data[c] = xr.load_dataset(ROOT / f"tmp/data/{d}_{split}_seed42.nc")
        for m in MODELS:
            timing[(c, m)] = json.loads((RES / f"timing_{d}_{m}{suf}.json").read_text())
            for o in METHODS:
                rec[(c, m, o)] = xr.load_dataset(RES / f"{d}_{m}{suf}_{o}_{split}.nc")
    return data, rec, timing


@torch.no_grad()
def forecasts(x0, eta, model, ds):
    return {k: forecast_from_analysis(STEP[k], x0, ccfg_of(ds), F, eta).numpy() for k in ("truth", model)}


@torch.no_grad()
def analysis_traj(step, x0, eta, ds, n_total):
    """Window trajectory (with eta if weak), then free run up to n_total steps: [n_total + 1, 3] for one case."""
    w = window_trajectory(step, x0[None], ccfg_of(ds), None if eta is None else eta[None])[0]
    s, x = [w], w[-1]
    for _ in range(n_total - (len(w) - 1)):
        x = step(x)
        s.append(x[None])
    return torch.cat(s, 0).numpy()


def summarize(data, rec, timing, weak):
    rows, curves, fcs = [], {}, {}
    for c, ds in data.items():
        xt, L = ds.x_true.values, int(ds.attrs["window"])
        truth_fc = xt[:, L:L + F + 1]
        for m in MODELS:
            jfin = np.stack([rec[(c, m, o)].J_hist.values[:, -1] for o in METHODS])
            jstar = np.nanmin(np.where(np.isfinite(jfin), jfin, np.nan), 0)
            tg = timing[(c, m)]["gradient_s"]
            for o in METHODS:
                r = rec[(c, m, o)]
                unit = float(r.attrs["unit_grad_eq"])
                jh = r.J_hist.values
                hit = jh <= 1.01 * jstar[:, None]
                k = np.where(hit.any(1), hit.argmax(1), np.nan) * unit
                x0, eta = split_control(control(r), ds, weak)
                fcs[(c, m, o)] = {k_: rmse(v, truth_fc) for k_, v in forecasts(x0, eta, m, ds).items()}
                ic_curve = rmse(r.x_hist.values, xt[:, None, 0])
                curves[(c, m, o)] = {"x": np.arange(jh.shape[1]) * unit, "rel": np.nanmedian(jh / jstar[:, None] - 1, 0),
                                     "ic": np.nanmedian(ic_curve, 0)}
                row = {
                    "obs": CONFIGS["weak" if weak else "strong"][c][2], "model": MODELS[m], "optimizer": METHODS[o],
                    "lr": r.attrs.get("lr", "-"),
                    "grad-eq to 1 %": float(np.nanmedian(k)),
                    "IQR": f"{np.nanquantile(k, 0.25):.1f}-{np.nanquantile(k, 0.75):.1f}",
                    "reached": float(np.isfinite(k).mean()),
                    "time to target [s]": float(np.nanmedian(k)) * tg,
                    "worse basins /200": int((jfin[list(METHODS).index(o)] > 1.1 * jstar).sum()),
                    "IC RMSE": float(np.nanmedian(rmse(x0.numpy(), xt[:, 0]))),
                    "fc@1TU true model": float(np.nanmedian(fcs[(c, m, o)]["truth"][:, LEAD1])),
                    "fc@1TU est. model": float(np.nanmedian(fcs[(c, m, o)][m][:, LEAD1])),
                }
                if weak:   # RMS eta per model step (dense: one increment per step; sparse: per 10 steps -> / sqrt 10)
                    per = int(ds.attrs["obs_every"])
                    row["RMS eta per step"] = float(np.median(np.sqrt((eta.numpy() ** 2).mean((1, 2)) / per)))
                rows.append(row)
    return pd.DataFrame(rows), curves, fcs


def fig_framework(data, rec, split, i, weak, suf):
    """One case: truth, obs, first guess, F1 analysis (L-BFGS, converged) + forecasts, E1 reference forecast; per config."""
    nc = len(data)
    fig, ax = plt.subplots(4, nc, figsize=(7.5 * nc, 12), gridspec_kw={"height_ratios": [1, 1, 1, 1.1]}, squeeze=False)
    nT = 110 + 300
    t = np.arange(nT + 1) * 0.01
    for c_i, (c, ds) in enumerate(data.items()):
        L = int(ds.attrs["window"])
        xt = ds.x_true.values[i, :nT + 1]
        to, y = ds.t_obs.values, ds.y_obs.values[i]
        xb = torch.tensor(rec[(c, "f1", "adagrad")].x_hist.values[i, 0])
        x_f1, eta_f1 = split_control(control(rec[(c, "f1", "lbfgs")])[i:i + 1], ds, weak)
        x_e1, eta_e1 = split_control(control(rec[(c, "truth", "lbfgs")])[i:i + 1], ds, weak)
        e = (lambda v: None if v is None else v[0])
        tr_b = analysis_traj(STEP["f1"], xb, None, ds, nT)
        tr_a = analysis_traj(STEP["f1"], x_f1[0], e(eta_f1), ds, nT)        # operational: F1 analysis + F1 forecast
        # F1 analysis -> TRUE-model forecast: analysis trajectory (F1 + eta) over the window, true model afterwards
        tr_at = np.concatenate([tr_a[:L + 1], analysis_traj(STEP["truth"], torch.tensor(tr_a[L]), None, ds,
                                                            nT - L)[1:nT - L + 1]], 0) if weak else \
            analysis_traj(STEP["truth"], x_f1[0], None, ds, nT)
        tr_e1 = analysis_traj(STEP["truth"], x_e1[0], e(eta_e1), ds, nT)    # reference: E1 analysis + true model
        for v, name in enumerate("xyz"):
            a = ax[v, c_i]
            a.axvspan(0, L * 0.01, color="orange", alpha=0.12, lw=0)
            a.plot(t, xt[:, v], "k", lw=1.6, label="truth")
            a.plot(to * 0.01, y[:, v], "o", color="0.45", ms=4 if len(to) < 20 else 1.8, label=f"obs ({len(to)})")
            a.plot(t, tr_b[:, v], color="tab:orange", lw=1, ls=":", label="first guess -> F1")
            a.plot(t, tr_a[:, v], color="tab:blue", lw=1.2, label="F1 analysis -> F1 forecast (operational)")
            a.plot(t, tr_at[:, v], color="tab:cyan", lw=1, ls="--", label="F1 analysis -> true-model forecast")
            a.plot(t, tr_e1[:, v], color="tab:red", lw=1, ls="-.", label="reference: E1 analysis -> true model")
            a.set_ylabel(name)
            if v == 0:
                lab = CONFIGS["weak" if weak else "strong"][c][2]
                a.set_title(f"{lab}{' weak' if weak else ''} — {split} case {i} (shaded: window)", fontsize=10)
        ax[2, c_i].set_xlabel("time [TU]")
        a = ax[3, c_i]
        jstar = min(float(rec[(c, "f1", oo)].J_hist.values[i, -1]) for oo in METHODS)
        for o in METHODS:
            r = rec[(c, "f1", o)]
            xk = np.arange(r.sizes["k"]) * float(r.attrs["unit_grad_eq"])
            a.semilogy(xk, np.maximum(r.J_hist.values[i] / jstar - 1, 1e-8), color=MCOL[o], label=METHODS[o])
        a.axhline(0.01, color="k", lw=0.6, ls=":")
        a.set_xscale("symlog", linthresh=10)
        a.set_xlim(0, 1000)
        a.set_ylim(1e-6, 1e2)
        a.set_xlabel("gradient-equivalents (1 = forward + adjoint run)")
        a.set_ylabel("J / J* - 1  (F1, this case)")
        a.legend(fontsize=7, ncol=2)
    ax[0, 0].legend(fontsize=7, ncol=2, loc="lower left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"dense_framework_case{i}{suf}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_curves(curves, configs, suf):
    fig, ax = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for c_i, m in enumerate(MODELS):
        for c, (_, _, clab) in configs.items():
            for o in METHODS:
                cv = curves[(c, m, o)]
                lab = f"{METHODS[o]}, {clab}"
                ax[0, c_i].semilogy(cv["x"], np.maximum(cv["rel"], 1e-8), color=MCOL[o], ls=LSTYLE[c], label=lab)
                ax[1, c_i].plot(cv["x"], cv["ic"], color=MCOL[o], ls=LSTYLE[c], label=lab)
        ax[0, c_i].axhline(0.01, color="k", lw=0.6, ls=":")
        ax[0, c_i].set_title(f"estimation model: {MODELS[m]} (median over 200 test cases)", fontsize=10)
        ax[0, c_i].set_ylabel("median J / J* - 1")
        ax[0, c_i].set_ylim(1e-6, 1e2)
        ax[1, c_i].set_ylabel("median IC RMSE vs truth")
        ax[1, c_i].set_xlabel("gradient-equivalents (symlog)")
        ax[1, c_i].set_xscale("symlog", linthresh=10)
        ax[1, c_i].set_xlim(0, 1000)
    ax[0, 0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"dense_curves{suf}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_forecast(fcs, configs, suf):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    lead = np.arange(F + 1) * 0.01
    for c_i, (key, lab) in enumerate((("truth", "forecast with the TRUE model (IC quality)"),
                                      ("est", "forecast with the estimation model (operational)"))):
        for c, (_, _, clab) in configs.items():
            for m, col in (("f1", "tab:blue"), ("truth", "tab:red")):
                v = fcs[(c, m, "lbfgs")]["truth" if key == "truth" else m]
                ax[c_i].plot(lead, np.nanmedian(v, 0), color=col, ls=LSTYLE[c], label=f"{MODELS[m]}, {clab}")
        ax[c_i].set_title(lab, fontsize=10)
        ax[c_i].set_xlabel("lead time after the window [TU]")
    ax[0].set_ylabel("median forecast RMSE (L-BFGS analyses)")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"dense_forecast{suf}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)


METHODS = dict(ALL_METHODS)


def main():
    global METHODS
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--constraint", default="strong", choices=["strong", "weak"])
    a = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    weak = a.constraint == "weak"
    if weak:
        METHODS = {k: v for k, v in ALL_METHODS.items() if k != "gn"}
    suf = "_weak" if weak else ""
    data, rec, timing = load(a.split, a.constraint)
    df, curves, fcs = summarize(data, rec, timing, weak)
    df.to_csv(RES / f"summary_{a.split}{suf}.csv", index=False)
    fmt = df.copy()
    for col in ("grad-eq to 1 %", "time to target [s]", "IC RMSE", "fc@1TU true model", "fc@1TU est. model", "reached",
                "RMS eta per step"):
        if col in fmt:
            fmt[col] = fmt[col].map(lambda v: f"{v:.3g}" if col != "reached" else f"{v:.2f}")
    md = "\n".join(["| " + " | ".join(fmt.columns) + " |", "|" + "---|" * len(fmt.columns)]
                   + ["| " + " | ".join(str(v) for v in row) + " |" for row in fmt.itertuples(index=False)])
    (RES / f"summary_{a.split}{suf}.md").write_text(md + "\n")
    print(md)
    for k, v in timing.items():
        print(k, {kk: v[kk] for kk in ("gradient_s", "forward_s", "lr", "n_obs")})
    fig_framework(data, rec, a.split, a.case, weak, suf)
    fig_curves(curves, CONFIGS[a.constraint], suf)
    fig_forecast(fcs, CONFIGS[a.constraint], suf)


if __name__ == "__main__":
    main()
