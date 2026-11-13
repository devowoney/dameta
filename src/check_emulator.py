"""
Emulator quality checks (pilot 2026-11-13). Three checks on held-out D1 TEST ICs, each comparing the emulator with
the true model M1 and with its own (biased) teacher, so learning error and the intended model error are separated.

Usage (project root):  python src/check_emulator.py rho275 [sig105 clean xz095 ...]
Writes tmp/results/emulator/<name>_qc.json, tmp/results/emulator/emulator_qc.md (all names given),
figures tmp/figs/m3_emuqc_<name>_{1_same_ic,2_pair,3_ensemble}.{png,pdf}.

1. Same initial condition: M1, teacher and emulator rolled out 5 TU from the TRUE x0 of 200 test ICs.
   RMSE vs lead (median, IQR), valid prediction time (VPT: error > 0.4 x climatological std).
2. Slightly different initial conditions: pairs (x0, x0 + delta), |delta| = 1e-3 in a random direction, propagated
   by the SAME model (M1 pair, emulator pair). Separation |x - x'| vs time: growth rate over 1-6 TU (~ lambda_1)
   and separation level at 8-10 TU — does the emulator reproduce the sensitivity to initial conditions?
3. Ensembles, 100 members:
   a) climatological: 100 ICs taken at different time steps (every 1 TU) of one long M1 trajectory, each propagated
      20 TU by M1 and by the emulator: ensemble mean / spread (std over members) vs time, and the pooled marginal
      PDFs of x, y, z (attractor statistics; 1-D Wasserstein distance emulator vs M1); long-run stability (NaN / blow-up).
   b) perturbed: 100 members x0 + N(0, 0.1^2 I) around 20 test ICs, propagated 10 TU: spread growth to saturation,
      M1 vs emulator (median over the 20 base ICs).
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import xarray as xr  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dameta.dynamics import L63Model  # noqa: E402
from dameta.dynamics.emulator import L63Emulator, teacher_tendency  # noqa: E402
from dameta.dynamics.l63 import rk4_step  # noqa: E402

torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "tmp/results/emulator", ROOT / "tmp/figs"
DT = 0.01
COL = {"M1": "#2a78d6", "teacher": "#52514e", "emulator": "#eb6834"}
STY = {"M1": "-", "teacher": ":", "emulator": "--"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e0",
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "figure.facecolor": "white", "savefig.bbox": "tight"})


@torch.no_grad()
def roll(step, x0, n):
    xs, x = [x0], x0
    for _ in range(n):
        x = step(x)
        xs.append(x)
    return torch.stack(xs, -2)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for e in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{e}", dpi=200)
    plt.close(fig)


def q(a, axis=0):
    return np.nanquantile(a, [0.25, 0.5, 0.75], axis=axis)


def check(name: str, ds: xr.Dataset) -> dict:
    emu = L63Emulator.load(ROOT / f"tmp/emulators/{name}.pt")
    for p in emu.parameters():
        p.requires_grad_(False)
    tcfg = emu.teacher
    m1 = L63Model()
    steps = {"M1": m1.step, "teacher": lambda v: rk4_step(lambda u: teacher_tendency(u, tcfg), v, tcfg.dt),
             "emulator": emu.step}
    xt = torch.tensor(ds.x_true.values)
    clim = float(xt.reshape(-1, 3).std(0).mean())
    thr = 0.4 * clim
    out = {"name": name, "teacher": vars(tcfg) if not hasattr(tcfg, "__dataclass_fields__") else
           {k: getattr(tcfg, k) for k in tcfg.__dataclass_fields__}}
    t = np.arange(501) * DT

    # ---- 1. same initial condition ------------------------------------------------------------------------------
    tr = {k: roll(s, xt[:, 0], 500).numpy() for k, s in steps.items()}
    truth = xt[:, :501].numpy()
    rm = {f"{a} vs M1": np.sqrt(((tr[a] - truth) ** 2).mean(-1)) for a in ("teacher", "emulator")}
    rm["emulator vs teacher"] = np.sqrt(((tr["emulator"] - tr["teacher"]) ** 2).mean(-1))
    c1 = {}
    for k, r in rm.items():
        hit = r > thr
        vpt = np.where(hit.any(1), hit.argmax(1), 500) * DT
        c1[k] = {f"rmse_{lt}TU": float(np.median(r[:, int(lt / DT)])) for lt in (0.1, 0.5, 1, 2)}
        c1[k]["vpt_TU"] = float(np.median(vpt))
    out["1_same_ic"] = c1
    fig, ax = plt.subplots(2, 3, figsize=(10, 5.2), constrained_layout=True)
    for j, i in enumerate([0, 1, 2]):
        for v, row in ((0, 0), (2, 1)):
            a = ax[row, j]
            a.plot(t, truth[i, :, v], STY["M1"], color=COL["M1"], lw=1.2, label="true model M1")
            a.plot(t, tr["teacher"][i, :, v], STY["teacher"], color=COL["teacher"], lw=1.2, label="teacher (biased)")
            a.plot(t, tr["emulator"][i, :, v], STY["emulator"], color=COL["emulator"], lw=1.2, label="emulator")
            a.set_ylabel("xyz"[v])
            if row == 1:
                a.set_xlabel("time [TU]")
        ax[0, j].set_title(f"test IC {i}")
    ax[0, 0].legend(loc="lower left", fontsize=7)
    fig.suptitle(f"[{name}] Check 1 — same initial condition (true x0)")
    save(fig, f"m3_emuqc_{name}_1_same_ic")

    # ---- 2. slightly different initial conditions -----------------------------------------------------------------
    g = torch.Generator().manual_seed(1)
    d = torch.randn(xt.shape[0], 3, generator=g)
    d = 1e-3 * d / d.norm(dim=1, keepdim=True)
    n2 = 1000
    sep, c2 = {}, {}
    for k in ("M1", "emulator"):
        a, b = roll(steps[k], xt[:, 0], n2), roll(steps[k], xt[:, 0] + d, n2)
        sep[k] = (a - b).norm(dim=-1).numpy()
        med = np.median(np.log(sep[k]), 0)
        i0, i1 = int(1 / DT), int(6 / DT)
        c2[k] = {"growth_rate_1to6TU": float(np.polyfit(np.arange(i0, i1) * DT, med[i0:i1], 1)[0]),
                 "saturation_sep_8to10TU": float(np.median(sep[k][:, int(8 / DT):])),
                 "time_to_sep_1_TU": float(np.median(np.argmax(sep[k] > 1, 1) * DT))}
    out["2_pair"] = c2
    t2 = np.arange(n2 + 1) * DT
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
    for k in ("M1", "emulator"):
        lo, md, hi = q(np.log10(sep[k]))
        ax[0].plot(t2, md, STY[k], color=COL[k], label=f"{k} pair (rate {c2[k]['growth_rate_1to6TU']:.2f}/TU)")
        ax[0].fill_between(t2, lo, hi, color=COL[k], alpha=0.15, lw=0)
    ax[0].set_xlabel("time [TU]")
    ax[0].set_ylabel("log10 |x(t) − x'(t)|  (median, IQR)")
    ax[0].legend(fontsize=7)
    ax[0].set_title("separation of two ICs 1e-3 apart")
    for k in ("M1", "emulator"):
        a, b = roll(steps[k], xt[:1, 0], 800), roll(steps[k], xt[:1, 0] + d[:1], 800)
        ax[1].plot(np.arange(801) * DT, a[0, :, 0], STY[k], color=COL[k], lw=1, label=f"{k}: x0")
        ax[1].plot(np.arange(801) * DT, b[0, :, 0], STY[k], color=COL[k], lw=1, alpha=0.45, label=f"{k}: x0 + δ")
    ax[1].set_xlabel("time [TU]")
    ax[1].set_ylabel("x")
    ax[1].legend(fontsize=7, ncol=2)
    ax[1].set_title("example pair (test IC 0)")
    fig.suptitle(f"[{name}] Check 2 — slightly different initial conditions")
    save(fig, f"m3_emuqc_{name}_2_pair")

    # ---- 3a. climatological 100-member ensemble ----------------------------------------------------------------
    long = roll(m1.step, xt[0, 0], 100 * 100)                       # 100 TU of M1
    ens0 = long[100::100][:100]                                      # 100 states, one per TU
    n3 = 2000
    ens = {k: roll(steps[k], ens0, n3).numpy() for k in ("M1", "emulator")}
    finite = bool(np.isfinite(ens["emulator"]).all()) and float(np.abs(ens["emulator"]).max()) < 100
    pooled = {k: ens[k][:, n3 // 2:].reshape(-1, 3) for k in ens}
    c3 = {"emulator_stable_20TU": finite,
          "wasserstein_xyz": [float(wasserstein_distance(pooled["M1"][:, v], pooled["emulator"][:, v]))
                              for v in range(3)],
          "mean_xyz_M1": pooled["M1"].mean(0).tolist(), "mean_xyz_emulator": pooled["emulator"].mean(0).tolist(),
          "std_xyz_M1": pooled["M1"].std(0).tolist(), "std_xyz_emulator": pooled["emulator"].std(0).tolist()}
    # ---- 3b. perturbed 100-member ensembles around 20 test ICs -------------------------------------------------
    g = torch.Generator().manual_seed(2)
    base = xt[:20, 0]
    pert = base[:, None, :] + 0.1 * torch.randn(20, 100, 3, generator=g)
    spread = {}
    for k in ("M1", "emulator"):
        tr3 = roll(steps[k], pert.reshape(-1, 3), 1000).reshape(20, 100, 1001, 3)
        spread[k] = tr3.std(1).mean(-1).numpy()                     # [20, T] mean over variables of member std
    c3["perturbed_spread_median"] = {k: {f"{lt}TU": float(np.median(spread[k][:, int(lt / DT)]))
                                         for lt in (0, 1, 2, 4, 6, 10)} for k in spread}
    out["3_ensemble"] = c3
    t3 = np.arange(n3 + 1) * DT
    fig, ax = plt.subplots(2, 3, figsize=(11, 5.6), constrained_layout=True)
    for v in range(3):
        a = ax[0, v]
        for k in ("M1", "emulator"):
            a.plot(t3, ens[k][:, :, v].mean(0), STY[k], color=COL[k], lw=1, label=f"{k} mean")
            a.plot(t3, ens[k][:, :, v].std(0), STY[k], color=COL[k], lw=1, alpha=0.5, label=f"{k} spread")
        a.set_title(f"3a climatological ensemble: {'xyz'[v]}")
        a.set_xlabel("time [TU]")
    ax[0, 0].legend(fontsize=7)
    for v in range(2):
        a = ax[1, v]
        bins = np.linspace(min(pooled["M1"][:, v * 2].min(), pooled["emulator"][:, v * 2].min()),
                           max(pooled["M1"][:, v * 2].max(), pooled["emulator"][:, v * 2].max()), 60)
        for k in ("M1", "emulator"):
            a.hist(pooled[k][:, v * 2], bins=bins, density=True, histtype="step", color=COL[k], ls=STY[k], lw=1.2,
                   label=k)
        a.set_title(f"pooled PDF of {'xz'[v]} (10-20 TU)")
    ax[1, 0].legend(fontsize=7)
    a = ax[1, 2]
    for k in ("M1", "emulator"):
        lo, md, hi = q(spread[k])
        tt = np.arange(spread[k].shape[1]) * DT
        a.plot(tt, md, STY[k], color=COL[k], label=k)
        a.fill_between(tt, lo, hi, color=COL[k], alpha=0.15, lw=0)
    a.set_yscale("log")
    a.set_xlabel("time [TU]")
    a.set_title("3b perturbed ensembles: spread (σ0 = 0.1)")
    a.legend(fontsize=7)
    fig.suptitle(f"[{name}] Check 3 — 100-member ensembles")
    save(fig, f"m3_emuqc_{name}_3_ensemble")
    return out


def to_md(results: list) -> str:
    lines = ["| check | metric | " + " | ".join(r["name"] for r in results) + " |",
             "|---|---|" + "---|" * len(results)]

    def row(c, m, f):
        lines.append(f"| {c} | {m} | " + " | ".join(f(r) for r in results) + " |")
    row("teacher", "parameters", lambda r: ", ".join(f"{k}={v:g}" for k, v in r["teacher"].items()
                                                     if v != {"sigma": 10.0, "rho": 28.0, "beta": 8 / 3,
                                                              "xz_scale": 1.0, "dt": 0.01}[k]) or "exact L63")
    for lt in (0.1, 0.5, 1, 2):
        row("1 same IC", f"RMSE emulator vs M1 @{lt} TU", lambda r, lt=lt: f"{r['1_same_ic']['emulator vs M1'][f'rmse_{lt}TU']:.3f}")
    row("1 same IC", "RMSE teacher vs M1 @1 TU (intended)", lambda r: f"{r['1_same_ic']['teacher vs M1']['rmse_1TU']:.3f}")
    row("1 same IC", "RMSE emulator vs teacher @1 TU (learning)", lambda r: f"{r['1_same_ic']['emulator vs teacher']['rmse_1TU']:.3f}")
    row("1 same IC", "VPT emulator vs M1 [TU]", lambda r: f"{r['1_same_ic']['emulator vs M1']['vpt_TU']:.2f}")
    row("2 pair", "growth rate M1 / emulator [1/TU]",
        lambda r: f"{r['2_pair']['M1']['growth_rate_1to6TU']:.2f} / {r['2_pair']['emulator']['growth_rate_1to6TU']:.2f}")
    row("2 pair", "time to separation 1 M1 / emulator [TU]",
        lambda r: f"{r['2_pair']['M1']['time_to_sep_1_TU']:.2f} / {r['2_pair']['emulator']['time_to_sep_1_TU']:.2f}")
    row("2 pair", "separation at 8-10 TU M1 / emulator",
        lambda r: f"{r['2_pair']['M1']['saturation_sep_8to10TU']:.1f} / {r['2_pair']['emulator']['saturation_sep_8to10TU']:.1f}")
    row("3a ensemble", "stable 20 TU", lambda r: str(r["3_ensemble"]["emulator_stable_20TU"]))
    row("3a ensemble", "Wasserstein x, y, z (vs M1)", lambda r: ", ".join(f"{w:.2f}" for w in r["3_ensemble"]["wasserstein_xyz"]))
    row("3a ensemble", "clim std z M1 / emulator",
        lambda r: f"{r['3_ensemble']['std_xyz_M1'][2]:.2f} / {r['3_ensemble']['std_xyz_emulator'][2]:.2f}")
    for lt in (1, 2, 4, 10):
        row("3b ensemble", f"spread @{lt} TU M1 / emulator",
            lambda r, lt=lt: f"{r['3_ensemble']['perturbed_spread_median']['M1'][f'{lt}TU']:.2f} / "
                             f"{r['3_ensemble']['perturbed_spread_median']['emulator'][f'{lt}TU']:.2f}")
    return "\n".join(lines) + "\n"


def main(names):
    ds = xr.open_dataset(ROOT / "tmp/data/l63_test_seed42.nc")
    res = []
    for n in names:
        r = check(n, ds)
        RES.mkdir(parents=True, exist_ok=True)
        (RES / f"{n}_qc.json").write_text(json.dumps(r, indent=1, default=float))
        res.append(r)
    md = to_md(res)
    (RES / "emulator_qc.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main(sys.argv[1:] or ["rho275"])
