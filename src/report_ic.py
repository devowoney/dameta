"""
Consolidated, paper-ready report of 4D-Var initial-condition (IC) estimation: E1 (physical model M1) and E3 (4D-Var
through the emulators clean / xz095 / xz090), on the held-out splits test (in-distribution) and test_ood
(lobe-transition region).

Usage (project root, env oceanai):   python src/report_ic.py
Inputs : tmp/results/e1/*.nc, tmp/results/e3_<emu>/*.nc, tmp/data/*.nc, tmp/emulators/*.pt
Outputs: tmp/report/table{1,2,3}_*.{md,csv}, tmp/report/fig{1,2,3}_*.{png,pdf}

Statistics (fixed before looking at the tables):
    - every summary is a median over the 200 cases of a split, with a 95 % percentile-bootstrap CI
      (2000 resamples of cases, seed 0);
    - E3 vs E1 comparisons are PAIRED by case (same IC, obs, background): median paired difference + bootstrap CI,
      two-sided Wilcoxon signed-rank p-value, fraction of cases where E3 is worse;
    - "forecast" always means the analysis integrated with the TRUE model M1 (IC quality), unless stated otherwise;
    - cost = gradient evaluations to reach J <= 1.01 J*, J* = best final J over the classical optimizers on the same
      case with the same dynamics.
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import xarray as xr
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dameta.da import CostConfig  # noqa: E402
from dameta.dynamics import L63Model  # noqa: E402
from dameta.dynamics.emulator import L63Emulator  # noqa: E402
from dameta.eval import gradient_fidelity  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES, OUT = ROOT / "tmp/results", ROOT / "tmp/report"
DT, LEAD = 0.01, 100                     # forecast lead index 100 = 1 time unit beyond the window
SPLITS = ["test", "test_ood"]
EMUS = ["clean", "xz095", "xz090"]
# Fixed categorical order (validated palette: CVD + normal-vision PASS; slots 3-4 low contrast -> markers/labels)
COLOR = {"E1": "#2a78d6", "clean": "#eb6834", "xz095": "#1baf7a", "xz090": "#eda100", "background": "#52514e"}
LABEL = {"background": "background (no DA)", "E1": "E1: exact model", "clean": "E3: emulator clean",
         "xz095": "E3: emulator xz095 (main)", "xz090": "E3: emulator xz090 (stress)"}
STYLE = {"E1": "-", "clean": "--", "xz095": "-.", "xz090": ":", "background": (0, (1, 1))}
MARK = {"E1": "o", "clean": "s", "xz095": "^", "xz090": "D", "background": "x"}
RNG = np.random.default_rng(0)

plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e0",
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "figure.facecolor": "white", "savefig.bbox": "tight"})


# ----------------------------------------------------------------------------------------------- statistics helpers
def boot_median(x: np.ndarray, n: int = 2000) -> tuple:
    """Median and 95 % percentile-bootstrap CI over cases (NaN cases dropped and reported via n)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    meds = np.median(x[RNG.integers(0, len(x), (n, len(x)))], axis=1)
    return float(np.median(x)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def fmt_ci(t: tuple, f: str = ".3f") -> str:
    m, lo, hi = t
    return "—" if not np.isfinite(m) else f"{m:{f}} [{lo:{f}}, {hi:{f}}]"


def vpt(fc: np.ndarray, thr: float) -> np.ndarray:
    """Valid prediction time (TU) per case: first lead with RMSE > thr; full range if never; NaN if diverged."""
    above = fc > thr
    out = np.where(above.any(1), above.argmax(1), fc.shape[1] - 1).astype(float) * DT
    out[~np.isfinite(fc).all(1)] = np.nan
    return out


def evals_to(j: np.ndarray, j_star: np.ndarray, tol: float = 0.01) -> np.ndarray:
    hit = j <= (1 + tol) * j_star[:, None]
    return np.where(hit.any(1), hit.argmax(1), np.nan).astype(float)


# ----------------------------------------------------------------------------------------------- loading
def load() -> dict:
    """runs[(dyn, split, constraint, pattern, method)] = per-case arrays (forecast always with M1)."""
    runs = {}
    for dyn, folder in [("E1", "e1")] + [(e, f"e3_{e}") for e in EMUS]:
        for p in sorted((RES / folder).glob("*.nc")):
            ds = xr.open_dataset(p)
            c = json.loads(ds.attrs["config"])
            if c["split"] not in SPLITS:
                continue
            e1 = dyn == "E1"
            runs[(dyn, c["split"], c["constraint"], c["pattern"], c["method"])] = {
                "lr": c["lr"], "ic": ds.ic.values,
                "ic_err": ds.rmse_x0.values, "ic_err_b": ds.rmse_x0_b.values,
                "fc": (ds.rmse_fc if e1 else ds.rmse_fc_m1).values,
                "fc_emu": None if e1 else ds.rmse_fc.values,
                "fc_b": ds.rmse_fc_b.values if e1 else None,           # background forecast with M1 (E1 files only)
                "J_true": ds.J_iter.values[:, -1] if e1 else ds.J_true_final.values,
                "J_iter": ds.J_iter.values, "diverged": ds.diverged.values,
            }
    return runs


def add_costs(runs: dict) -> None:
    """evals to 1 % of J* (J* = best final J over GD/Adagrad on the same case, same dynamics)."""
    groups = {}
    for k, r in runs.items():
        groups.setdefault(k[:4], []).append(r)
    for rs in groups.values():
        j_star = np.nanmin(np.stack([r["J_iter"][:, -1] for r in rs]), axis=0)
        for r in rs:
            r["evals"] = evals_to(r["J_iter"], j_star)


# ----------------------------------------------------------------------------------------------- tables
def table1_e1(runs: dict, thr: float) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        ref = runs[("E1", split, "strong", "full", "adagrad")]
        rows.append({"split": split, "constraint": "—", "obs": "—", "method": "background (no DA)", "lr": "—",
                     "diverged": "—", "IC RMSE": fmt_ci(boot_median(ref["ic_err_b"])),
                     "fc RMSE @1 TU": fmt_ci(boot_median(ref["fc_b"][:, LEAD])),
                     "VPT [TU]": fmt_ci(boot_median(vpt(ref["fc_b"], thr)), ".2f"), "J (χ² ref m/2)": "—",
                     "evals to 1 % of J*": "—", "reach": "—"})
        for constraint in ["strong", "weak"]:
            for pattern, m in [("full", 33), ("x_only", 11)]:
                for method in ["adagrad", "gd"]:
                    r = runs[("E1", split, constraint, pattern, method)]
                    rows.append({
                        "split": split, "constraint": constraint, "obs": pattern, "method": method,
                        "lr": f"{r['lr']:g}", "diverged": f"{int(r['diverged'].sum())}/{len(r['ic'])}",
                        "IC RMSE": fmt_ci(boot_median(r["ic_err"])),
                        "fc RMSE @1 TU": fmt_ci(boot_median(r["fc"][:, LEAD])),
                        "VPT [TU]": fmt_ci(boot_median(vpt(r["fc"], thr)), ".2f"),
                        "J (χ² ref m/2)": f"{np.nanmedian(r['J_true']):.2f} ({m / 2:g})",
                        "evals to 1 % of J*": fmt_ci(boot_median(r["evals"]), ".0f"),
                        "reach": f"{np.isfinite(r['evals']).mean():.0%}"})
    return pd.DataFrame(rows)


def table2_e3(runs: dict, thr: float) -> pd.DataFrame:
    """E3 vs E1, paired by case, classical Adagrad (claim baseline) — every available configuration."""
    rows = []
    for emu in EMUS:
        for split in SPLITS:
            for constraint in ["strong", "weak"]:
                for pattern in ["full", "x_only"]:
                    k3 = (emu, split, constraint, pattern, "adagrad")
                    if k3 not in runs:
                        continue
                    e3, e1 = runs[k3], runs[("E1", split, constraint, pattern, "adagrad")]
                    assert np.array_equal(e3["ic"], e1["ic"]), "cases must be paired"
                    d_fc = e3["fc"][:, LEAD] - e1["fc"][:, LEAD]
                    d_ic = e3["ic_err"] - e1["ic_err"]
                    ok = np.isfinite(d_fc)
                    rows.append({
                        "emulator": emu, "split": split, "constraint": constraint, "obs": pattern,
                        "IC RMSE E3 (E1)": f"{np.nanmedian(e3['ic_err']):.3f} ({np.nanmedian(e1['ic_err']):.3f})",
                        "Δ IC RMSE": fmt_ci(boot_median(d_ic)),
                        "fc @1 TU E3 (E1)": f"{np.nanmedian(e3['fc'][:, LEAD]):.2f} ({np.nanmedian(e1['fc'][:, LEAD]):.2f})",
                        "Δ fc @1 TU (gap)": fmt_ci(boot_median(d_fc)),
                        "p (Wilcoxon)": f"{wilcoxon(d_fc[ok]).pvalue:.1e}",
                        "E3 worse": f"{(d_fc[ok] > 0).mean():.0%}",
                        "VPT E3 (E1)": f"{np.nanmedian(vpt(e3['fc'], thr)):.2f} ({np.nanmedian(vpt(e1['fc'], thr)):.2f})",
                        "P1 J_true E3 / E1": f"{np.nanmedian(e3['J_true'] / e1['J_true']):.2f}",
                        "evals E3 (E1)": f"{np.nanmedian(e3['evals']):.0f} ({np.nanmedian(e1['evals']):.0f})"})
    return pd.DataFrame(rows)


def per_case_g1() -> dict:
    """Per-case G1 (cos and relative error, strong/full) at x0_b and at x0_true for each emulator and split."""
    torch.set_default_dtype(torch.float64)
    m1, cfg, out = L63Model(), CostConfig(), {}
    for emu_name in EMUS:
        emu = L63Emulator.load(ROOT / f"tmp/emulators/{emu_name}.pt")
        for p in emu.parameters():
            p.requires_grad_(False)
        for split in SPLITS:
            ds = xr.open_dataset(ROOT / f"tmp/data/l63_{split}_seed42.nc")
            y, xb = torch.tensor(ds.y_obs.values), torch.tensor(ds.x0_b.values)
            xt, mask = torch.tensor(ds.x_true.values[:, 0]), torch.tensor(ds.obs_mask.sel(pattern="full").values)
            for where, x0 in [("xb", xb), ("truth", xt)]:
                g = gradient_fidelity(emu.step, m1.step, y, mask, xb, cfg, x0, weak=False)
                out[(emu_name, split, where)] = {k: v.numpy() for k, v in g.items()}
            with torch.no_grad():                                  # model error over one obs interval (10 steps)
                tr = torch.tensor(ds.x_true.values[:, :101])
                x = tr[:, 0:91:10].reshape(-1, 3)
                for _ in range(10):
                    x = emu.step(x)
                out[(emu_name, split, "q_err")] = (x - tr[:, 10:101:10].reshape(-1, 3)).std(0).numpy()
    return out


def table3_emulators(g1: dict) -> pd.DataFrame:
    rows = []
    for emu in EMUS:
        diag = json.loads((RES / f"emulator/{emu}_diag.json").read_text())
        for split in SPLITS:
            d = diag[split]
            q = g1[(emu, split, "q_err")]
            rows.append({
                "emulator": emu, "split": split, "teacher xz_scale": diag["teacher"]["xz_scale"],
                "free fc RMSE @1 TU (true x0)": f"{d['free_fc_rmse_1TU']['med']:.3f}",
                "VPT [TU] (true x0)": f"{d['vpt_TU']['med']:.2f}",
                "λ₁ (M1 0.904)": f"{d['lyapunov']['med']:.3f}",
                "TL cos (window)": f"{d['tl_cos']['med']:.3f}",
                "G1 cos @ x0_b": fmt_ci(boot_median(g1[(emu, split, 'xb')]['cos'])),
                "G1 cos @ x0_true": fmt_ci(boot_median(g1[(emu, split, 'truth')]['cos'])),
                "G1 cos<0 @ x0_true": f"{(g1[(emu, split, 'truth')]['cos'] < 0).mean():.0%}",
                "model err / obs interval (x,y,z)": ", ".join(f"{v:.3f}" for v in q),
            })
    return pd.DataFrame(rows)


def to_md(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(lines + ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)])


# ----------------------------------------------------------------------------------------------- figures
def save(fig, name: str) -> None:
    for e in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{e}", dpi=200)
    plt.close(fig)


def fig1_overview(runs: dict, thr: float) -> None:
    """IC error, forecast error vs lead, VPT — strong constraint, full obs, Adagrad; rows = test / test_ood."""
    order = ["background", "E1"] + EMUS
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), gridspec_kw={"width_ratios": [1, 1.6, 1]})
    for i, split in enumerate(SPLITS):
        e1 = runs[("E1", split, "strong", "full", "adagrad")]
        data = {"background": (e1["ic_err_b"], e1["fc_b"]), "E1": (e1["ic_err"], e1["fc"])}
        for emu in EMUS:
            r = runs[(emu, split, "strong", "full", "adagrad")]
            data[emu] = (r["ic_err"], r["fc"])
        # (a) IC error distributions
        ax = axes[i, 0]
        bp = ax.boxplot([data[k][0] for k in order], widths=0.6, patch_artist=True, showfliers=False,
                        medianprops={"color": "#0b0b0b", "lw": 1.4})
        for patch, k in zip(bp["boxes"], order):
            patch.set(facecolor=COLOR[k], alpha=0.35, edgecolor=COLOR[k], lw=1.2)
        ax.set_xticks(range(1, len(order) + 1), ["bg", "E1", "clean", "xz095", "xz090"])
        ax.set_ylabel("IC RMSE (analysis vs truth)")
        ax.set_title(f"({'ad'[i]}) IC error — {split}", loc="left", fontsize=9)
        # (b) forecast RMSE vs lead (median + IQR band for E1 and xz095 only, to keep the panel readable)
        ax = axes[i, 1]
        lead = np.arange(data["E1"][1].shape[1]) * DT
        for k in order:
            fc = data[k][1]
            med = np.nanmedian(fc, 0)
            ax.plot(lead, med, color=COLOR[k], ls=STYLE[k], lw=2 if k != "background" else 1.4, label=LABEL[k])
            if k in ("E1", "xz095"):
                ax.fill_between(lead, np.nanpercentile(fc, 25, 0), np.nanpercentile(fc, 75, 0), color=COLOR[k],
                                alpha=0.12, lw=0)
            ax.plot(lead[-1], med[-1], MARK[k], color=COLOR[k], ms=5)
        ax.axhline(thr, color="#52514e", lw=0.8, ls="--", label="VPT threshold (0.4 × clim. std)")
        ax.set_yscale("log")
        ax.set_xlabel("lead time beyond window (TU)")
        ax.set_ylabel("forecast RMSE with M1 (median)")
        ax.set_title(f"({'be'[i]}) forecast from analysis — {split}", loc="left", fontsize=9)
        # (c) VPT distributions
        ax = axes[i, 2]
        vp = [vpt(data[k][1], thr) for k in order]
        bp = ax.boxplot([v[np.isfinite(v)] for v in vp], widths=0.6, patch_artist=True, showfliers=False,
                        medianprops={"color": "#0b0b0b", "lw": 1.4})
        for patch, k in zip(bp["boxes"], order):
            patch.set(facecolor=COLOR[k], alpha=0.35, edgecolor=COLOR[k], lw=1.2)
        ax.set_xticks(range(1, len(order) + 1), ["bg", "E1", "clean", "xz095", "xz090"])
        ax.set_ylabel("valid prediction time (TU)")
        ax.set_title(f"({'cf'[i]}) VPT — {split}", loc="left", fontsize=9)
    h, lab = axes[0, 1].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("4D-Var initial-condition estimation — strong constraint, full observations, Adagrad (200 cases)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig1_ic_estimation_overview")


def fig2_cost(runs: dict) -> None:
    """E1 optimizer cost: convergence to J* and evals-to-1 % (median + 95 % CI) for every configuration."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1, 1.5]})
    col = {"adagrad": COLOR["E1"], "gd": "#52514e"}
    ls = {"adagrad": "-", "gd": "--"}
    for ax, constraint in zip(axes[:2], ["strong", "weak"]):
        rs = {m: runs[("E1", "test", constraint, "full", m)] for m in ["adagrad", "gd"]}
        j_star = np.nanmin(np.stack([r["J_iter"][:, -1] for r in rs.values()]), 0)
        for m, r in rs.items():
            exc = r["J_iter"] / j_star[:, None] - 1
            k = np.arange(exc.shape[1])
            ax.plot(k[1:], np.nanmedian(exc, 0)[1:], color=col[m], ls=ls[m], lw=2, label=f"{m} (lr {r['lr']:g})")
            ax.fill_between(k[1:], np.nanpercentile(exc, 25, 0)[1:], np.nanpercentile(exc, 75, 0)[1:],
                            color=col[m], alpha=0.12, lw=0)
        ax.axhline(0.01, color="#52514e", lw=0.8, ls=":")
        ax.text(1.2, 0.0115, "1 % of J*", fontsize=7, color="#52514e")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 10)
        ax.set_xlabel("gradient evaluations")
        ax.set_ylabel("J / J* − 1 (median, IQR)")
        ax.set_title(f"({'ab'[constraint == 'weak']}) {constraint} constraint, full obs — test", loc="left", fontsize=9)
        ax.legend(fontsize=7, loc="lower left")
    ax = axes[2]
    configs = [(s, c, p) for s in SPLITS for c in ["strong", "weak"] for p in ["full", "x_only"]]
    x = np.arange(len(configs))
    for j, m in enumerate(["adagrad", "gd"]):
        stats = [boot_median(runs[("E1", s, c, p, m)]["evals"]) for s, c, p in configs]
        med = np.array([t[0] for t in stats])
        err = np.array([[t[0] - t[1], t[2] - t[0]] for t in stats]).T
        ax.bar(x + (j - 0.5) * 0.38, med, 0.36, color=col[m], alpha=0.85 if m == "adagrad" else 0.55, label=m,
               edgecolor="white", lw=2)
        ax.errorbar(x + (j - 0.5) * 0.38, med, yerr=err, fmt="none", ecolor="#0b0b0b", lw=0.9, capsize=2)
    ax.set_xticks(x, [f"{'T' if s == 'test' else 'OOD'}\n{c[:2]}/{'full' if p == 'full' else 'x'}"
                      for s, c, p in configs], fontsize=7)
    ax.set_yscale("log")
    ax.set_ylabel("evals to 1 % of J* (median, 95 % CI)")
    ax.set_title("(c) cost of classical 4D-Var (E1), all configurations", loc="left", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "fig2_e1_optimizer_cost")


def fig3_fidelity(runs: dict, g1: dict) -> None:
    """Gradient fidelity G1 per emulator and its link to the analysis-forecast penalty (test, strong/full)."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), gridspec_kw={"width_ratios": [1, 1, 1.3]})
    for ax, where, title in [(axes[0], "xb", "(a) G1 at the background x0_b"),
                             (axes[1], "truth", "(b) G1 at the true IC (near the minimum)")]:
        data = [g1[(e, "test", where)]["cos"] for e in EMUS]
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=True, medianprops={"color": "#0b0b0b"},
                        flierprops={"ms": 2, "alpha": 0.4})
        for patch, e in zip(bp["boxes"], EMUS):
            patch.set(facecolor=COLOR[e], alpha=0.35, edgecolor=COLOR[e])
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.set_xticks([1, 2, 3], EMUS)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("cos(∇J emulator, ∇J exact)")
        ax.set_title(title + " — test", loc="left", fontsize=9)
    ax = axes[2]                                   # between-emulator relation (per-case relation is weak, see text)
    for e in EMUS:
        for split, filled in [("test", True), ("test_ood", False)]:
            e3, e1 = runs[(e, split, "strong", "full", "adagrad")], runs[("E1", split, "strong", "full", "adagrad")]
            gx = boot_median(g1[(e, split, "truth")]["rel_err"])
            gy = boot_median(e3["fc"][:, LEAD] - e1["fc"][:, LEAD])
            ax.errorbar(gx[0], gy[0], xerr=[[gx[0] - gx[1]], [gx[2] - gx[0]]], yerr=[[gy[0] - gy[1]], [gy[2] - gy[0]]],
                        fmt=MARK[e], ms=8, color=COLOR[e], mfc=COLOR[e] if filled else "white", mew=1.6, lw=1,
                        capsize=2, label=f"{e} — {split}")
            off = {("xz090", False): (-8, 7, "right"), ("xz095", True): (8, -10, "left")}.get((e, filled), (8, -3, "left"))
            ax.annotate(f"{e} ({'test' if filled else 'OOD'})", (gx[0], gy[0]), xytext=off[:2], ha=off[2],
                        textcoords="offset points", fontsize=7, color="#0b0b0b")
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=0.05)
    ax.set_ylim(-0.15, 10)
    ax.set_xlabel("G1 relative gradient error at the true IC (median, 95 % CI)")
    ax.set_ylabel("paired gap: fc RMSE @1 TU, E3 − E1")
    ax.set_title("(c) gradient fidelity vs analysis penalty", loc="left", fontsize=9)
    ax.text(0.02, 0.97, "filled = test, open = test_ood", transform=ax.transAxes, fontsize=7, va="top",
            color="#52514e")
    fig.tight_layout()
    save(fig, "fig3_e3_gradient_fidelity")


# ----------------------------------------------------------------------------------------------- main
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    thr = 0.4 * float(xr.open_dataset(ROOT / "tmp/data/l63_train_seed42.nc").x_true.std())
    runs = load()
    add_costs(runs)
    g1 = per_case_g1()
    tables = {"table1_e1_classical": table1_e1(runs, thr), "table2_e3_vs_e1_paired": table2_e3(runs, thr),
              "table3_emulator_fidelity": table3_emulators(g1)}
    for name, df in tables.items():
        df.to_csv(OUT / f"{name}.csv", index=False)
        (OUT / f"{name}.md").write_text(to_md(df) + "\n")
        print(f"\n## {name}\n" + to_md(df))
    fig1_overview(runs, thr)
    fig2_cost(runs)
    fig3_fidelity(runs, g1)
    print(f"\nVPT threshold = {thr:.2f}; outputs in {OUT}")


if __name__ == "__main__":
    main()
