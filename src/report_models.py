"""
Model-rectangle characterization (pilot 2026-11-20): distances between the 5 models of src/configs/models.yaml.

    F1 ──▶ E1        vertical edges   (L1–F1, F1–F2, L1–F2, E1–E2): physical misspecification   (Problem 1)
    │       │        horizontal edges (F1–E1, F2–E2):                 emulator approximation      (Problem 2)
    F2 ──▶ E2        total edges      (L1–E1, L1–E2)

Per edge (A, B), on D1 v2 val and test ICs (statistics: mean ± std over cases):
  m_pooled       RMS over (x, y, z) of the pooled std of the one-obs-interval difference M_A^10(x) - M_B^10(x), from
                 TRUE states every obs interval, divided by sigma_o (same metric as the false-model screen)
  m_case         per-case RMS of that difference / sigma_o
  rmse_0.5/1TU   |M_A^n(x0) - M_B^n(x0)| (RMS over variables) from the true x0
  vpt            first lead where that difference exceeds 0.4 x climatological std [TU]
  grad J_o       4D-Var observation-term gradient of A vs B (strong constraint, full obs, window and obs of D1 v2):
                 cos and relative error |g_A - g_B| / |g_B|, at the true x0 and at x0 + N(0, I) (fixed seed).
                 J_b is left out on purpose: it is identical for both models and would push cos towards 1.
  Probabilistic view (pilot 2026-11-20): quantiles q05/q50/q95 of the one-obs-interval difference norm / (sqrt(3)
                 sigma_o) over all samples; invariant measures of A and B (50 ICs x 40 TU after 10 TU burn-in):
                 1-D Wasserstein per variable and 3-D energy distance; noise floor = L1 vs L1 from other ICs.
Per model: lambda1 (renormalised TL propagation, 20 TU, 50 ICs) and its Lyapunov time.
Figures: tmp/figs/rectangle (2x2 diagram), tmp/figs/rectangle_pdfs (marginal invariant PDFs of the 5 models and the
one-obs-interval difference distribution of each edge).

Usage (project root, after eps1 / eps2 are set in models.yaml and E1 / E2 are trained):
    python src/report_models.py           -> tmp/results/models/rectangle.{csv,md,json}, tmp/figs/rectangle.{png,pdf}
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dameta.da import CostConfig, FourDVarCost  # noqa: E402
from dameta.dynamics.models import build_model, load_registry, model_params, theta_distance  # noqa: E402
from dameta.eval.emulator_metrics import lyapunov_exponent  # noqa: E402
from report_e1 import to_markdown  # noqa: E402
from screen_teachers import climate_pool, measure_distances  # noqa: E402

torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parent.parent
SIGMA_O = float(np.sqrt(2.0))
MODELS = ["l1", "f1", "f2", "e1", "e2"]
EDGES = [("l1", "f1", "vertical"), ("f1", "f2", "vertical"), ("l1", "f2", "vertical"), ("e1", "e2", "vertical"),
         ("f1", "e1", "horizontal"), ("f2", "e2", "horizontal"), ("l1", "e1", "total"), ("l1", "e2", "total")]


def ms(a) -> str:
    a = np.asarray(a, dtype=float)
    return f"{np.nanmean(a):.3f} ± {np.nanstd(a):.3f}"


@torch.no_grad()
def roll(step, x0, n):
    xs, x = [x0], x0
    for _ in range(n):
        x = step(x)
        xs.append(x)
    return torch.stack(xs, -2)


def grad_jo(step, ds, x0):
    """Gradient of the observation term J_o only (b_inv = 0), strong constraint, full obs."""
    ccfg = CostConfig(window=int(ds.attrs["window"]), obs_every=int(ds.attrs["obs_every"]),
                      sigma_o2=float(ds.attrs["sigma_o2"]))
    n = x0.shape[0]
    cost = FourDVarCost(step, torch.tensor(ds.y_obs.values), torch.tensor(ds.obs_mask.sel(pattern="full").values),
                        x0, ccfg, weak=False, b_inv=torch.zeros(n, 3, 3))
    return cost.grad(cost.pack(x0, None))[1]


def edge_metrics(sa, sb, ds, xt, pts, oe):
    n_int = 50
    starts = xt[:, :n_int * oe:oe].reshape(-1, 3)
    with torch.no_grad():
        a, b = starts, starts
        for _ in range(oe):
            a, b = sa(a), sb(b)
    d = (a - b).reshape(xt.shape[0], n_int, 3)
    e = d.reshape(-1, 3).std(0)
    en = (d.norm(dim=-1) / (np.sqrt(3) * SIGMA_O)).reshape(-1)
    qs = torch.quantile(en, torch.tensor([0.05, 0.5, 0.95])).numpy()
    m_case = (d ** 2).mean((1, 2)).sqrt().numpy() / SIGMA_O
    clim = float(xt.reshape(-1, 3).std(0).mean())
    ta, tb = roll(sa, xt[:, 0], 300), roll(sb, xt[:, 0], 300)
    diff = ((ta - tb) ** 2).mean(-1).sqrt()
    hit = (diff > 0.4 * clim).numpy()
    vpt = np.where(hit.any(1), hit.argmax(1), 300) * 0.01
    row = {"m_pooled": float(e.pow(2).mean().sqrt()) / SIGMA_O, "m_case": ms(m_case),
           "err_q05": float(qs[0]), "err_q50": float(qs[1]), "err_q95": float(qs[2]),
           "rmse_0.5TU": ms(diff[:, 50].numpy()), "rmse_1TU": ms(diff[:, 100].numpy()), "vpt_TU": ms(vpt)}
    row["_err_samples"] = en.numpy()
    for tag, x0 in pts.items():
        ga, gb = grad_jo(sa, ds, x0), grad_jo(sb, ds, x0)
        cos = torch.nn.functional.cosine_similarity(ga, gb, dim=1).numpy()
        rel = ((ga - gb).norm(dim=1) / gb.norm(dim=1)).numpy()
        row[f"cos_gJo_{tag}"], row[f"rel_gJo_{tag}"] = ms(cos), ms(rel)
        row[f"_cos_{tag}"] = float(np.mean(cos))
    return row


def figure(df: pd.DataFrame, reg, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pos = {"l1": (-1.3, 0.5), "f1": (0, 1), "e1": (1.6, 1), "f2": (0, 0), "e2": (1.6, 0)}
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    val = df[df.split == "test"].set_index("edge")
    for a, b, kind in EDGES:
        if a == "l1" and b != "f1":
            continue
        (x0, y0), (x1, y1) = pos[a], pos[b]
        col = {"vertical": "#2a78d6", "horizontal": "#eb6834", "total": "#999"}[kind]
        ax.plot([x0, x1], [y0, y1], color=col, lw=2, zorder=1)
        r = val.loc[f"{a.upper()}–{b.upper()}"]
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, f"m={r.m_pooled:.2f}σo\ncos∇Jo={r._cos_truth:.3f}", fontsize=7,
                ha="center", va="center", bbox=dict(fc="white", ec="none", alpha=0.85))
    for k, (x, y) in pos.items():
        lab = k.upper()
        if k in ("f1", "f2"):
            p = model_params(k, reg)
            lab += f"\nσ{p.sigma:.3g} ρ{p.rho:.4g} β{p.beta:.3g}"
        ax.text(x, y, lab, ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round", fc="#f4f3ee", ec="#52514e"), zorder=2)
    ax.text(0.8, 1.3, "emulator approximation (Problem 2) →", ha="center", fontsize=8, color="#eb6834")
    ax.text(-0.55, 0.5, "model\nerror\n(P1) ↓", ha="center", va="center", fontsize=8, color="#2a78d6")
    ax.set(xlim=(-1.9, 2.2), ylim=(-0.4, 1.5))
    ax.axis("off")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=150)


def pdf_figure(pools, err, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"l1": "#0b0b0b", "f1": "#2a78d6", "f2": "#1b3f8a", "e1": "#eb6834", "e2": "#a33d12"}
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.2))
    for i, v in enumerate("xyz"):
        for m, pool in pools.items():
            ax[i].hist(pool[:, i].numpy(), bins=80, density=True, histtype="step", color=col[m], label=m.upper(),
                       lw=1.6 if m == "l1" else 1)
        ax[i].set(xlabel=v, ylabel="invariant PDF")
    ax[0].legend(fontsize=7)
    for edge, e in err.items():                        # empirical CDF: all edges readable on one log axis
        xs = np.sort(e)
        ax[3].plot(xs, np.arange(1, len(xs) + 1) / len(xs), label=edge, lw=1.2,
                   ls="--" if edge in ("F1–E1", "F2–E2") else "-")
    ax[3].axvline(1 / np.sqrt(3), color="grey", ls=":", lw=0.8)
    ax[3].set(xscale="log", xlabel="one-obs-interval |Δ| / (√3 σ_o)", ylabel="CDF (test)")
    ax[3].legend(fontsize=6)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=150)


def main():
    reg = load_registry()
    steps = {m: build_model(m, reg, str(ROOT / "tmp/emulators"))[0] for m in MODELS}
    desc = {m: build_model(m, reg, str(ROOT / "tmp/emulators"))[1] for m in MODELS}
    rows, lam, errs = [], {}, {}
    xt_test = torch.tensor(xr.load_dataset(ROOT / "tmp/data/l63v2_test_seed42.nc").x_true.values)
    pools = {m: climate_pool(steps[m], xt_test[:50, 0]) for m in MODELS}         # invariant-measure samples
    noise = measure_distances(climate_pool(steps["l1"], xt_test[50:100, 0]), pools["l1"])
    for split in ("val", "test"):
        ds = xr.load_dataset(ROOT / f"tmp/data/l63v2_{split}_seed42.nc")
        xt = torch.tensor(ds.x_true.values)
        x0t = xt[:, 0]
        g = torch.Generator().manual_seed(0)
        pts = {"truth": x0t, "pert": x0t + torch.randn(x0t.shape, generator=g)}
        oe = int(ds.attrs["obs_every"])
        for a, b, kind in EDGES:
            r = edge_metrics(steps[a], steps[b], ds, xt, pts, oe)
            samples = r.pop("_err_samples")
            if split == "test":
                errs[f"{a.upper()}–{b.upper()}"] = samples
                r.update(measure_distances(pools[a], pools[b]))
            rows.append({"split": split, "edge": f"{a.upper()}–{b.upper()}", "axis": kind, **r})
            print(split, rows[-1]["edge"], f"m {r['m_pooled']:.3f}", r["cos_gJo_truth"], flush=True)
        if split == "test":
            for m in MODELS:
                lv = lyapunov_exponent(steps[m], x0t[:50], 2000, 0.01).numpy()
                lam[m] = {"lambda1": ms(lv), "lyap_time_TU": float(1 / lv.mean()), "desc": desc[m]}
                if m in ("f1", "f2"):
                    lam[m].update({f"dtheta_{k}": v for k, v in theta_distance(m, reg).items()})
    df = pd.DataFrame(rows)
    out = ROOT / "tmp/results/models/rectangle"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".csv"), index=False)
    (out.with_suffix(".json")).write_text(json.dumps(lam, indent=1))
    pub = df[[c for c in df.columns if not c.startswith("_")]]
    lm = pd.DataFrame([{"model": m.upper(), **v} for m, v in lam.items()])
    md = ("## Edges (mean ± std over cases; m in units of σ_o = 1.41)\n\n" + to_markdown(pub.round(4))
          + "\n\nInvariant-measure distances on test only; noise floor (L1 vs L1, other ICs): "
          + " ".join(f"{k} {v:.3f}" for k, v in noise.items())
          + "\n\n## Models (test, 50 ICs)\n\n" + to_markdown(lm.round(4)) + "\n")
    out.with_suffix(".md").write_text(md)
    (ROOT / "tmp/figs").mkdir(parents=True, exist_ok=True)
    figure(df, reg, ROOT / "tmp/figs/rectangle")
    pdf_figure(pools, errs, ROOT / "tmp/figs/rectangle_pdfs")
    print(md)


if __name__ == "__main__":
    main()
