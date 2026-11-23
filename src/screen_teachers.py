"""
Screen candidate false models (biased L63 parameters) against the true model L1 on VAL ICs.

Usage (project root, on a compute node via sbatch):
    python src/screen_teachers.py                 # legacy candidate list (M3 v2 screen) on D1 val
    python src/screen_teachers.py --sweep         # model-rectangle screen: eps sweep along d (src/configs/models.yaml)
Writes tmp/results/emulator/teacher_screen.{csv,md} (legacy) or tmp/results/models/false_screen.{csv,md} + figure.

Metrics per candidate (integrated with RK4 at the same dt as L1, from TRUE states along the truth trajectories):
  obs_int_err_{x,y,z}  pooled std of the one-observation-interval model error (10 steps from true states)
  m_pooled             RMS over (x, y, z) of obs_int_err_* divided by sigma_o  -> pre-registered pick rule (sweep)
  m_case               per-case RMS one-interval error / sigma_o, mean +- std over cases
  rmse_0.5TU / 1TU     free-forecast RMSE from the true x0, mean +- std over cases
  vpt                  valid prediction time [TU]: first lead where the error exceeds 0.4 x climatological std
  lambda1              leading Lyapunov exponent (renormalised TL propagation, 20 TU, 50 ICs), mean +- std
  stable_20TU          all 20-TU free runs finite and bounded (|x| < 100); z_mean / z_std of the last 10 TU
Probabilistic view (pilot 2026-11-20: inspect the models as probability distributions, not only point errors):
  err_q05/q50/q95      quantiles of the one-obs-interval error norm |M_F^10(x) - M_L^10(x)| / (sqrt(3) sigma_o) over
                       all (case, interval) samples (q50 vs m_pooled shows how heavy the tail is)
  W1_x/y/z, energy     invariant measure vs L1: 50 val ICs x 40 TU after 10 TU burn-in, pooled; 1-D Wasserstein per
                       variable and 3-D energy distance (4000-point subsamples). `noise` = L1 vs L1 from 50 OTHER val ICs
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dameta.dynamics.emulator import TeacherConfig, teacher_rollout, teacher_tendency  # noqa: E402
from dameta.dynamics.l63 import rk4_step  # noqa: E402
from dameta.dynamics.models import load_registry, perturbed_params, theta_distance  # noqa: E402,F401
from dameta.eval.emulator_metrics import lyapunov_exponent  # noqa: E402
from report_e1 import to_markdown  # noqa: E402

torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parent.parent
SIGMA_O = float(np.sqrt(2.0))
LEGACY = {
    "clean (truth)": {},
    "xz095 (current main)": {"xz_scale": 0.95},
    "rho 27.5": {"rho": 27.5}, "rho 27": {"rho": 27.0}, "rho 28.5": {"rho": 28.5}, "rho 29": {"rho": 29.0},
    "sigma 9.5": {"sigma": 9.5}, "sigma 10.5": {"sigma": 10.5}, "sigma 11": {"sigma": 11.0},
    "beta 2.5": {"beta": 2.5}, "beta 2.8": {"beta": 2.8},
}
EPS_GRID_PCT = [0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10,       # pre-registered sweep (pilot plan 2026-11-20)
                12, 15, 18, 20, 22, 25]                         # edge extension (F2 hit 10 %; pilot 2026-11-20)
F1_TARGET, F2_TARGET = 0.15, 1.0                                # m_pooled targets (x sigma_o)


def ms(a) -> str:
    a = np.asarray(a, dtype=float)
    return f"{np.nanmean(a):.3f} ± {np.nanstd(a):.3f}"


@torch.no_grad()
def climate_pool(step, x0, n_burn=1000, n_keep=4000, every=5) -> torch.Tensor:
    """Pooled states of free runs (invariant-measure sample): [n_ic * n_keep / every, 3]."""
    x, out = x0, []
    for i in range(n_burn + n_keep):
        x = step(x)
        if i >= n_burn and (i - n_burn) % every == 0:
            out.append(x)
    return torch.cat(out, 0)


def energy_distance(a: torch.Tensor, b: torch.Tensor, n: int = 4000, seed: int = 0) -> float:
    """3-D energy distance 2E|X-Y| - E|X-X'| - E|Y-Y'| on random subsamples of n points."""
    g = torch.Generator().manual_seed(seed)
    a = a[torch.randperm(len(a), generator=g)[:n]]
    b = b[torch.randperm(len(b), generator=g)[:n]]
    return float(2 * torch.cdist(a, b).mean() - torch.cdist(a, a).mean() - torch.cdist(b, b).mean())


def measure_distances(pool, ref) -> dict:
    from scipy.stats import wasserstein_distance
    d = {f"W1_{v}": float(wasserstein_distance(pool[:, i].numpy(), ref[:, i].numpy())) for i, v in enumerate("xyz")}
    d["energy"] = energy_distance(pool, ref)
    return d


def screen(candidates: dict, data_file: Path, obs_every: int = 10) -> pd.DataFrame:
    """candidates: name -> TeacherConfig kwargs. Returns one row per candidate (numeric + formatted columns)."""
    ds = xr.open_dataset(data_file)
    xt = torch.tensor(ds.x_true.values)                        # [n, T + 1, 3] truth (L1)
    n_t = xt.shape[1] - 1
    clim = float(xt.reshape(-1, 3).std(0).mean())
    thr = 0.4 * clim
    n_int = min(50, (n_t - obs_every) // obs_every)
    starts = xt[:, :n_int * obs_every:obs_every]               # [n, n_int, 3] true states every obs interval
    ends = xt[:, obs_every:(n_int + 1) * obs_every:obs_every]
    rows = []
    l1 = lambda v: rk4_step(lambda u: teacher_tendency(u, TeacherConfig()), v, 0.01)  # noqa: E731
    ref = climate_pool(l1, xt[:50, 0])
    noise = measure_distances(climate_pool(l1, xt[50:100, 0]), ref)
    for name, kw in candidates.items():
        t = TeacherConfig(**kw)
        step = lambda v, t=t: rk4_step(lambda u: teacher_tendency(u, t), v, t.dt)  # noqa: E731
        with torch.no_grad():
            x = starts.reshape(-1, 3)
            for _ in range(obs_every):
                x = step(x)
        err = (x.reshape(starts.shape) - ends)                  # [n, n_int, 3]
        e = err.reshape(-1, 3).std(0).numpy()
        m_case = (err ** 2).mean((1, 2)).sqrt().numpy() / SIGMA_O
        traj = teacher_rollout(xt[:, 0], 200, t)
        fe = ((traj - xt[:, :201]) ** 2).mean(-1).sqrt()       # [n, 201]
        hit = (fe > thr).numpy()
        vpt = np.where(hit.any(1), hit.argmax(1), 200) * t.dt
        lam = lyapunov_exponent(step, xt[:50, 0], 2000, t.dt).numpy()
        long = teacher_rollout(xt[:50, 0], 2000, t)             # 20 TU free runs
        stable = bool(torch.isfinite(long).all() and long.abs().max() < 100)
        z = long[:, 1000:, 2]
        en = (err.norm(dim=-1) / (np.sqrt(3) * SIGMA_O)).reshape(-1)
        qs = torch.quantile(en, torch.tensor([0.05, 0.5, 0.95])).numpy()
        dist = measure_distances(climate_pool(step, xt[:50, 0]), ref) if stable else {}
        rows.append({"model": name, "sigma": t.sigma, "rho": t.rho, "beta": t.beta,
                     "obs_int_err_x": e[0], "obs_int_err_y": e[1], "obs_int_err_z": e[2],
                     "m_pooled": float(np.sqrt((e ** 2).mean())) / SIGMA_O,
                     "m_case_mean": float(m_case.mean()), "m_case_std": float(m_case.std()),
                     "rmse_0.5TU": ms(fe[:, 50].numpy()), "rmse_1TU": ms(fe[:, 100].numpy()),
                     "vpt_TU": ms(vpt), "lambda1_mean": float(lam.mean()), "lambda1_std": float(lam.std()),
                     "stable_20TU": stable, "z_mean": float(z.mean()), "z_std": float(z.std()),
                     "err_q05": float(qs[0]), "err_q50": float(qs[1]), "err_q95": float(qs[2]), **dist})
    df = pd.DataFrame(rows)
    df.attrs.update(n_cases=int(xt.shape[0]), vpt_thr=thr, data=str(data_file),
                    noise=" ".join(f"{k} {v:.3f}" for k, v in noise.items()), noise_energy=noise["energy"])
    return df


def sweep_candidates(reg) -> dict:
    cands = {"L1 (eps 0)": {}}
    for pct in EPS_GRID_PCT:
        p = perturbed_params(reg.theta_L, reg.direction, pct / 100, reg.dt)
        cands[f"eps {pct:g}%"] = {"sigma": p.sigma, "rho": p.rho, "beta": p.beta}
    cands["ref: rho 27.5 (v2 F1)"] = {"rho": 27.5}
    return cands


def pick(df: pd.DataFrame, target: float) -> str:
    """Pre-registered rule: grid eps with m_pooled closest to target among stable, chaotic candidates."""
    ok = df[df.model.str.startswith("eps") & df.stable_20TU & (df.lambda1_mean > 0)]
    return str(ok.iloc[(ok.m_pooled - target).abs().argmin()].model)


def figure(df: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sw = df[df.model.str.startswith("eps") | df.model.str.startswith("L1")].copy()
    sw["eps"] = [0.0] + [float(s.split()[1].rstrip("%")) for s in sw.model[1:]]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    ax[0].errorbar(sw.eps, sw.m_case_mean, yerr=sw.m_case_std, fmt="o-", label="per case (mean ± std)", ms=4)
    ax[0].plot(sw.eps, sw.m_pooled, "s--", label="pooled (pick rule)", ms=4)
    for y, lab in ((F1_TARGET, "F1 target"), (F2_TARGET, "F2 target")):
        ax[0].axhline(y, color="grey", lw=0.8, ls=":")
        ax[0].text(sw.eps.max(), y, lab, ha="right", va="bottom", fontsize=8)
    ax[0].set(xlabel="ε [%] along d = (1,1,1)/√3 (relative)", ylabel="one-obs-interval model error / σ_o")
    ax[0].legend(fontsize=8)
    ax[1].errorbar(sw.eps, sw.lambda1_mean, yerr=sw.lambda1_std, fmt="o-", ms=4)
    ax[1].set(xlabel="ε [%]", ylabel="λ₁ [1/TU]")
    ax[0].fill_between(sw.eps, sw.err_q05 * np.sqrt(3), sw.err_q95 * np.sqrt(3), alpha=0.2,
                       label="5-95 % of samples")
    ax[0].legend(fontsize=8)
    if "energy" in sw:
        ax[2].plot(sw.eps, sw.energy, "o-", ms=4, label="energy distance (3-D)")
        for v in "xyz":
            ax[2].plot(sw.eps, sw[f"W1_{v}"], ".--", lw=0.8, label=f"W1 {v}")
        ax[2].axhline(df.attrs["noise_energy"], color="grey", ls=":", lw=0.8)
        ax[2].text(0, df.attrs["noise_energy"], "L1 vs L1 noise", fontsize=7, va="bottom")
        ax[2].set(xlabel="ε [%]", ylabel="invariant-measure distance to L1")
        ax[2].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="model-rectangle eps sweep along d (models.yaml)")
    a = ap.parse_args()
    if not a.sweep:                                            # legacy M3 v2 screen (unchanged behaviour)
        df = screen(LEGACY, ROOT / "tmp/data/l63_val_seed42.nc")
        out = ROOT / "tmp/results/emulator/teacher_screen"
        extra = ""
    else:
        reg = load_registry()
        df = screen(sweep_candidates(reg), ROOT / "tmp/data/l63v2_val_seed42.nc")
        out = ROOT / "tmp/results/models/false_screen"
        d = np.array(list(reg.direction), dtype=float)
        df.insert(4, "dtheta_abs", np.sqrt(((df[["sigma", "rho", "beta"]].values
                                             - np.array([10.0, 28.0, 8 / 3])) ** 2).sum(1)))
        f1, f2 = pick(df, F1_TARGET), pick(df, F2_TARGET)
        extra = (f"\n\nDirection d = {d / np.linalg.norm(d)} (relative). Pre-registered pick (m_pooled closest to "
                 f"{F1_TARGET} / {F2_TARGET} x sigma_o, stable and lambda1 > 0): **F1 = {f1}**, **F2 = {f2}**.\n")
        (ROOT / "tmp/figs").mkdir(parents=True, exist_ok=True)
        figure(df, ROOT / "tmp/figs/false_screen")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".csv"), index=False)
    md = (to_markdown(df.round(4)) + f"\n\n{df.attrs['n_cases']} val cases; VPT threshold 0.4 x clim std = "
          f"{df.attrs['vpt_thr']:.2f}; observation error std sigma_o = {SIGMA_O:.2f}; data {df.attrs['data']}."
          f"\nInvariant-measure noise floor (L1 vs L1, other ICs): {df.attrs['noise']}." + extra)
    out.with_suffix(".md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
