"""
Hydra entry point: classical 4D-Var on D1 v2 (M5 blueprint) — model-error-isolated BICE / BPE.

    model=truth    E1 reference (true model inside 4D-Var; SAME first guess and B as F1)
    model=f1       knowledge numerical model (L63 with rho 27.5)            problem=bice | bpe
    model=emulator false model F2 (emulator, default rho275)                problem=bice
First guess = 0.5-TU member of the time-lagged ensemble of the estimation model's false model; B = hybrid
(alpha B_ens + (1 - alpha) s2 I, tmp/data/hybrid_B_seed<S>.json). Window / obs geometry from the D1 v2 file.

Usage (project root, env oceanai):
    python src/run_4dvar_v2.py model=f1 method=adagrad lr=0.1
    python src/run_4dvar_v2.py model=f1 problem=bpe
    python src/run_4dvar_v2.py model=emulator split=test

Record (NetCDF, dims ic, k, kstep, lead, var[, ctrl_t, param]):
    x0_iter, J_iter, cos_step_grad, n_grad_evals, converged, diverged, x0_a, [eta_final], [theta_a (bpe)],
    rmse_x0, rmse_x0_b (first guess), J_true_final (P1: cost of the analysis with the TRUE model, same obs and B),
    rmse_fc     forecast from the analysis with the ESTIMATION model (bpe: with the estimated parameters) — operational
    rmse_fc_m1  forecast from the analysis with the TRUE model — initial-condition quality
    rmse_fc_b   forecast from the first guess with the estimation model
"""

import json
import logging
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
import xarray as xr
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dameta import __version__  # noqa: E402
from dameta.da import CostConfig, FourDVarCost, ParamFourDVarCost, forecast_from_analysis, run_fourdvar  # noqa: E402
from dameta.da.gradients import FDGradient, HybridGradient  # noqa: E402
from dameta.data.background import load_hybrid_background  # noqa: E402
from dameta.dynamics import L63Model, L63Params  # noqa: E402
from dameta.dynamics.emulator import L63Emulator  # noqa: E402

OmegaConf.register_new_resolver("project_root", lambda: str(Path(__file__).resolve().parent.parent), replace=True)
OmegaConf.register_new_resolver("bpe_suffix", lambda p: "_bpe" if p == "bpe" else "", replace=True)
OmegaConf.register_new_resolver("grad_suffix", lambda g: "" if g == "exact" else f"_{g}", replace=True)
log = logging.getLogger(__name__)


def truth_step(ds):
    return L63Model(L63Params(sigma=ds.attrs["sigma"], rho=ds.attrs["rho"], beta=ds.attrs["beta"],
                              dt=ds.attrs["dt"])).step


def estimation_step(cfg, ds):
    if cfg.model == "truth":
        return truth_step(ds)
    if cfg.model == "f1":
        return L63Model(L63Params(sigma=cfg.f1.sigma, rho=cfg.f1.rho, beta=cfg.f1.beta, dt=ds.attrs["dt"])).step
    if cfg.model == "emulator":
        emu = L63Emulator.load(Path(cfg.emulator_dir) / f"{cfg.emulator}.pt")
        for p in emu.parameters():
            p.requires_grad_(False)
        assert abs(emu.cfg.dt - ds.attrs["dt"]) < 1e-12
        return emu.step
    raise ValueError(cfg.model)


def background_name(cfg):
    if cfg.background:
        return cfg.background
    return cfg.emulator if cfg.model == "emulator" else "f1"


def rmse(a, b):
    return np.sqrt(((a - b) ** 2).mean(-1))


def run(cfg: DictConfig) -> xr.Dataset:
    torch.set_default_dtype(torch.float64)
    assert cfg.problem in ("bice", "bpe")
    assert cfg.problem == "bice" or (cfg.model == "f1" and cfg.constraint == "strong"), "BPE: model f1, strong only"
    ds = xr.load_dataset(Path(cfg.data_dir) / f"l63v2_{cfg.split}_seed{cfg.seed}.nc")
    x0_b, b_h, b_inv, ab = load_hybrid_background(cfg.data_dir, cfg.split, background_name(cfg), int(cfg.seed))
    if cfg.n_cases is not None:
        n = int(cfg.n_cases)
        ds, x0_b, b_h, b_inv = ds.isel(ic=slice(0, n)), x0_b[:n], b_h[:n], b_inv[:n]
    ccfg = CostConfig(window=int(ds.attrs["window"]), obs_every=int(ds.attrs["obs_every"]),
                      sigma_o2=float(cfg.cost.sigma_o2), sigma_q=float(cfg.cost.sigma_q))
    y = torch.tensor(ds.y_obs.values)
    mask = torch.tensor(ds.obs_mask.sel(pattern=cfg.pattern).values)
    weak = cfg.constraint == "weak"
    step_t = truth_step(ds)
    if cfg.problem == "bpe":
        theta_b = torch.tensor([cfg.f1.sigma, cfg.f1.rho, cfg.f1.beta])
        cost = ParamFourDVarCost(y, mask, x0_b, ccfg, theta_b, cfg.bpe_prior_rel_std * theta_b, ds.attrs["dt"], b_inv)
    else:
        step = estimation_step(cfg, ds)
        cost = FourDVarCost(step, y, mask, x0_b, ccfg, weak=weak, b_inv=b_inv)
    base = cost
    if cfg.gradient != "exact":
        # adjoint-free baselines (cell-8 claim): F1 forward runs only for J
        assert cfg.problem == "bice" and cfg.model == "f1", "fd / hybrid gradients: model f1, problem bice"
        if cfg.gradient == "fd":
            cost = FDGradient(base, float(cfg.fd_eps))
        elif cfg.gradient == "hybrid":
            emu = L63Emulator.load(Path(cfg.emulator_dir) / f"{cfg.surrogate_emulator}.pt")
            for p in emu.parameters():
                p.requires_grad_(False)
            cost = HybridGradient(base, FourDVarCost(emu.step, y, mask, x0_b, ccfg, weak=weak, b_inv=b_inv))
        else:
            raise ValueError(cfg.gradient)

    res = run_fourdvar(cost, x0_b, cfg.method, float(cfg.lr), int(cfg.max_iter), float(cfg.tol))

    # ---- evaluation against the truth ----------------------------------------------------------------------------
    x_true, L, F = ds.x_true.values, ccfg.window, int(cfg.forecast)
    truth_fc = x_true[:, L:L + F + 1]
    c_fin = torch.tensor(res["control_final"])
    finite = torch.isfinite(c_fin).all(1)
    x0_a = torch.where(finite[:, None], c_fin[:, :3], x0_b)
    eta = torch.nan_to_num(torch.tensor(res["eta_final"])) if weak else None
    if cfg.problem == "bpe":
        theta_a = torch.where(finite[:, None], c_fin[:, 3:6], cost.theta_b.expand(len(x0_a), 3))
        step_est = lambda x: cost.step_theta(x, theta_a)                       # noqa: E731
        step_b = L63Model(L63Params(sigma=cfg.f1.sigma, rho=cfg.f1.rho, beta=cfg.f1.beta, dt=ds.attrs["dt"])).step
    else:
        step_est = step_b = base.step
    with torch.no_grad():
        fc_a = forecast_from_analysis(step_est, x0_a, ccfg, F, eta).numpy()
        fc_m1 = forecast_from_analysis(step_t, x0_a, ccfg, F, eta).numpy()
        fc_b = forecast_from_analysis(step_b, x0_b, ccfg, F).numpy()
        c_true = FourDVarCost(step_t, y, mask, x0_b, ccfg, weak=weak, b_inv=b_inv)
        j_true = c_true(*c_true.unpack(c_true.pack(x0_a, eta)))
    fin = finite.numpy()
    nanif = lambda a: np.where(fin.reshape((-1,) + (1,) * (a.ndim - 1)), a, np.nan)   # noqa: E731
    out = xr.Dataset(
        data_vars={
            "x0_iter": (("ic", "k", "var"), res["x0_iter"]), "J_iter": (("ic", "k"), res["J_iter"]),
            "cos_step_grad": (("ic", "kstep"), res["cos_step_grad"]), "n_grad_evals": (("ic",), res["n_grad_evals"]),
            "converged": (("ic",), res["converged"]), "diverged": (("ic",), ~fin),
            "x0_a": (("ic", "var"), res["x0_a"]),
            "rmse_x0": (("ic",), nanif(rmse(x0_a.numpy(), x_true[:, 0]))),
            "rmse_x0_b": (("ic",), rmse(x0_b.numpy(), x_true[:, 0])),
            "J_true_final": (("ic",), nanif(j_true.numpy())),
            "rmse_fc": (("ic", "lead"), nanif(rmse(fc_a, truth_fc))),
            "rmse_fc_m1": (("ic", "lead"), nanif(rmse(fc_m1, truth_fc))),
            "rmse_fc_b": (("ic", "lead"), rmse(fc_b, truth_fc)),
        },
        coords={"ic": ds.ic.values, "k": np.arange(res["x0_iter"].shape[1]), "kstep": np.arange(1, cfg.max_iter + 1),
                "lead": np.arange(F + 1), "var": ["x", "y", "z"]},
        attrs={"config": json.dumps(OmegaConf.to_container(cfg, resolve=True)), "wall_s": res["wall_s"],
               "dameta_version": __version__, "torch_version": torch.__version__, "dataset": "D1v2",
               "configuration": f"{cfg.problem}:{cfg.model}" + (f":{cfg.emulator}" if cfg.model == "emulator" else ""),
               "background": background_name(cfg), "hybrid_alpha": ab["alpha"], "hybrid_s2": ab["s2"]},
    )
    if weak:
        out["eta_final"] = (("ic", "ctrl_t", "var"), res["eta_final"])
    if cfg.gradient != "exact":
        out.attrs["gradient"] = cfg.gradient
        out.attrs["n_fwd_model_total"] = int(cost.n_fwd_model)      # forward runs of F1 (all cases, batched)
        out.attrs["fwd_per_gradient"] = int(cost.n_fwd_model // max(cost.n_grad_evals, 1))
    if cfg.problem == "bpe":
        out["theta_a"] = (("ic", "param"), theta_a.numpy())
        out = out.assign_coords(param=["sigma", "rho", "beta"])
    return out


@hydra.main(version_base=None, config_path="configs", config_name="fourdvar_v2")
def main(cfg: DictConfig) -> None:
    out = run(cfg)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{cfg.exp_name}.nc"
    out.to_netcdf(path)
    extra = ""
    if "theta_a" in out:
        extra = " | median theta_a " + ", ".join(f"{float(v):.3f}" for v in np.nanmedian(out.theta_a.values, 0))
    log.info("%s %s | diverged %d/%d | converged %d | median evals %.0f | median J %.2f | rmse_x0 %.3f (fg %.3f) | "
             "fc@1TU model %.3f / truth %.3f%s | wall %.1fs -> %s", out.attrs["configuration"], cfg.exp_name,
             int(out.diverged.sum()), out.sizes["ic"], int(out.converged.sum()), float(np.median(out.n_grad_evals)),
             float(np.nanmedian(out.J_iter[:, -1])), float(np.nanmedian(out.rmse_x0)),
             float(np.median(out.rmse_x0_b)), float(np.nanmedian(out.rmse_fc.sel(lead=100))),
             float(np.nanmedian(out.rmse_fc_m1.sel(lead=100))), extra, out.attrs["wall_s"], path)


if __name__ == "__main__":
    main()
