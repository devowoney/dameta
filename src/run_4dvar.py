"""
Hydra entry point: classical 4D-Var (E1) on one dataset split; writes one D3 analysis record to tmp/results/.

Usage (project root, env oceanai):
    python src/run_4dvar.py                                        # defaults in src/configs/fourdvar.yaml
    python src/run_4dvar.py -m method=gd lr=0.003,0.01,0.03        # Hydra multirun sweep (lr tuning on val)

D3 record (NetCDF, float64), dims ic, k (iterations 0..K), kstep (1..K), lead (0..F), var, ctrl_t (weak only):
    x0_iter, J_iter, cos_step_grad, n_grad_evals, converged, x0_a, rmse_x0 (analysis), rmse_x0_b (background),
    rmse_fc (forecast from analysis), rmse_fc_b (forecast from background), [eta_final]; attrs = full config.
Diverged cases (non-finite analysis) keep NaN metrics: they are counted and reported, never silently dropped.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make `dameta` importable without installation

from dameta import __version__  # noqa: E402
from dameta.da import CostConfig, FourDVarCost, forecast_from_analysis, run_fourdvar  # noqa: E402
from dameta.dynamics import L63Model, L63Params  # noqa: E402
from dameta.dynamics.emulator import L63Emulator  # noqa: E402

OmegaConf.register_new_resolver("project_root", lambda: str(Path(__file__).resolve().parent.parent), replace=True)
log = logging.getLogger(__name__)


def m1_step(ds: xr.Dataset):
    """Physical model M1 with the dataset's own parameters (truth generator)."""
    p = L63Params(sigma=ds.attrs["sigma"], rho=ds.attrs["rho"], beta=ds.attrs["beta"], dt=ds.attrs["dt"])
    return L63Model(p).step


def build_step(cfg: DictConfig, ds: xr.Dataset):
    """Dynamics used inside 4D-Var: M1 (E1) or the frozen emulator M2 tmp/emulators/<emulator>.pt (E3)."""
    if cfg.dynamics == "l63":
        return m1_step(ds)
    if cfg.dynamics == "emulator":
        emu = L63Emulator.load(Path(cfg.emulator_dir) / f"{cfg.emulator}.pt")
        for p in emu.parameters():
            p.requires_grad_(False)                       # frozen emulator: gradients w.r.t. the control only
        assert abs(emu.cfg.dt - ds.attrs["dt"]) < 1e-12, "emulator/data time step mismatch"
        return emu.step
    raise ValueError(f"unknown dynamics {cfg.dynamics!r}")


def rmse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """RMSE over the variable axis (last); keeps all other axes."""
    return np.sqrt(((a - b) ** 2).mean(-1))


def run(cfg: DictConfig) -> xr.Dataset:
    """Run one classical 4D-Var configuration and return its D3 record (also used by tests and tuning)."""
    torch.set_default_dtype(torch.float64)
    ds = xr.open_dataset(Path(cfg.data_dir) / f"l63_{cfg.split}_seed{cfg.seed}.nc")
    if cfg.n_cases is not None:
        ds = ds.isel(ic=slice(0, int(cfg.n_cases)))
    ccfg = CostConfig(**cfg.cost)
    assert ccfg.window == ds.attrs["window"] and ccfg.obs_every == ds.attrs["obs_every"], "cost/data geometry mismatch"

    step = build_step(cfg, ds)
    y, x0_b = torch.tensor(ds.y_obs.values), torch.tensor(ds.x0_b.values)
    mask = torch.tensor(ds.obs_mask.sel(pattern=cfg.pattern).values)
    cost = FourDVarCost(step, y, mask, x0_b, ccfg, weak=(cfg.constraint == "weak"))

    res = run_fourdvar(cost, x0_b, cfg.method, float(cfg.lr), int(cfg.max_iter), float(cfg.tol))
    return analysis_record(cfg, ds, step, cost, res, "E1" if cfg.dynamics == "l63" else f"E3:{cfg.emulator}")


def analysis_record(cfg: DictConfig, ds: xr.Dataset, step, cost: FourDVarCost, res: dict, configuration: str):
    """D3 record from an optimizer result (shared by E1/E3 classical and E2/E4 meta runs, M4)."""
    ccfg, y, x0_b = cost.cfg, cost.y, cost.x0_b
    mask = cost.mask.bool()
    # Forecast skill beyond the window (truth at t = L .. L + F), from the analysis and from the background.
    x_true = ds.x_true.values
    L, F = ccfg.window, int(cfg.forecast)
    truth_fc = x_true[:, L:L + F + 1]
    finite = np.isfinite(res["x0_a"]).all(-1)
    if "eta_final" in res:
        finite &= np.isfinite(res["eta_final"]).all((-1, -2))
    fin_t = torch.tensor(finite)
    x0_safe = torch.where(fin_t[:, None], torch.tensor(res["x0_a"]), x0_b)          # placeholder for diverged cases
    eta_safe = torch.nan_to_num(torch.tensor(res["eta_final"])) if "eta_final" in res else None
    fc_a = forecast_from_analysis(step, x0_safe, ccfg, F, eta_safe).numpy()
    fc_b = forecast_from_analysis(step, x0_b, ccfg, F).numpy()
    rmse_fc = np.where(finite[:, None], rmse(fc_a, truth_fc), np.nan)               # diverged -> NaN, reported
    rmse_x0 = np.where(finite, rmse(np.nan_to_num(res["x0_a"]), x_true[:, 0]), np.nan)

    out = xr.Dataset(
        data_vars={
            "x0_iter": (("ic", "k", "var"), res["x0_iter"]),
            "J_iter": (("ic", "k"), res["J_iter"]),
            "cos_step_grad": (("ic", "kstep"), res["cos_step_grad"]),
            "n_grad_evals": (("ic",), res["n_grad_evals"]),
            "converged": (("ic",), res["converged"]),
            "diverged": (("ic",), ~finite),
            "x0_a": (("ic", "var"), res["x0_a"]),
            "rmse_x0": (("ic",), rmse_x0),
            "rmse_x0_b": (("ic",), rmse(ds.x0_b.values, x_true[:, 0])),
            "rmse_fc": (("ic", "lead"), rmse_fc),
            "rmse_fc_b": (("ic", "lead"), rmse(fc_b, truth_fc)),
        },
        coords={"ic": ds.ic.values, "k": np.arange(res["x0_iter"].shape[1]), "kstep": np.arange(1, cfg.max_iter + 1),
                "lead": np.arange(F + 1), "var": ["x", "y", "z"]},
        attrs={"config": json.dumps(OmegaConf.to_container(cfg, resolve=True)), "wall_s": res["wall_s"],
               "dameta_version": __version__, "configuration": configuration,
               "torch_version": torch.__version__},
    )
    if "eta_final" in res:
        out["eta_final"] = (("ic", "ctrl_t", "var"), res["eta_final"])
    if cfg.dynamics != "l63":
        # E3: judge the emulator-based analysis with the TRUE model M1.
        #   J_true_final (P1): 4D-Var cost of the analysis evaluated with M1 (same obs, B, R[, Q])
        #   rmse_fc_m1: forecast with M1 from the analysis (IC quality); rmse_fc above = forecast with the emulator
        step_true = m1_step(ds)
        with torch.no_grad():
            c_true = FourDVarCost(step_true, y, mask, x0_b, ccfg, weak=cost.weak)
            j_true = c_true(*c_true.unpack(cost.pack(x0_safe, eta_safe)))
        fc_m1 = forecast_from_analysis(step_true, x0_safe, ccfg, F, eta_safe).numpy()
        out["J_true_final"] = (("ic",), np.where(finite, j_true.numpy(), np.nan))
        out["rmse_fc_m1"] = (("ic", "lead"), np.where(finite[:, None], rmse(fc_m1, truth_fc), np.nan))
    return out


@hydra.main(version_base=None, config_path="configs", config_name="fourdvar")
def main(cfg: DictConfig) -> None:
    out = run(cfg)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{cfg.exp_name}.nc"
    out.to_netcdf(path)
    log.info("%s | diverged %d/%d | converged %d | median evals %.0f | median J_final %.2f | median rmse_x0 %.3f "
             "(bg %.3f) | median rmse_fc@lead100 %.3f | wall %.1fs -> %s",
             cfg.exp_name, int(out.diverged.sum()), out.sizes["ic"], int(out.converged.sum()),
             float(np.median(out.n_grad_evals)), float(np.nanmedian(out.J_iter[:, -1])),
             float(np.nanmedian(out.rmse_x0)), float(np.median(out.rmse_x0_b)),
             float(np.nanmedian(out.rmse_fc.sel(lead=100))), out.attrs["wall_s"], path)


if __name__ == "__main__":
    main()
