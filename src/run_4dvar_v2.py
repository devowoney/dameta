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
import time
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


def setup_problem(cfg: DictConfig, split: str, n_cases=None) -> dict:
    """D1 v2 data, lagged hybrid background and the (exact-gradient) cost of `split` — shared with the meta runner."""
    torch.set_default_dtype(torch.float64)
    assert cfg.problem in ("bice", "bpe")
    assert cfg.problem == "bice" or (cfg.model == "f1" and cfg.constraint == "strong"), "BPE: model f1, strong only"
    # feature flag `data_prefix` (2026-11-16): l63v2 = D1 v2 (obs every 10 steps); l63v2d = dense obs (every step,
    # src/make_dense_obs.py). The background (first guess, B) is the D1 v2 one in both cases.
    ds = xr.load_dataset(Path(cfg.data_dir) / f"{cfg.get('data_prefix', 'l63v2')}_{split}_seed{cfg.seed}.nc")
    x0_b, b_h, b_inv, ab = load_hybrid_background(cfg.data_dir, split, background_name(cfg), int(cfg.seed))
    if n_cases is not None:
        n = int(n_cases)
        ds, x0_b, b_h, b_inv = ds.isel(ic=slice(0, n)), x0_b[:n], b_h[:n], b_inv[:n]
    ccfg = CostConfig(window=int(ds.attrs["window"]), obs_every=int(ds.attrs["obs_every"]),
                      sigma_o2=float(cfg.cost.sigma_o2), sigma_q=float(cfg.cost.sigma_q))
    y = torch.tensor(ds.y_obs.values)
    mask = torch.tensor(ds.obs_mask.sel(pattern=cfg.pattern).values)
    weak = cfg.constraint == "weak"
    if cfg.problem == "bpe":
        theta_b = torch.tensor([cfg.f1.sigma, cfg.f1.rho, cfg.f1.beta])
        cost = ParamFourDVarCost(y, mask, x0_b, ccfg, theta_b, cfg.bpe_prior_rel_std * theta_b, ds.attrs["dt"], b_inv)
    else:
        cost = FourDVarCost(estimation_step(cfg, ds), y, mask, x0_b, ccfg, weak=weak, b_inv=b_inv)
    return {"ds": ds, "ccfg": ccfg, "x0_b": x0_b, "b_inv": b_inv, "ab": ab, "y": y, "mask": mask, "weak": weak,
            "cost": cost}


def subset_cost(cfg: DictConfig, P: dict, idx) -> FourDVarCost:
    """Cost of the cases idx of a set-up problem (training batches)."""
    if cfg.problem == "bpe":
        c = P["cost"]
        return ParamFourDVarCost(P["y"][idx], P["mask"], P["x0_b"][idx], P["ccfg"], c.theta_b, c.theta_std, c.dt,
                                 P["b_inv"][idx])
    return FourDVarCost(P["cost"].step, P["y"][idx], P["mask"], P["x0_b"][idx], P["ccfg"], weak=P["weak"],
                        b_inv=P["b_inv"][idx])


def evaluate_v2(cfg: DictConfig, P: dict, res: dict, configuration: str) -> xr.Dataset:
    """Truth-based D3-v2 record of an optimizer result (classical or meta; see module docstring)."""
    ds, ccfg, x0_b, b_inv, weak, base = P["ds"], P["ccfg"], P["x0_b"], P["b_inv"], P["weak"], P["cost"]
    step_t = truth_step(ds)
    x_true, L, F = ds.x_true.values, ccfg.window, int(cfg.forecast)
    truth_fc = x_true[:, L:L + F + 1]
    if "control_final" in res:
        c_fin = torch.tensor(res["control_final"])
    else:                                              # meta runners: rebuild the control from x0_a [+ eta_final]
        parts = [torch.tensor(res["x0_a"])] + ([torch.tensor(res["eta_final"]).flatten(1)] if weak else [])
        c_fin = torch.cat(parts, 1)
    finite = torch.isfinite(c_fin).all(1)
    x0_a = torch.where(finite[:, None], c_fin[:, :3], x0_b)
    eta = torch.nan_to_num(torch.tensor(res["eta_final"])) if weak else None
    if cfg.problem == "bpe":
        theta_a = torch.where(finite[:, None], c_fin[:, 3:6], base.theta_b.expand(len(x0_a), 3))
        step_est = lambda x: base.step_theta(x, theta_a)                       # noqa: E731
        step_b = L63Model(L63Params(sigma=cfg.f1.sigma, rho=cfg.f1.rho, beta=cfg.f1.beta, dt=ds.attrs["dt"])).step
    else:
        step_est = step_b = base.step
    with torch.no_grad():
        fc_a = forecast_from_analysis(step_est, x0_a, ccfg, F, eta).numpy()
        fc_m1 = forecast_from_analysis(step_t, x0_a, ccfg, F, eta).numpy()
        fc_b = forecast_from_analysis(step_b, x0_b, ccfg, F).numpy()
        c_true = FourDVarCost(step_t, P["y"], P["mask"], x0_b, ccfg, weak=weak, b_inv=b_inv)
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
        coords={"ic": ds.ic.values, "k": np.arange(res["x0_iter"].shape[1]),
                "kstep": np.arange(1, res["cos_step_grad"].shape[1] + 1), "lead": np.arange(F + 1),
                "var": ["x", "y", "z"]},
        attrs={"config": json.dumps(OmegaConf.to_container(cfg, resolve=True)), "wall_s": res["wall_s"],
               "dameta_version": __version__, "torch_version": torch.__version__, "dataset": "D1v2",
               "configuration": configuration, "background": background_name(cfg),
               "hybrid_alpha": P["ab"]["alpha"], "hybrid_s2": P["ab"]["s2"]},
    )
    if weak:
        out["eta_final"] = (("ic", "ctrl_t", "var"), res["eta_final"])
    if "grad_cum" in res:          # gradient evaluations (incl. line-search trials) spent to reach iterate k
        out["grad_cum"] = (("ic", "k"), res["grad_cum"])
    if cfg.problem == "bpe":
        out["theta_a"] = (("ic", "param"), theta_a.numpy())
        out = out.assign_coords(param=["sigma", "rho", "beta"])
    return out


class G1Recorder:
    """
    on_step hook of run_fourdvar: at every iterate c^k (cases still active)
        g1_grad[k] = cos(g^k, grad J_true(c^k))            gradient the optimizer consumes vs the true sensitivity
        g1_move[k] = cos(c^{k+1} - c^k, -grad J_true(c^k))  the move actually applied vs the true descent direction
    J_true = same problem (obs, first guess, hybrid B, Q) with the true model (L63 rho 28). Frozen cases: NaN.
    """

    def __init__(self, P: dict, cost):
        self.true_cost = FourDVarCost(truth_step(P["ds"]), P["y"], P["mask"], P["x0_b"], P["ccfg"], weak=P["weak"],
                                      b_inv=P["b_inv"])
        self.n = P["x0_b"].shape[0]
        self.g1_grad, self.g1_move = [], []
        self.prev = None                                   # (c^k, grad J_true(c^k), active) of the previous call
        self.time_s = 0.0                                  # own wall-clock, subtracted from the optimizer's wall_s

    def __call__(self, k, c, g, active):
        t0 = time.perf_counter()
        cos = torch.nn.functional.cosine_similarity
        _, g_true = self.true_cost.grad(c.detach())
        act = active.bool()
        nan = torch.full((self.n,), float("nan"), dtype=c.dtype)
        self.g1_grad.append(torch.where(act, cos(g, g_true, dim=1), nan))
        if self.prev is not None:                          # the move from the previous iterate is now known
            c_p, gt_p, act_p = self.prev
            self.g1_move.append(torch.where(act_p, cos(c.detach() - c_p, -gt_p, dim=1), nan))
        self.prev = (c.detach().clone(), g_true, act)
        self.time_s += time.perf_counter() - t0

    def arrays(self, max_iter: int):
        """(g1_grad [ic, k], g1_move [ic, kstep]) as DataArray tuples padded with NaN to the record length."""
        def pad(lst, length):
            a = torch.stack(lst, 1).numpy() if lst else np.zeros((self.n, 0))
            return np.concatenate([a, np.full((self.n, length - a.shape[1]), np.nan)], 1)
        return (("ic", "k"), pad(self.g1_grad, max_iter + 1)), (("ic", "kstep"), pad(self.g1_move, max_iter))


def run(cfg: DictConfig) -> xr.Dataset:
    P = setup_problem(cfg, cfg.split, cfg.n_cases)
    base, cost = P["cost"], P["cost"]
    if cfg.gradient != "exact":
        # adjoint-free baselines (cell-8 claim): F1 forward runs only for J
        assert cfg.problem == "bice" and cfg.model == "f1", "fd / hybrid gradients: model f1, problem bice"
        if cfg.gradient == "fd":
            cost = FDGradient(base, float(cfg.fd_eps))
        elif cfg.gradient == "hybrid":
            emu = L63Emulator.load(Path(cfg.emulator_dir) / f"{cfg.surrogate_emulator}.pt")
            for p in emu.parameters():
                p.requires_grad_(False)
            cost = HybridGradient(base, FourDVarCost(emu.step, P["y"], P["mask"], P["x0_b"], P["ccfg"],
                                                     weak=P["weak"], b_inv=P["b_inv"]))
        else:
            raise ValueError(cfg.gradient)
    # ICE table (2026-11-19, flag g1_iter): per-iterate gradient fidelity w.r.t. the TRUE dynamics, recorded by a
    # read-only hook (its true-model gradients are reporting only: not counted in n_grad_evals / grad_cum).
    hook, g1 = (None, None)
    if cfg.get("g1_iter", False):
        g1 = G1Recorder(P, base)
        hook = g1
    res = run_fourdvar(cost, P["x0_b"], cfg.method, float(cfg.lr), int(cfg.max_iter), float(cfg.tol), on_step=hook)
    conf = f"{cfg.problem}:{cfg.model}" + (f":{cfg.emulator}" if cfg.model == "emulator" else "")
    out = evaluate_v2(cfg, P, res, conf)
    if g1 is not None:
        out["g1_grad"], out["g1_move"] = g1.arrays(int(cfg.max_iter))
        out.attrs["wall_s"] = float(out.attrs["wall_s"]) - g1.time_s     # optimizer time only
        out.attrs["g1_hook_s"] = g1.time_s
    if cfg.gradient != "exact":
        out.attrs["gradient"] = cfg.gradient
        out.attrs["n_fwd_model_total"] = int(cost.n_fwd_model)      # forward runs of F1 (all cases, batched)
        out.attrs["fwd_per_gradient"] = int(cost.n_fwd_model // max(cost.n_grad_evals, 1))
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
