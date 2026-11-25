"""
Hydra entry point: classical 4D-Var on D1 v2 (M5 blueprint) — model-error-isolated BICE / BPE.

Estimation model = one of the model rectangle (src/configs/models.yaml, branch m7-model-rectangle, 2026-11-20):
    model=l1 (alias truth)   true L63 inside 4D-Var (perfect-model reference; first guess / B of `l1_background`)
    model=f1 | f2            false L63, theta_L * (1 + eps d) (mild | strong)    problem=bice | bpe
    model=e1 | e2            emulator trained on F1 | F2 trajectories            problem=bice
    model=emulator           legacy v2 path: emulator file `emulator` (e.g. rho275)
First guess = 0.5-TU member of the time-lagged ensemble of the estimation model itself; B = hybrid
(alpha B_ens + (1 - alpha) s2 I, tmp/data/hybrid_B_seed<S>.json). Window / obs geometry from the D1 v2 file.

Usage (project root, env oceanai):
    python src/run_4dvar_v2.py model=f1 method=adagrad lr=0.1
    python src/run_4dvar_v2.py model=f2 problem=bpe
    python src/run_4dvar_v2.py model=e1 split=test

Record (NetCDF, dims ic, k, kstep, lead, var[, ctrl_t, param]):
    x0_iter, J_iter, cos_step_grad, n_grad_evals, converged, diverged, x0_a, [eta_final], [theta_a (bpe)],
    rmse_x0, rmse_x0_b (first guess), J_true_final (P1: cost of the analysis with the TRUE model, same obs and B),
    rmse_fc     forecast from the analysis with the ESTIMATION model (bpe: with the estimated parameters) — operational
    rmse_fc_m1  forecast from the analysis with the TRUE model — initial-condition quality
    rmse_fc_b   forecast from the first guess with the estimation model
    sol_deg     |x0_a - x_true(0)|_2                                                    (M8, 2026-11-25)
    traj_deg    1/K sum_k |x_a^k - x_true^k|_2, k = 0 .. L + F (window + forecast), x_a^k from the ESTIMATION model
With landscape_iter=true (M8): cos_truth, lam_min, lam_max, kappa_H, n_neg per iterate [ic, k] (every hess_every
iterations) and at the analysis (*_a [ic], eig_a [ic, d]); see src/dameta/eval/landscape.py.
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
from dameta.da.cost import window_trajectory  # noqa: E402
from dameta.da.gradients import FDGradient, HybridGradient  # noqa: E402
from dameta.data.background import load_hybrid_background  # noqa: E402
from dameta.dynamics import L63Model, L63Params  # noqa: E402
from dameta.dynamics.emulator import L63Emulator  # noqa: E402
from dameta.dynamics.models import build_model, canonical, load_registry, model_params  # noqa: E402
from dameta.eval.landscape import cos_truth, hessian_batch, spectrum  # noqa: E402

OmegaConf.register_new_resolver("project_root", lambda: str(Path(__file__).resolve().parent.parent), replace=True)
OmegaConf.register_new_resolver("bpe_suffix", lambda p: "_bpe" if p == "bpe" else "", replace=True)
OmegaConf.register_new_resolver("grad_suffix", lambda g: "" if g == "exact" else f"_{g}", replace=True)
log = logging.getLogger(__name__)


def truth_step(ds):
    return L63Model(L63Params(sigma=ds.attrs["sigma"], rho=ds.attrs["rho"], beta=ds.attrs["beta"],
                              dt=ds.attrs["dt"])).step


def registry(cfg):
    """Model-rectangle registry (models.yaml) with optional top-level overrides, e.g. registry_overrides.eps1=0.01."""
    ov = cfg.get("registry_overrides", None)
    return load_registry(**(OmegaConf.to_container(ov) if ov else {}))


def model_kind(cfg) -> str:
    if cfg.model == "emulator":
        return "emulator"
    return str(registry(cfg).models[canonical(cfg.model, registry(cfg))].kind)


def estimation_step(cfg, ds):
    if cfg.model != "emulator":
        reg = registry(cfg)
        step, desc, p = build_model(cfg.model, reg, cfg.emulator_dir)
        if canonical(cfg.model, reg) == "l1":                      # registry L1 must be the data's truth model
            assert (p.sigma, p.rho, p.beta, p.dt) == (ds.attrs["sigma"], ds.attrs["rho"], ds.attrs["beta"],
                                                      ds.attrs["dt"]), "registry L1 != data truth"
        log.info("estimation model %s", desc)
        return step
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
    if cfg.model == "emulator":
        return cfg.emulator
    name = canonical(cfg.model, registry(cfg))
    return cfg.get("l1_background", "f1") if name == "l1" else name


def rmse(a, b):
    return np.sqrt(((a - b) ** 2).mean(-1))


def setup_problem(cfg: DictConfig, split: str, n_cases=None) -> dict:
    """D1 v2 data, lagged hybrid background and the (exact-gradient) cost of `split` — shared with the meta runner."""
    torch.set_default_dtype(torch.float64)
    assert cfg.problem in ("bice", "bpe")
    assert cfg.problem == "bice" or (model_kind(cfg) == "l63" and cfg.constraint == "strong"), \
        "BPE: L63-kind model (l1 / f1 / f2), strong constraint only"
    # feature flag `data_prefix` (2026-11-16): l63v2 = D1 v2 (obs every 10 steps); l63v2d = dense obs (every step,
    # src/make_dense_obs.py). The background (first guess, B) is the D1 v2 one in both cases.
    ds = xr.load_dataset(Path(cfg.data_dir) / f"{cfg.get('data_prefix', 'l63v2')}_{split}_seed{cfg.seed}.nc")
    x0_b, b_h, b_inv, ab = load_hybrid_background(cfg.data_dir, split, background_name(cfg), int(cfg.seed))
    # weaker background term (pilot 2026-11-25, flag cost.b_infl): J_b uses b_infl * B_hybrid, so that J_b convexifies
    # J less and the model-dependent J_o shapes the geometry (1.0 = tuned hybrid B, earlier behaviour)
    infl = float(cfg.cost.get("b_infl", 1.0))
    b_h, b_inv = infl * b_h, b_inv / infl
    ab = {**ab, "b_infl": infl}
    if n_cases is not None:
        n = int(n_cases)
        ds, x0_b, b_h, b_inv = ds.isel(ic=slice(0, n)), x0_b[:n], b_h[:n], b_inv[:n]
    ccfg = CostConfig(window=int(ds.attrs["window"]), obs_every=int(ds.attrs["obs_every"]),
                      sigma_o2=float(ds.attrs["sigma_o2"] if cfg.cost.sigma_o2 is None else cfg.cost.sigma_o2),
                      sigma_q=float(cfg.cost.sigma_q))
    y = torch.tensor(ds.y_obs.values)
    mask = torch.tensor(ds.obs_mask.sel(pattern=cfg.pattern).values)
    weak = cfg.constraint == "weak"
    if cfg.problem == "bpe":
        p = model_params(cfg.model, registry(cfg))                 # parameter prior = the model's own theta
        theta_b = torch.tensor([p.sigma, p.rho, p.beta])
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
        step_b = L63Model(model_params(cfg.model, registry(cfg))).step         # prior theta = the model's own
    else:
        step_est = step_b = base.step
    with torch.no_grad():
        fc_a = forecast_from_analysis(step_est, x0_a, ccfg, F, eta).numpy()
        win_a = window_trajectory(step_est, x0_a, ccfg, eta).numpy()                 # [n, L + 1, 3]
        fc_m1 = forecast_from_analysis(step_t, x0_a, ccfg, F, eta).numpy()
        fc_b = forecast_from_analysis(step_b, x0_b, ccfg, F).numpy()
        c_true = FourDVarCost(step_t, P["y"], P["mask"], x0_b, ccfg, weak=weak, b_inv=b_inv)
        j_true = c_true(*c_true.unpack(c_true.pack(x0_a, eta)))
    fin = finite.numpy()
    traj_a = np.concatenate([win_a[:, :L], fc_a], 1)                                # t = 0 .. L + F
    traj_deg = np.linalg.norm(traj_a - x_true[:, :L + F + 1], axis=-1).mean(1)
    nanif = lambda a: np.where(fin.reshape((-1,) + (1,) * (a.ndim - 1)), a, np.nan)   # noqa: E731
    out = xr.Dataset(
        data_vars={
            "x0_iter": (("ic", "k", "var"), res["x0_iter"]), "J_iter": (("ic", "k"), res["J_iter"]),
            "cos_step_grad": (("ic", "kstep"), res["cos_step_grad"]), "n_grad_evals": (("ic",), res["n_grad_evals"]),
            "converged": (("ic",), res["converged"]), "diverged": (("ic",), ~fin),
            "x0_a": (("ic", "var"), res["x0_a"]),
            "rmse_x0": (("ic",), nanif(rmse(x0_a.numpy(), x_true[:, 0]))),
            "rmse_x0_b": (("ic",), rmse(x0_b.numpy(), x_true[:, 0])),
            "sol_deg": (("ic",), nanif(np.linalg.norm(x0_a.numpy() - x_true[:, 0], axis=-1))),
            "traj_deg": (("ic",), nanif(traj_deg)),
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
               "hybrid_alpha": P["ab"]["alpha"], "hybrid_s2": P["ab"]["s2"], "b_infl": P["ab"]["b_infl"],
               "sigma_o2": ccfg.sigma_o2, "obs_every": ccfg.obs_every},
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


class LandscapeRecorder:
    """
    on_step hook of run_fourdvar (M8): at iterate c^k of the ESTIMATION-model problem (cases still active)
        cos_truth[k] = cos(x_true(0) - x0^k, -grad_x0 J(c^k))   (gradient already computed by the optimizer)
        lam_min, lam_max, kappa_H, n_neg [k] = spectrum of the Hessian of J w.r.t. the control (every hess_every)
    Frozen / skipped iterates: NaN. Its own time is excluded from the optimizer wall_s (reporting only).
    """

    KEYS = ("lam_min", "lam_max", "kappa_H", "n_neg")

    def __init__(self, P: dict, cost, hess_every: int = 1):
        self.cost, self.every = cost, max(int(hess_every), 1)
        self.x0_true = torch.tensor(P["ds"].x_true.values[:, 0])
        self.n = self.x0_true.shape[0]
        self.rec = {k: [] for k in ("cos_truth",) + self.KEYS}
        self.time_s = 0.0

    def __call__(self, k, c, g, active):
        t0 = time.perf_counter()
        act = active.bool()
        nan = torch.full((self.n,), float("nan"), dtype=c.dtype)
        self.rec["cos_truth"].append(torch.where(act, cos_truth(c[:, :3].detach(), g[:, :3], self.x0_true), nan))
        if k % self.every == 0:
            sp = spectrum(hessian_batch(self.cost, c))
            for key in self.KEYS:
                self.rec[key].append(torch.where(act, sp[key], nan))
        else:
            for key in self.KEYS:
                self.rec[key].append(nan)
        self.time_s += time.perf_counter() - t0

    def arrays(self, max_iter: int) -> dict:
        def pad(lst):
            a = torch.stack(lst, 1).numpy() if lst else np.zeros((self.n, 0))
            return (("ic", "k"), np.concatenate([a, np.full((self.n, max_iter + 1 - a.shape[1]), np.nan)], 1))
        return {key: pad(v) for key, v in self.rec.items()}

    def at_analysis(self, c_a: torch.Tensor) -> dict:
        """Same diagnostics at the final control (all cases; reporting only)."""
        t0 = time.perf_counter()
        _, g = self.cost.grad(c_a)
        self.cost.n_grad_evals -= 1                                  # reporting call: not an optimizer evaluation
        sp = spectrum(hessian_batch(self.cost, c_a))
        out = {"cos_truth_a": (("ic",), cos_truth(c_a[:, :3], g[:, :3], self.x0_true).numpy()),
               "eig_a": (("ic", "eig"), sp["eig"].numpy())}
        out.update({f"{key}_a": (("ic",), sp[key].numpy()) for key in self.KEYS})
        self.time_s += time.perf_counter() - t0
        return out


def run(cfg: DictConfig) -> xr.Dataset:
    P = setup_problem(cfg, cfg.split, cfg.n_cases)
    base, cost = P["cost"], P["cost"]
    if cfg.gradient != "exact":
        # adjoint-free baselines (cell-8 claim): F1 forward runs only for J
        assert cfg.problem == "bice" and model_kind(cfg) == "l63", "fd / hybrid gradients: L63-kind model, bice"
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
    # M8 (2026-11-25, flag landscape_iter): truth-correction cosine + Hessian spectrum per iterate (exact cost only)
    lands = None
    if cfg.get("landscape_iter", False):
        assert hook is None and cfg.gradient == "exact", "landscape_iter: exact gradient, no other hook"
        lands = LandscapeRecorder(P, base, int(cfg.get("hess_every", 1)))
        hook = lands
    res = run_fourdvar(cost, P["x0_b"], cfg.method, float(cfg.lr), int(cfg.max_iter), float(cfg.tol), on_step=hook)
    conf = f"{cfg.problem}:{cfg.model}" + (f":{cfg.emulator}" if cfg.model == "emulator" else "")
    out = evaluate_v2(cfg, P, res, conf)
    if lands is not None:                                          # M8 landscape diagnostics
        for key, val in lands.arrays(int(cfg.max_iter)).items():
            out[key] = val
        out.attrs["wall_s"] = float(out.attrs["wall_s"]) - lands.time_s    # in-loop hook time only
        for key, val in lands.at_analysis(torch.tensor(res["control_final"])).items():
            out[key] = val
        out.attrs["landscape_hook_s"] = lands.time_s
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
