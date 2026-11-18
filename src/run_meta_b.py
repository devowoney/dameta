"""
Hydra entry point: Meta-B (E2-B with M1, E4-B with an emulator). Meta-trains on TRAIN ICs (or loads `checkpoint`),
saves the learner to ${ckpt_dir}/${exp_name}.pt, then runs FROZEN inference on `split` and writes a D3 record
(same variables as run_4dvar.py) to ${out_dir}/${exp_name}_${split}.nc.

Usage (project root, env oceanai):
    python src/run_meta_b.py                                             # delta, strong, M1, Adagrad teacher, val
    python src/run_meta_b.py variant=supervised
    python src/run_meta_b.py -m delta.w_perf_delta=1,2.5,5 tag=_sweep    # hyper-parameter sweep on val
    python src/run_meta_b.py split=test checkpoint=tmp/meta/e2b/<exp>.pt # frozen test with a selected learner

Extra record variables: step_ratio (delta), n_fwd_evals and cos_ghat_grad (supervised).
Cost: delta = n_grad_evals (1 gradient per outer iteration, the chain has no model call);
      supervised = n_grad_evals 0, n_fwd_evals forward runs (J for the stopping rule).
Meta-training cost is reported in attrs (train_wall_s, train_grad_evals), not in the per-case analysis cost.
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
from omegaconf import DictConfig, OmegaConf, open_dict

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dameta.da import CostConfig, FourDVarCost  # noqa: E402
from dameta.meta import (LearnedOptimization, MetaB, MetaBConfig, MetaBSup, StretchNet, SupConfig,  # noqa: E402
                         UnrollConfig, build_sup, build_unroll, collect_teacher_pairs, control_stats, obs_features,
                         run_meta_a, run_meta_b, run_meta_b_sup, train_meta_b)
from dameta.meta.meta_b_lookahead import LookaheadConfig, collect_lookahead_pairs, fit_lookahead  # noqa: E402
from dameta.meta.meta_greedy import GreedyConfig, GreedyOptimizer, build_greedy, run_greedy  # noqa: E402
from dameta.meta.learned_gradient import (LearnedGradient, LGConfig, build_lg,  # noqa: E402
                                          collect_pairs as collect_lg_pairs,
                                          collect_sampled_pairs as collect_lg_sampled, run_learned_gradient)
from dameta.dynamics.emulator import L63Emulator  # noqa: E402
from run_4dvar import analysis_record, build_step  # noqa: E402  (also registers the project_root resolver)

log = logging.getLogger(__name__)


def load_split(cfg: DictConfig, split: str, n=None) -> xr.Dataset:
    ds = xr.open_dataset(Path(cfg.data_dir) / f"l63_{split}_seed{cfg.seed}.nc")
    return ds if n is None else ds.isel(ic=slice(0, int(n)))


def cost_factory(cfg: DictConfig, ds: xr.Dataset, step):
    ccfg = CostConfig(**cfg.cost)
    assert ccfg.window == ds.attrs["window"] and ccfg.obs_every == ds.attrs["obs_every"], "cost/data geometry mismatch"
    y, xb = torch.tensor(ds.y_obs.values), torch.tensor(ds.x0_b.values)
    mask = torch.tensor(ds.obs_mask.sel(pattern=cfg.pattern).values)
    weak = cfg.constraint == "weak"
    return lambda idx=slice(None): FourDVarCost(step, y[idx], mask, xb[idx], ccfg, weak=weak)


class Problem:
    """Data access for one split: v1 (D1, truth + N(0, I) background) or v2 (D1 v2, lagged false-model first guess,
    hybrid B, false models f1 / emulator, truth reference; run_4dvar_v2.setup_problem)."""

    def __init__(self, cfg: DictConfig, split: str, n=None):
        self.cfg, self.v2 = cfg, cfg.get("dataset", "v1") == "v2"
        if self.v2:
            import run_4dvar_v2 as R2
            self.R2 = R2
            self.P = R2.setup_problem(cfg, split, n)
            self.ds, self.ccfg, self.x0_b = self.P["ds"], self.P["ccfg"], self.P["x0_b"]
            self.step = self.P["cost"].step
        else:
            self.ds = load_split(cfg, split, n)
            self.step = build_step(cfg, self.ds)
            self.ccfg = CostConfig(**cfg.cost)
            self._f = cost_factory(cfg, self.ds, self.step)
            self.x0_b = torch.tensor(self.ds.x0_b.values)
        self.x_true = torch.tensor(self.ds.x_true.values)

    def cost(self, idx=slice(None)):
        return self.R2.subset_cost(self.cfg, self.P, idx) if self.v2 else self._f(idx)

    def cost_with(self, step, idx=slice(None)):
        """Same data / background / B with another model (cell-8 surrogate)."""
        c = self.cost(idx)
        return FourDVarCost(step, c.y, c.mask, c.x0_b, c.cfg, weak=c.weak, b_inv=c.b_inv)


def problem(cfg: DictConfig, split: str, n=None) -> Problem:
    return Problem(cfg, split, n)


def train_learner(cfg: DictConfig, step, d: int, ctrl_mean, ctrl_std):
    """Meta-train the learner of cfg.variant on train ICs. Returns (learner, info dict)."""
    ptr = problem(cfg, "train", cfg.train.n_train)
    make_tr, xb_tr = ptr.cost, ptr.x0_b
    t0 = time.perf_counter()
    if cfg.variant == "delta":
        torch.manual_seed(cfg.delta.train_seed)
        net = StretchNet(d, **cfg.net)
        dk = OmegaConf.to_container(cfg.delta)
        mcfg = MetaBConfig(**{k: dk[k] for k in MetaBConfig.__dataclass_fields__})
        learner = MetaB(net, mcfg, ctrl_mean, ctrl_std)
        writer = _writer(cfg)

        def log_ep(h):
            log.info("episode %s", {k: round(v, 4) if isinstance(v, float) else v for k, v in h.items()})
            if writer is not None and "val_score" in h:
                writer.add_scalar("train/val_score", h["val_score"], h["epoch"])
            elif writer is not None:
                i = h["epoch"] * 10_000 + h["batch"]
                for key in ("J_end_med", "L_perf_applied_mean", "step_ratio_med"):
                    writer.add_scalar(f"train/{key}", h[key], i)

        val_score = None
        if cfg.delta.select_epoch:
            # Best-epoch selection on VAL ICs: median J after select_k frozen outer iterations (lower = faster)
            va = problem(cfg, "val", cfg.delta.select_n_val)
            cost_va, xb_va = va.cost(), va.x0_b

            def val_score(lrn):
                r = run_meta_b(cost_va, xb_va, lrn, cfg.method, float(cfg.lr), max_iter=int(cfg.delta.select_k),
                               tol=0.0)
                return float(np.nanmedian(r["J_iter"][:, -1]))

        hist = train_meta_b(learner, lambda idx: make_tr(idx), xb_tr, cfg.method, float(cfg.lr),
                            n_epochs=cfg.delta.n_epochs, batch=cfg.delta.batch, episode_iters=cfg.delta.episode_iters,
                            seed=cfg.delta.train_seed, log=log_ep, val_score=val_score)
        info = {"train_history": hist, "train_grad_evals": len(xb_tr) * cfg.delta.episode_iters * cfg.delta.n_epochs}
    elif cfg.variant == "supervised":
        scfg = SupConfig(**OmegaConf.to_container(cfg.supervised))
        va = problem(cfg, "val", cfg.train.n_val_pairs_ic)
        cost_tr, cost_va = make_tr(), va.cost()
        ob = (lambda c: obs_features(c, ctrl_mean, ctrl_std)) if scfg.obs_input else (lambda c: None)  # noqa: E731
        pairs = collect_teacher_pairs(cost_tr, xb_tr, cfg.method, float(cfg.lr), int(cfg.max_iter), float(cfg.tol),
                                      obs_feats=ob(cost_tr))
        vpairs = collect_teacher_pairs(cost_va, va.x0_b, cfg.method, float(cfg.lr),
                                       int(cfg.max_iter), float(cfg.tol), obs_feats=ob(cost_va))
        learner = build_sup(d, scfg, ctrl_mean, ctrl_std, pairs, n_obs=ptr.ccfg.n_obs)
        log.info("supervised pairs: train %d, val %d", len(pairs["k"]), len(vpairs["k"]))
        hist = learner.fit(pairs, vpairs, log=lambda h: log.info("epoch %s", {k: round(v, 5) if isinstance(v, float)
                                                                               else v for k, v in h.items()}))
        info = {"train_history": hist, "train_grad_evals": int(len(pairs["k"])), "n_pairs_val": int(len(vpairs["k"]))}
    elif cfg.variant == "lookahead":
        # M5 cell 2: Meta-B chain trained (supervised) to the teacher iterate m steps ahead
        torch.manual_seed(cfg.lookahead.seed)
        dk = OmegaConf.to_container(cfg.delta)
        learner = MetaB(StretchNet(d, **cfg.net), MetaBConfig(**{k: dk[k] for k in MetaBConfig.__dataclass_fields__}),
                        ctrl_mean, ctrl_std)
        lcfg = LookaheadConfig(**OmegaConf.to_container(cfg.lookahead))
        va = problem(cfg, "val", cfg.train.n_val_pairs_ic)
        pairs = collect_lookahead_pairs(make_tr(), xb_tr, cfg.method, float(cfg.lr), lcfg.m, int(cfg.max_iter),
                                        float(cfg.tol))
        vpairs = collect_lookahead_pairs(va.cost(), va.x0_b, cfg.method,
                                         float(cfg.lr), lcfg.m, int(cfg.max_iter), float(cfg.tol))
        log.info("lookahead pairs: train %d, val %d", len(pairs["k"]), len(vpairs["k"]))
        hist = fit_lookahead(learner, pairs, vpairs, lcfg, log=lambda h: log.info("epoch %s", {
            k: round(v, 5) if isinstance(v, float) else v for k, v in h.items()}))
        info = {"train_history": hist, "train_grad_evals": int(len(pairs["k"]))}
    elif cfg.variant == "greedy":
        # M5 cell 3: free update without gradient input, online greedy J(x^{k+1}) / J(x^k)
        gk = OmegaConf.to_container(cfg.greedy)
        gcfg = GreedyConfig(**{k: gk[k] for k in GreedyConfig.__dataclass_fields__})
        n_obs = ptr.ccfg.n_obs
        learner = build_greedy(d, gcfg, ctrl_mean, ctrl_std, n_obs, hidden=gk["hidden"], n_layers=gk["n_layers"],
                               output_scale=gk["output_scale"])
        pack = lambda xb: torch.cat([xb, torch.zeros(xb.shape[0], 3 * (n_obs - 1))], 1) \
            if cfg.constraint == "weak" else xb  # noqa: E731
        va = problem(cfg, "val", gk["n_val"])
        hist = learner.fit(lambda idx: make_tr(idx), pack(xb_tr), va.cost(),
                           pack(va.x0_b), log=lambda h: log.info("episode %s", {
                               k: round(v, 5) if isinstance(v, float) else v for k, v in h.items()}))
        info = {"train_history": hist, "train_grad_evals": 0}
    elif cfg.variant == "unroll":
        # Learned optimization (M5 cells 5 / 7): K unrolled free-update steps, end-to-end on J(x^K)
        uk = OmegaConf.to_container(cfg.unroll)
        ucfg = UnrollConfig(**{k: uk[k] for k in UnrollConfig.__dataclass_fields__})
        n_obs = ptr.ccfg.n_obs
        learner = build_unroll(d, ucfg, ctrl_mean, ctrl_std, n_obs, hidden=uk["hidden"], n_layers=uk["n_layers"],
                               output_scale=uk["output_scale"])
        weak = cfg.constraint == "weak"
        pack = lambda xb: torch.cat([xb, torch.zeros(xb.shape[0], 3 * (n_obs - 1))], 1) if weak else xb  # noqa: E731
        va = problem(cfg, "val", uk["n_val"])
        hist = learner.fit(lambda idx: make_tr(idx), pack(xb_tr), va.cost(),
                           pack(va.x0_b),
                           log=lambda h: log.info("epoch %s", {k: round(v, 5) if isinstance(v, float) else v
                                                               for k, v in h.items()}))
        info = {"train_history": hist, "train_grad_evals": (len(xb_tr) * ucfg.K * ucfg.epochs) if ucfg.use_grad else 0}
    elif cfg.variant == "learned_grad":
        # M5 cell 8: adjoint-free learned gradient (innovations from a forward run -> grad J_o), supervised on the
        # emulator's autodiff grad J_o along the adjoint-free hybrid classical trajectory
        lk = OmegaConf.to_container(cfg.learned_grad)
        emu = L63Emulator.load(Path(cfg.emulator_dir) / f"{lk['surrogate_emulator']}.pt")
        for p in emu.parameters():
            p.requires_grad_(False)
        va = problem(cfg, "val", lk["n_val_ic"])
        lcfg = LGConfig(**{k: lk[k] for k in LGConfig.__dataclass_fields__})
        if lcfg.data == "sampled":                 # decorrelated B-shaped controls (+ thinned hybrid trajectory)
            mk = lambda pb, xb: collect_lg_sampled(pb.cost(), pb.cost_with(emu.step), xb, cfg.method,  # noqa: E731
                                                   float(cfg.lr), lcfg, int(cfg.max_iter), float(cfg.tol))
        else:                                      # states of the adjoint-free hybrid classical run
            mk = lambda pb, xb: collect_lg_pairs(pb.cost(), pb.cost_with(emu.step), xb, cfg.method,  # noqa: E731
                                                 float(cfg.lr), int(cfg.max_iter), float(cfg.tol))
        pairs, vpairs = mk(ptr, xb_tr), mk(va, va.x0_b)
        learner = build_lg(d, ptr.ccfg.n_obs, lcfg, ctrl_mean, ctrl_std, pairs)
        log.info("learned-gradient (%s, %s) pairs: train %d, val %d", lcfg.arch, lcfg.data, len(pairs["k"]),
                 len(vpairs["k"]))
        hist = learner.fit(pairs, vpairs, weak=cfg.constraint == "weak", log=lambda h: log.info("epoch %s", {
            k: round(v, 5) if isinstance(v, float) else v for k, v in h.items()}))
        info = {"train_history": hist, "train_pairs": int(len(pairs["k"])), "train_grad_evals": 0}
    else:
        raise ValueError(f"unknown variant {cfg.variant!r}")
    info["train_wall_s"] = time.perf_counter() - t0
    return learner, info


def _writer(cfg):
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(str(Path(cfg.tb_dir) / cfg.exp_name))
    except Exception:                                   # tensorboard optional
        return None


def run(cfg: DictConfig) -> xr.Dataset:
    torch.set_default_dtype(torch.float64)
    ev = problem(cfg, cfg.split, cfg.n_cases)
    step = ev.step
    ccfg, weak = ev.ccfg, cfg.constraint == "weak"
    d = 3 + (3 * (ccfg.n_obs - 1) if weak else 0)

    if cfg.checkpoint:
        sd = torch.load(cfg.checkpoint, weights_only=False)
        learner = {"delta": MetaB, "lookahead": MetaB, "supervised": MetaBSup, "greedy": GreedyOptimizer,
                   "unroll": LearnedOptimization, "learned_grad": LearnedGradient}[cfg.variant].from_state_dict(sd)
        info = {"loaded_from": str(cfg.checkpoint)}
    else:
        x_tr = problem(cfg, "train", cfg.train.n_train).x_true
        cm, cs = control_stats(x_tr, ccfg.n_obs, ccfg.sigma_q, weak)
        learner, info = train_learner(cfg, step, d, cm, cs)
        Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)
        torch.save(learner.state_dict(), Path(cfg.ckpt_dir) / f"{cfg.exp_name}.pt")

    cost = ev.cost()
    xb = ev.x0_b
    if cfg.variant in ("delta", "lookahead"):
        res = run_meta_b(cost, xb, learner, cfg.method, float(cfg.lr), int(cfg.max_iter), float(cfg.tol))
    elif cfg.variant == "greedy":
        res = run_greedy(cost, xb, learner, int(cfg.max_iter), float(cfg.tol))
    elif cfg.variant == "unroll":
        res = run_meta_a(cost, xb, learner)                      # fixed budget K, no stopping rule
        with open_dict(cfg):
            cfg.max_iter = int(learner.cfg.K)                    # record step axis = K
    elif cfg.variant == "learned_grad":
        res = run_learned_gradient(cost, xb, learner, cfg.method, float(cfg.lr), int(cfg.max_iter), float(cfg.tol),
                                   diag_exact=bool(cfg.diag_true_grad))
    else:
        res = run_meta_b_sup(cost, xb, learner, cfg.method, float(cfg.lr), int(cfg.max_iter), float(cfg.tol),
                             diag_true_grad=bool(cfg.diag_true_grad))
    if ev.v2:
        tag = f"v2:{cfg.problem}:{cfg.model}" + (f":{cfg.emulator}" if cfg.model == "emulator" else "") \
            + f":{cfg.variant}"
        out = ev.R2.evaluate_v2(cfg, ev.P, res, tag)
    else:
        tag = ("E2-B" if cfg.dynamics == "l63" else f"E4-B:{cfg.emulator}") + f":{cfg.variant}"
        out = analysis_record(cfg, ev.ds, step, cost, res, tag)
    for key in ("step_ratio", "cos_ghat_grad"):
        if key in res:
            out[key] = (("ic", "kstep"), res[key])
    if "grad_cum" in res and cfg.method == "lbfgs_wolfe":   # cost incl. line-search gradients (strong Wolfe teacher)
        out["grad_cum"] = (("ic", "k"), res["grad_cum"])
    if "n_fwd_evals" in res:
        out["n_fwd_evals"] = (("ic",), res["n_fwd_evals"])
    out.attrs["train_info"] = json.dumps({k: v for k, v in info.items()}, default=float)
    return out


@hydra.main(version_base=None, config_path="configs", config_name="meta_b")
def main(cfg: DictConfig) -> None:
    out = run(cfg)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{cfg.exp_name}_{cfg.split}.nc"
    out.to_netcdf(path)
    cost_var = "n_fwd_evals" if cfg.variant in ("supervised", "greedy", "learned_grad") else "n_grad_evals"
    log.info("%s | diverged %d/%d | converged %d | median %s %.0f | median J_final %.2f | median rmse_x0 %.3f "
             "(bg %.3f) | median rmse_fc@lead100 %.3f | wall %.1fs -> %s",
             cfg.exp_name, int(out.diverged.sum()), out.sizes["ic"], int(out.converged.sum()), cost_var,
             float(np.median(out[cost_var])), float(np.nanmedian(out.J_iter[:, -1])),
             float(np.nanmedian(out.rmse_x0)), float(np.median(out.rmse_x0_b)),
             float(np.nanmedian(out.rmse_fc.sel(lead=100))), out.attrs["wall_s"], path)


if __name__ == "__main__":
    main()
