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
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dameta.da import CostConfig, FourDVarCost  # noqa: E402
from dameta.meta import (MetaB, MetaBConfig, MetaBSup, StretchNet, SupConfig, build_sup,  # noqa: E402
                         collect_teacher_pairs, control_stats, run_meta_b, run_meta_b_sup, train_meta_b)
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


def train_learner(cfg: DictConfig, step, d: int, ctrl_mean, ctrl_std):
    """Meta-train the learner of cfg.variant on train ICs. Returns (learner, info dict)."""
    tr = load_split(cfg, "train", cfg.train.n_train)
    make_tr = cost_factory(cfg, tr, step)
    xb_tr = torch.tensor(tr.x0_b.values)
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
            if writer is not None:
                i = h["epoch"] * 10_000 + h["batch"]
                for key in ("J_end_med", "L_perf_applied_mean", "step_ratio_med"):
                    writer.add_scalar(f"train/{key}", h[key], i)

        hist = train_meta_b(learner, lambda idx: make_tr(idx), xb_tr, cfg.method, float(cfg.lr),
                            n_epochs=cfg.delta.n_epochs, batch=cfg.delta.batch, episode_iters=cfg.delta.episode_iters,
                            seed=cfg.delta.train_seed, log=log_ep)
        info = {"train_history": hist, "train_grad_evals": len(xb_tr) * cfg.delta.episode_iters * cfg.delta.n_epochs}
    elif cfg.variant == "supervised":
        pairs = collect_teacher_pairs(make_tr(), xb_tr, cfg.method, float(cfg.lr), int(cfg.max_iter), float(cfg.tol))
        va = load_split(cfg, "val", cfg.train.n_val_pairs_ic)
        vpairs = collect_teacher_pairs(cost_factory(cfg, va, step)(), torch.tensor(va.x0_b.values), cfg.method,
                                       float(cfg.lr), int(cfg.max_iter), float(cfg.tol))
        scfg = SupConfig(**OmegaConf.to_container(cfg.supervised))
        learner = build_sup(d, scfg, ctrl_mean, ctrl_std, pairs)
        log.info("supervised pairs: train %d, val %d", len(pairs["k"]), len(vpairs["k"]))
        hist = learner.fit(pairs, vpairs, log=lambda h: log.info("epoch %s", {k: round(v, 5) if isinstance(v, float)
                                                                               else v for k, v in h.items()}))
        info = {"train_history": hist, "train_grad_evals": int(len(pairs["k"])), "n_pairs_val": int(len(vpairs["k"]))}
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
    ev = load_split(cfg, cfg.split, cfg.n_cases)
    step = build_step(cfg, ev)
    ccfg, weak = CostConfig(**cfg.cost), cfg.constraint == "weak"
    d = 3 + (3 * (ccfg.n_obs - 1) if weak else 0)

    if cfg.checkpoint:
        sd = torch.load(cfg.checkpoint, weights_only=False)
        learner = (MetaB if cfg.variant == "delta" else MetaBSup).from_state_dict(sd)
        info = {"loaded_from": str(cfg.checkpoint)}
    else:
        x_tr = torch.tensor(load_split(cfg, "train", cfg.train.n_train).x_true.values)
        cm, cs = control_stats(x_tr, ccfg.n_obs, ccfg.sigma_q, weak)
        learner, info = train_learner(cfg, step, d, cm, cs)
        Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)
        torch.save(learner.state_dict(), Path(cfg.ckpt_dir) / f"{cfg.exp_name}.pt")

    cost = cost_factory(cfg, ev, step)()
    xb = torch.tensor(ev.x0_b.values)
    if cfg.variant == "delta":
        res = run_meta_b(cost, xb, learner, cfg.method, float(cfg.lr), int(cfg.max_iter), float(cfg.tol))
    else:
        res = run_meta_b_sup(cost, xb, learner, cfg.method, float(cfg.lr), int(cfg.max_iter), float(cfg.tol),
                             diag_true_grad=bool(cfg.diag_true_grad))
    tag = ("E2-B" if cfg.dynamics == "l63" else f"E4-B:{cfg.emulator}") + f":{cfg.variant}"
    out = analysis_record(cfg, ev, step, cost, res, tag)
    for key in ("step_ratio", "cos_ghat_grad"):
        if key in res:
            out[key] = (("ic", "kstep"), res[key])
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
    cost_var = "n_grad_evals" if cfg.variant == "delta" else "n_fwd_evals"
    log.info("%s | diverged %d/%d | converged %d | median %s %.0f | median J_final %.2f | median rmse_x0 %.3f "
             "(bg %.3f) | median rmse_fc@lead100 %.3f | wall %.1fs -> %s",
             cfg.exp_name, int(out.diverged.sum()), out.sizes["ic"], int(out.converged.sum()), cost_var,
             float(np.median(out[cost_var])), float(np.nanmedian(out.J_iter[:, -1])),
             float(np.nanmedian(out.rmse_x0)), float(np.median(out.rmse_x0_b)),
             float(np.nanmedian(out.rmse_fc.sel(lead=100))), out.attrs["wall_s"], path)


if __name__ == "__main__":
    main()
