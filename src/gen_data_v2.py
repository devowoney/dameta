"""
Hydra entry point: D1 v2 truth + observations, then the time-lagged background of every false model.

Usage (project root, env oceanai):
    python src/gen_data_v2.py                                  # all splits, backgrounds for f1 and rho275
    python src/gen_data_v2.py n_ic.train=50 'background.false_models={f1:{kind:l63,sigma:10,rho:27.5,beta:2.6667}}'
    python src/gen_data_v2.py backgrounds_only=true            # model rectangle: reuse truth/obs files, write the
                                                               # backgrounds of background.registry_models
Outputs: tmp/data/l63v2_<split>_seed<S>.nc and tmp/data/l63v2_<split>_seed<S>_bg_<name>.nc
"""

import logging
import sys
from pathlib import Path

import hydra
import numpy as np
import xarray as xr
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dameta.data.background import lagged_background  # noqa: E402
from dameta.data.generate_v2 import DataConfigV2, generate_split_v2  # noqa: E402
from dameta.dynamics import L63Model, L63Params  # noqa: E402
from dameta.dynamics.emulator import L63Emulator  # noqa: E402
from dameta.dynamics.models import build_model, load_registry  # noqa: E402

OmegaConf.register_new_resolver("project_root", lambda: str(Path(__file__).resolve().parent.parent), replace=True)
log = logging.getLogger(__name__)


def false_step(spec: DictConfig, dt: float, emulator_dir: str):
    if spec.kind == "l63":
        return L63Model(L63Params(sigma=spec.sigma, rho=spec.rho, beta=spec.beta, dt=dt)).step, \
            f"L63(sigma={spec.sigma}, rho={spec.rho}, beta={spec.beta:.6g})"
    if spec.kind == "emulator":
        emu = L63Emulator.load(Path(emulator_dir) / f"{spec.file}.pt")
        for p in emu.parameters():
            p.requires_grad_(False)
        assert abs(emu.cfg.dt - dt) < 1e-12
        return emu.step, f"emulator {spec.file} (teacher {emu.teacher})"
    raise ValueError(spec.kind)


@hydra.main(version_base=None, config_path="configs", config_name="gen_data_v2")
def main(cfg: DictConfig) -> None:
    import torch
    torch.set_default_dtype(torch.float64)
    params, dcfg = L63Params(**cfg.l63), DataConfigV2(**cfg.data)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # feature flag (m7 model rectangle, 2026-11-20): backgrounds of the registry models (src/configs/models.yaml)
    # are added to the explicit legacy `false_models`; backgrounds_only reuses the existing truth / obs files.
    models = {name: (lambda n=name: build_model(n, reg, cfg.emulator_dir)[:2])
              for reg in [load_registry()] for name in cfg.background.get("registry_models", [])}
    models.update({name: (lambda s=spec: false_step(s, params.dt, cfg.emulator_dir))
                   for name, spec in cfg.background.false_models.items()})
    for split, n_ic in cfg.n_ic.items():
        path = out / f"l63v2_{split}_seed{cfg.seed}.nc"
        if cfg.get("backgrounds_only", False):
            ds = xr.load_dataset(path)
            assert ds.sizes["ic"] == int(n_ic), f"{path}: {ds.sizes['ic']} cases != n_ic.{split}"
        else:
            ds = generate_split_v2(int(n_ic), split, int(cfg.seed), dcfg, params)
            ds.to_netcdf(path)
            log.info("wrote %s  x_true%s  x_true_past%s  y_obs%s", path, tuple(ds.x_true.shape),
                     tuple(ds.x_true_past.shape), tuple(ds.y_obs.shape))
        for name, make in models.items():
            step, desc = make()
            bg = lagged_background(ds, step, name, cfg.background.lags, cfg.background.tau_b, cfg.background.floor,
                                   desc, int(cfg.background.get("time_shift", 0)))
            bpath = out / f"l63v2_{split}_seed{cfg.seed}_bg_{name}.nc"
            bg.to_netcdf(bpath)
            err = np.sqrt(((bg.x0_b.values - ds.x_true.values[:, 0]) ** 2).mean(-1))
            spread = np.sqrt(np.trace(bg.B.values, axis1=1, axis2=2) / 3)
            log.info("  background %-7s median |x0_b - x_true| %.3f   median sqrt(tr B / 3) %.3f -> %s",
                     name, float(np.median(err)), float(np.median(spread)), bpath.name)


if __name__ == "__main__":
    main()
