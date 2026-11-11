"""
Hydra entry point: train the M2 emulator and save it to tmp/emulators/<name>.pt.

Usage (project root, env oceanai):
    python src/train_emulator.py                                   # clean emulator (exact L63 teacher)
    python src/train_emulator.py name=rho26 teacher.rho=26         # deliberately degraded teacher (example)
"""

import logging
import sys
from pathlib import Path

import hydra
import torch
import xarray as xr
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make `dameta` importable without installation

from dameta.dynamics.emulator import EmulatorConfig, TeacherConfig  # noqa: E402
from dameta.dynamics.train_emulator import TrainConfig, train_emulator  # noqa: E402

OmegaConf.register_new_resolver("project_root", lambda: str(Path(__file__).resolve().parent.parent), replace=True)
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="emulator")
def main(cfg: DictConfig) -> None:
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    data = Path(cfg.data_dir)
    tr = xr.open_dataset(data / f"l63_train_seed{cfg.seed}.nc")
    va = xr.open_dataset(data / f"l63_val_seed{cfg.seed}.nc")
    n_steps = tr.sizes["t"] - 1                                         # same trajectory length as D1 (L + F)
    teacher = TeacherConfig(**cfg.teacher)
    emu, record = train_emulator(torch.tensor(tr.x_true.values[:, 0]), torch.tensor(va.x_true.values[:, 0]), n_steps,
                                 teacher, EmulatorConfig(**cfg.emulator), TrainConfig(**cfg.train), log=log.info)
    record["config"] = OmegaConf.to_container(cfg, resolve=True)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emu.save(out / f"{cfg.name}.pt", teacher, record)
    log.info("saved %s  best val rmse@%d = %.4f  wall %.0fs", out / f"{cfg.name}.pt", cfg.train.val_roll,
             record["best_val_rmse"], record["wall_s"])


if __name__ == "__main__":
    main()
