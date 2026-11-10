"""
Hydra entry point: generate the D1 dataset (train / val / test) into tmp/data/.

Usage (from the project root, conda env oceanai):
    python src/gen_data.py                         # default config src/configs/gen_data.yaml
    python src/gen_data.py seed=7 n_ic.train=100   # Hydra overrides
"""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make `dameta` importable without installation

from dameta.data import DataConfig, generate_split  # noqa: E402
from dameta.dynamics import L63Params  # noqa: E402

# Project root resolver (ported from bice): keeps every Hydra artifact under <project>/tmp whatever the cwd
OmegaConf.register_new_resolver("project_root", lambda: str(Path(__file__).resolve().parent.parent), replace=True)
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="gen_data")
def main(cfg: DictConfig) -> None:
    params = L63Params(**cfg.l63)
    dcfg = DataConfig(**cfg.data)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, n_ic in cfg.n_ic.items():
        ds = generate_split(int(n_ic), split, int(cfg.seed), dcfg, params)
        path = out_dir / f"l63_{split}_seed{cfg.seed}.nc"
        ds.to_netcdf(path)
        log.info("wrote %s  x_true%s  y_obs%s", path, tuple(ds.x_true.shape), tuple(ds.y_obs.shape))


if __name__ == "__main__":
    main()
