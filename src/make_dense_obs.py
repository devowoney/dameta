"""
D1 v2-dense (pilot request 2026-11-16): same truth, window (1.1 TU = 110 steps), first guess and B as D1 v2, but the
truth is observed at EVERY model step (obs_every = 1 -> 111 observation times instead of 12).

Nested with D1 v2: at the 12 original observation times (t = 0, 10, ..., 110) the observation (and its noise) is
copied from D1 v2; only the 99 new times get fresh noise N(0, sigma_o2 I) (seeded per split). The dense and sparse
problems therefore differ ONLY by the added observations. Backgrounds (first guess, lagged B) do not depend on the
observations: the D1 v2 background files are reused as they are (bg files are linked under the dense prefix).

Output: tmp/data/l63v2d_<split>_seed<S>.nc (+ symlinks l63v2d_<split>_seed<S>_bg_<name>.nc -> D1 v2 background)
Usage (project root, env oceanai):  python src/make_dense_obs.py [--splits val test test_ood] [--seed 42]
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "tmp/data"
SPLIT_OFFSET = {"train": 0, "val": 1, "test": 2, "test_ood": 3}


def make_dense(ds: xr.Dataset, seed: int) -> xr.Dataset:
    L, s2 = int(ds.attrs["window"]), float(ds.attrs["sigma_o2"])
    t_obs = np.arange(L + 1)
    rng = np.random.default_rng([seed, 2026_11_16, SPLIT_OFFSET[ds.attrs["split"]]])
    y = ds.x_true.values[:, :L + 1] + np.sqrt(s2) * rng.standard_normal((ds.sizes["ic"], L + 1, 3))
    old = ds.t_obs.values
    y[:, old] = ds.y_obs.values                                   # nested: keep the D1 v2 observations exactly
    mask = np.zeros((2, L + 1, 3), dtype=bool)
    mask[0] = True
    mask[1, :, 0] = True
    out = ds.drop_vars(["y_obs", "obs_mask", "t_obs"])
    out = out.assign_coords(t_obs=t_obs)
    out["y_obs"] = (("ic", "t_obs", "var"), y)
    out["obs_mask"] = (("pattern", "t_obs", "var"), mask)
    out.attrs.update({"obs_every": 1, "dataset": "D1v2-dense",
                      "description": ds.attrs["description"] + " | DENSE: obs at every step, nested with D1 v2"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "test", "test_ood"])
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    for split in a.splits:
        ds = xr.load_dataset(DATA / f"l63v2_{split}_seed{a.seed}.nc")
        out = make_dense(ds, a.seed)
        path = DATA / f"l63v2d_{split}_seed{a.seed}.nc"
        out.to_netcdf(path)
        for bg in DATA.glob(f"l63v2_{split}_seed{a.seed}_bg_*.nc"):
            link = DATA / bg.name.replace("l63v2_", "l63v2d_", 1)
            if not link.exists():
                link.symlink_to(bg.name)
        assert np.array_equal(out.y_obs.values[:, ds.t_obs.values], ds.y_obs.values)
        print(f"{path.name}: y_obs {out.y_obs.shape}, obs_every {out.attrs['obs_every']}")


if __name__ == "__main__":
    main()
