"""
Per-model diagonal model-error covariance Q for weak-constraint 4D-Var (M8, pilot 2026-11-26).

For every estimation model m of the rectangle, the one-observation-interval error started from the truth,
    e_i = M_m^(obs_every)( x_true(t_i) ) - x_true(t_i + obs_every),     t_i = 0, obs_every, ..., L - obs_every,
is collected over the TRAIN split (1000 cases x n_obs - 1 intervals) and
    Q_m = diag( E[e_x^2], E[e_y^2], E[e_z^2] )          (second moment = RMS^2, bias included)
i.e. an oracle Q (uses the truth, standard in OSSEs): weak 4D-Var receives the right model-error size per variable.
Reason: sigma_q = 0.1 was 4-30x below the model error on the 1.6-TU / obs-every-20 setup, so eta stayed ~0 and the
weak analysis equalled the strong one.

Usage (project root):  python src/estimate_q.py [--data-dir tmp/data] [--split train]  -> <data-dir>/model_q_seed42.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dameta.dynamics.models import build_model, load_registry  # noqa: E402

MODELS = ("f1", "f2", "e1", "e2")


@torch.no_grad()
def interval_errors(step, x_true: torch.Tensor, window: int, every: int) -> torch.Tensor:
    """[n, n_obs - 1, 3] one-interval errors of `step` started from the truth at every obs time of the window."""
    errs = []
    for t0 in range(0, window, every):
        x = x_true[:, t0]
        for _ in range(every):
            x = step(x)
        errs.append(x - x_true[:, t0 + every])
    return torch.stack(errs, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="tmp/data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    d = (ROOT / a.data_dir).resolve()
    ds = xr.load_dataset(d / f"l63v2_{a.split}_seed{a.seed}.nc")
    window, every = int(ds.attrs["window"]), int(ds.attrs["obs_every"])
    x_true = torch.tensor(ds.x_true.values[:, : window + 1])
    reg, out = load_registry(), {}
    for m in MODELS:
        step = build_model(m, reg, str(ROOT / "tmp/emulators"))[0]
        e = interval_errors(step, x_true, window, every).reshape(-1, 3).numpy()
        q = (e ** 2).mean(0)
        out[m] = {"q_var": q.tolist(), "rms": np.sqrt(q).tolist(), "bias": e.mean(0).tolist(),
                  "std": e.std(0).tolist(), "n_samples": int(e.shape[0]), "split": a.split, "window": window,
                  "obs_every": every}
        print(f"{m}: RMS (x, y, z) {np.round(np.sqrt(q), 3)}  bias {np.round(e.mean(0), 3)}  "
              f"std {np.round(e.std(0), 3)}  ({e.shape[0]} intervals)")
    path = d / f"model_q_seed{a.seed}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
