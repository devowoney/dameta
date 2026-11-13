"""
Screen candidate degraded TEACHER models (biased L63 parameters) against the true model M1, on D1 VAL ICs.

Usage (project root):  python src/screen_teachers.py
Writes tmp/results/emulator/teacher_screen.{csv,md}.

Metrics per teacher (medians over val ICs; teacher integrated with RK4 at the same dt as M1):
  obs_int_err_{x,y,z}  std of the one-observation-interval model error (10 steps from TRUE states along the
                       truth trajectories) — the model error 4D-Var actually sees between observations
  rmse_0.5TU / 1TU     free-forecast RMSE from the true x0 (teacher vs truth)
  vpt                  valid prediction time [TU]: first lead where the error exceeds 0.4 x climatological std
  lambda1              leading Lyapunov exponent of the teacher (renormalised TL propagation, 20 TU)
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dameta.dynamics.emulator import TeacherConfig, teacher_rollout, teacher_tendency  # noqa: E402
from dameta.dynamics.l63 import rk4_step  # noqa: E402
from dameta.eval.emulator_metrics import lyapunov_exponent  # noqa: E402
from report_e1 import to_markdown  # noqa: E402

torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parent.parent
CANDIDATES = {
    "clean (truth)": {},
    "xz095 (current main)": {"xz_scale": 0.95},
    "rho 27.5": {"rho": 27.5}, "rho 27": {"rho": 27.0}, "rho 28.5": {"rho": 28.5}, "rho 29": {"rho": 29.0},
    "sigma 9.5": {"sigma": 9.5}, "sigma 10.5": {"sigma": 10.5}, "sigma 11": {"sigma": 11.0},
    "beta 2.5": {"beta": 2.5}, "beta 2.8": {"beta": 2.8},
}


def main():
    ds = xr.open_dataset(ROOT / "tmp/data/l63_val_seed42.nc")
    xt = torch.tensor(ds.x_true.values)                        # [n, 601, 3] truth (M1)
    clim = float(xt.reshape(-1, 3).std(0).mean())
    thr = 0.4 * clim
    starts = xt[:, :500:10].reshape(-1, 3)                     # true states every obs interval
    ends = xt[:, 10:510:10].reshape(-1, 3)
    rows = []
    for name, kw in CANDIDATES.items():
        t = TeacherConfig(**kw)
        step = lambda v, t=t: rk4_step(lambda u: teacher_tendency(u, t), v, t.dt)  # noqa: E731
        x = starts
        for _ in range(10):
            x = step(x)
        e = (x - ends).std(0).numpy()
        traj = teacher_rollout(xt[:, 0], 200, t)
        err = ((traj - xt[:, :201]) ** 2).mean(-1).sqrt()      # [n, 201]
        hit = (err > thr).numpy()
        vpt = np.where(hit.any(1), hit.argmax(1), 200) * t.dt
        lam = lyapunov_exponent(step, xt[:50, 0], 2000, t.dt)
        rows.append({"teacher": name, "obs_int_err_x": e[0], "obs_int_err_y": e[1], "obs_int_err_z": e[2],
                     "rmse_0.5TU": float(err[:, 50].median()), "rmse_1TU": float(err[:, 100].median()),
                     "vpt_TU": float(np.median(vpt)), "lambda1": float(lam.median())})
    df = pd.DataFrame(rows)
    out = ROOT / "tmp/results/emulator/teacher_screen"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".csv"), index=False)
    md = to_markdown(df) + f"\n\nVPT threshold 0.4 x clim std = {thr:.2f}; observation error std sigma_o = 1.41.\n"
    out.with_suffix(".md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
