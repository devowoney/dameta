"""
Screen degraded emulators against their OWN teacher (pilot 2026-11-23: emulator approximation must be visible).

Usage (project root):  python src/screen_emulators.py <teacher model> <emulator name> [<emulator name> ...]
    e.g. python src/screen_emulators.py f1 e1_d100 e1_d30 e1_a16 ...
Writes tmp/results/models/emulator_screen_<teacher>.{csv,md}.

Per emulator (D1 v2 val ICs, teacher = registry model, RK4 dt 0.01):
  m_h         RMS over (x, y, z) of the pooled std of the one-obs-interval difference emulator - teacher, from true
              states every obs interval, / sigma_o   -> pre-registered pick: closest to 0.15 (= F1's model error)
  rmse_1TU    |emulator - teacher| from the true x0 at 1 TU (mean ± std over cases)
  stable_20TU, lambda1  long-run sanity (50 ICs)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dameta.dynamics.emulator import L63Emulator  # noqa: E402
from dameta.dynamics.models import build_model, load_registry  # noqa: E402
from dameta.eval.emulator_metrics import lyapunov_exponent  # noqa: E402
from report_e1 import to_markdown  # noqa: E402

torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parent.parent
SIGMA_O = float(np.sqrt(2.0))
TARGET = 0.15


@torch.no_grad()
def roll(step, x, n):
    out = [x]
    for _ in range(n):
        x = step(x)
        out.append(x)
    return torch.stack(out, 1)


def main(teacher, names):
    reg = load_registry()
    t_step = build_model(teacher, reg)[0]
    ds = xr.load_dataset(ROOT / "tmp/data/l63v2_val_seed42.nc")
    xt = torch.tensor(ds.x_true.values)
    starts = xt[:, :500:10].reshape(-1, 3)
    with torch.no_grad():
        ref = starts
        for _ in range(10):
            ref = t_step(ref)
    tr_t = roll(t_step, xt[:, 0], 100)
    rows = []
    for n in names:
        emu = L63Emulator.load(ROOT / f"tmp/emulators/{n}.pt")
        for q in emu.parameters():
            q.requires_grad_(False)
        rec = torch.load(ROOT / f"tmp/emulators/{n}.pt", weights_only=False).get("record", {})
        cfg = rec.get("config", {})
        with torch.no_grad():
            x = starts
            for _ in range(10):
                x = emu.step(x)
        e = (x - ref).std(0)
        diff = ((roll(emu.step, xt[:, 0], 100) - tr_t) ** 2).mean(-1).sqrt()[:, 100].numpy()
        long = roll(emu.step, xt[:50, 0], 2000)
        stable = bool(torch.isfinite(long).all() and long.abs().max() < 100)
        lam = lyapunov_exponent(emu.step, xt[:50, 0], 2000, 0.01).numpy() if stable else np.full(50, np.nan)
        rows.append({"emulator": n, "n_train_ic": cfg.get("n_train_ic", 1000),
                     "width": emu.cfg.width, "depth": emu.cfg.depth,
                     "m_h": float(e.pow(2).mean().sqrt()) / SIGMA_O,
                     "rmse_1TU": f"{np.nanmean(diff):.3f} ± {np.nanstd(diff):.3f}",
                     "stable_20TU": stable, "lambda1": f"{np.nanmean(lam):.3f} ± {np.nanstd(lam):.3f}",
                     "_lam": float(np.nanmean(lam))})
    df = pd.DataFrame(rows)
    ok = df[df.stable_20TU & (df._lam > 0)]
    pick = str(ok.iloc[(ok.m_h - TARGET).abs().argmin()].emulator) if len(ok) else "none"
    out = ROOT / f"tmp/results/models/emulator_screen_{teacher}"
    out.parent.mkdir(parents=True, exist_ok=True)
    pub = df.drop(columns="_lam")
    pub.to_csv(out.with_suffix(".csv"), index=False)
    md = (to_markdown(pub.round(4)) + f"\n\nTeacher {teacher.upper()}; {len(xt)} val cases; pre-registered pick "
          f"(m_h closest to {TARGET} sigma_o, stable, lambda1 > 0): **{pick}**\n")
    out.with_suffix(".md").write_text(md)
    print(md)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
