"""
Emulator diagnostics + gradient fidelity G1 on test and test_ood (never used in training).

Usage:  python src/eval_emulator.py clean [more names ...]
Writes tmp/results/emulator/<name>_diag.json (medians + quartiles) and prints a markdown summary.

Measured per split:
    one-step RMSE; free-forecast RMSE from the TRUE x0 at leads 0.5 / 1 / 2 TU; VPT (threshold 0.4 x clim std);
    leading Lyapunov exponent (16 trajectories x 50 TU; M1 reference 0.904 +/- 0.012);
    TL fidelity over the window (cos, relative error) along each model's own trajectory from the true x0;
    G1 = gradient of the 4D-Var cost through the emulator vs exact M1 gradient (same obs/background/B/R/Q):
         strong/full at x0_b and at x0_true, weak/full at (x0_b, eta = 0), strong/x_only at x0_b.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dameta.da import CostConfig  # noqa: E402
from dameta.dynamics import L63Model  # noqa: E402
from dameta.dynamics.emulator import L63Emulator  # noqa: E402
from dameta.eval import free_forecast_rmse, gradient_fidelity, lyapunov_exponent, tl_fidelity  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DT = 0.01


def q(x) -> dict:
    """Median and quartiles of a 1-D tensor/array (NaN-safe)."""
    a = np.asarray(x, dtype=float)
    return {"med": float(np.nanmedian(a)), "q25": float(np.nanpercentile(a, 25)), "q75": float(np.nanpercentile(a, 75))}


def evaluate(name: str) -> dict:
    torch.set_default_dtype(torch.float64)
    emu, m1 = L63Emulator.load(ROOT / f"tmp/emulators/{name}.pt"), L63Model()
    for p in emu.parameters():
        p.requires_grad_(False)                                   # emulator frozen: only state gradients needed
    cfg = CostConfig()
    clim_std = float(xr.open_dataset(ROOT / "tmp/data/l63_train_seed42.nc").x_true.std())
    out = {"name": name, "teacher": emu.teacher.__dict__, "best_val_rmse": emu.record["best_val_rmse"]}
    for split in ["test", "test_ood"]:
        ds = xr.open_dataset(ROOT / f"tmp/data/l63_{split}_seed42.nc")
        xt = torch.tensor(ds.x_true.values)
        y, xb = torch.tensor(ds.y_obs.values), torch.tensor(ds.x0_b.values)
        r = {}
        with torch.no_grad():
            r["one_step_rmse"] = q(((emu.step(xt[:, :-1]) - xt[:, 1:]) ** 2).mean(-1).sqrt().mean(1))
        fc = free_forecast_rmse(emu.step, xt[:, 0], xt).numpy()
        for lead in (50, 100, 200):
            r[f"free_fc_rmse_{lead * DT:g}TU"] = q(fc[:, lead])
        above = fc > 0.4 * clim_std
        r["vpt_TU"] = q(np.where(above.any(1), above.argmax(1), fc.shape[1] - 1) * DT)
        r["lyapunov"] = q(lyapunov_exponent(emu.step, xt[:16, 0], 5000, DT).numpy())
        tl = tl_fidelity(emu.step, m1.step, xt[:, 0], cfg.window)
        r["tl_cos"], r["tl_rel_err"] = q(tl["cos"].numpy()), q(tl["rel_err"].numpy())
        for tag, pattern, weak, x0 in [("strong_full_at_xb", "full", False, xb), ("strong_full_at_truth", "full", False,
                                       xt[:, 0]), ("weak_full_at_xb", "full", True, xb),
                                       ("strong_xonly_at_xb", "x_only", False, xb)]:
            mask = torch.tensor(ds.obs_mask.sel(pattern=pattern).values)
            ctrl = torch.cat([x0, torch.zeros(len(x0), 3 * (cfg.n_obs - 1))], 1) if weak else x0
            g1 = gradient_fidelity(emu.step, m1.step, y, mask, xb, cfg, ctrl, weak)
            r[f"G1_cos_{tag}"], r[f"G1_rel_{tag}"] = q(g1["cos"].numpy()), q(g1["rel_err"].numpy())
            r[f"G1_frac_cos_gt_0.9_{tag}"] = float((g1["cos"] > 0.9).double().mean())
        out[split] = r
    return out


def main(names: list) -> None:
    res_dir = ROOT / "tmp/results/emulator"
    res_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        res = evaluate(name)
        (res_dir / f"{name}_diag.json").write_text(json.dumps(res, indent=2))
        print(f"\n### emulator `{name}`  teacher={res['teacher']}  best val rmse@0.5TU={res['best_val_rmse']:.4f}")
        print("| metric | test (median [IQR]) | test_ood (median [IQR]) |\n|---|---|---|")
        for k in res["test"]:
            a, b = res["test"][k], res["test_ood"][k]
            fa = f"{a['med']:.4g} [{a['q25']:.3g}, {a['q75']:.3g}]" if isinstance(a, dict) else f"{a:.3f}"
            fb = f"{b['med']:.4g} [{b['q25']:.3g}, {b['q75']:.3g}]" if isinstance(b, dict) else f"{b:.3f}"
            print(f"| {k} | {fa} | {fb} |")


if __name__ == "__main__":
    main(sys.argv[1:] or ["clean"])
