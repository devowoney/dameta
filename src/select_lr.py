"""
Apply the pre-registered learning-rate selection rule to an lr sweep of classical 4D-Var (E1) on the val split.

Rule (fixed 2026-11-10, before any result was seen; memory/branches/m2-classical-4dvar/decisions.md):
    1. discard lr values with more than 1 % diverged cases;
    2. pick the lr with the lowest median final cost J (truth-free, like 4D-Var itself);
    3. tie-break (relative difference < 1e-3 in median J): fewer median gradient evaluations.

Usage:  python src/select_lr.py [sweep_dir]     (default tmp/results/e1_sweep)
Prints a markdown table per (constraint, pattern, method) and writes <sweep_dir>/selected_lr.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent


def summarize(path: Path) -> dict:
    """One sweep file -> summary row."""
    ds = xr.open_dataset(path)
    cfg = json.loads(ds.attrs["config"])
    j_final = ds.J_iter.values[:, -1]
    return {
        "constraint": cfg["constraint"], "pattern": cfg["pattern"], "method": cfg["method"], "lr": float(cfg["lr"]),
        "diverged_frac": float(ds.diverged.values.mean()),
        "conv_frac": float(ds.converged.values.mean()),
        "J_med": float(np.nanmedian(np.where(np.isfinite(j_final), j_final, np.nan))),
        "evals_med": float(np.median(ds.n_grad_evals.values)),
        "rmse_x0_med": float(np.nanmedian(ds.rmse_x0.values)),
        "rmse_fc100_med": float(np.nanmedian(ds.rmse_fc.sel(lead=100).values)),
        "wall_s": float(ds.attrs["wall_s"]),
    }


def select(rows: list) -> dict:
    """Apply the rule to the rows of one (constraint, pattern, method) group; returns the chosen row or None."""
    ok = [r for r in rows if r["diverged_frac"] <= 0.01 and np.isfinite(r["J_med"])]
    if not ok:
        return None
    best_j = min(r["J_med"] for r in ok)
    ties = [r for r in ok if (r["J_med"] - best_j) / best_j < 1e-3]
    return min(ties, key=lambda r: r["evals_med"])


def main(sweep_dir: Path) -> None:
    rows = [summarize(p) for p in sorted(sweep_dir.glob("*.nc"))]
    groups = {}
    for r in rows:
        groups.setdefault((r["constraint"], r["pattern"], r["method"]), []).append(r)
    chosen = {}
    for key in sorted(groups):
        g = sorted(groups[key], key=lambda r: r["lr"])
        best = select(g)
        chosen["/".join(key)] = best["lr"] if best else None
        print(f"\n### {' / '.join(key)}  -> selected lr = {best['lr'] if best else 'NONE (all diverge)'}")
        print("| lr | diverged | converged | median J | median evals | median rmse_x0 | median rmse_fc@1TU |")
        print("|---|---|---|---|---|---|---|")
        for r in g:
            mark = " **←**" if best and r["lr"] == best["lr"] else ""
            print(f"| {r['lr']:g}{mark} | {r['diverged_frac']:.0%} | {r['conv_frac']:.0%} | {r['J_med']:.2f} | "
                  f"{r['evals_med']:.0f} | {r['rmse_x0_med']:.3f} | {r['rmse_fc100_med']:.3f} |")
    (sweep_dir / "selected_lr.json").write_text(json.dumps(chosen, indent=2))
    print(f"\nwrote {sweep_dir / 'selected_lr.json'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tmp/results/e1_sweep")
