"""
M8-R (pilot 2026-11-26): truth-free scalar model-error covariance Q_m = q_m I by the chi2 (Talagrand) consistency
of the weak-constraint 4D-Var minimum.

For a consistent linear-Gaussian system, E[2 J_min] = p (number of scalar observations in the window). On the q grid
(src/scripts/m8_lists.py qtune: weak L-BFGS, 50 TRAIN cases, full obs, shared truth-model background, B x b_infl):
    r(q) = MEDIAN over cases of 2 J_min / p,   p = obs times x observed variables (full: 9 x 3 = 27)
and q_m = the grid value with |log r(q)| smallest (r closest to 1). Median, not mean (pilot 2026-11-26): a few cases
in local minima (2 J_min / p >> 1) dominated the mean on the L1-truth problems (std 2-4 x the mean).
Plateau rule (pilot 2026-11-26): if |r(q_min) - 1| <= PLATEAU already at the smallest grid q (chi2 cannot see the model
error), q_m = the LARGEST grid q with r(q) >= 1 - PLATEAU (edge of the plateau: most model error the data tolerate). Only observations / the cost are used (no truth).
A pick on a grid edge is flagged (rule: extend the grid at that edge). Diverged cases are excluded and counted.

Usage (project root):  python src/select_q.py
    -> <data dir>/q_chi2_seed42.json  {model: {q, sigma_q, ratio, ratio_std, edge, n_div, grid: {q: ratio}}}
    -> tmp/results/m8r/qtune.md
"""

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src/scripts"))
from m8_lists import Q_EXTRA, Q_GRID, STAGES  # noqa: E402

RES = ROOT / "tmp/results/m8r/qtune"
PLATEAU = 0.05


def main():
    md = ["# M8-R — chi2-tuned scalar Q (weak L-BFGS, 50 train cases, full obs)", "",
          "ratio = MEDIAN 2 J_min / p over cases (table: median; pick column also mean ± std), p = number of "
          "scalar observations; pick = median closest to 1.",
          ""]
    for tag, (ddir, bg, _, models) in STAGES.items():
        out = {}
        grid_q = sorted(Q_GRID + Q_EXTRA.get(tag, []))
        md += [f"## truth {tag} (data {ddir}, shared background {bg})", "",
               "| model | " + " | ".join(f"{q:g}" for q in grid_q) + " | pick q | median | mean ± std | edge |",
               "|---" * (len(grid_q) + 5) + "|"]
        for m in models:
            grid, mean, std, ndiv = {}, {}, {}, {}
            for q in grid_q:
                f = RES / tag / f"q{q:g}" / m / "bice_weak_full_lbfgs_wolfe_lr1_train.nc"
                if not f.exists():
                    continue
                ds = xr.load_dataset(f)
                cfg = json.loads(ds.attrs["config"])
                d = xr.load_dataset(Path(cfg["data_dir"]) / f"l63v2_train_seed{cfg['seed']}.nc")
                p = int(d.obs_mask.sel(pattern="full").values.sum())
                ok = ~ds.diverged.values
                r = 2 * ds.J_iter.values[ok, -1] / p
                grid[q], mean[q], std[q] = float(np.median(r)), float(r.mean()), float(r.std())
                ndiv[q] = int((~ok).sum())
            if not grid:
                md.append(f"| {m} | missing |")
                continue
            qs = list(grid)
            plateau = abs(grid[qs[0]] - 1) <= PLATEAU
            if plateau:
                best = max(q for q in qs if grid[q] >= 1 - PLATEAU)
                edge = best == grid_q[-1]
            else:
                best = min(qs, key=lambda q: abs(np.log(grid[q])))
                edge = best in (grid_q[0], grid_q[-1])
            out[m] = {"q": best, "sigma_q": best ** 0.5, "ratio_median": grid[best], "ratio_mean": mean[best], "ratio_std": std[best],
                      "edge": edge, "plateau": plateau, "n_div": ndiv[best], "n_cases": 50, "split": "train", "pattern": "full",
                      "criterion": "median 2 J_min / p closest to 1; plateau rule (5 %)", "grid": {f"{q:g}": grid[q] for q in qs},
                      "grid_n_div": {f"{q:g}": ndiv[q] for q in qs}}
            md.append(f"| {m} | " + " | ".join(f"{grid[q]:.2f}" if q in grid else "—" for q in grid_q)
                      + f" | **{best:g}** | {grid[best]:.2f} | {mean[best]:.2f} ± {std[best]:.2f} | {'EDGE' if edge else ''}{' plateau' if plateau else ''} |")
        md.append("")
        if out:
            path = ROOT / ddir / "q_chi2_seed42.json"
            path.write_text(json.dumps(out, indent=1))
            print(f"wrote {path}")
    (RES.parent / "qtune.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
