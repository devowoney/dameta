"""
Dev report (2026-11-18, pilot: "the target is to beat L-BFGS"): learned optimizers vs the classical strong-Wolfe
L-BFGS on the dev testbed (dense-obs weak F1, sigma_q 0.1/sqrt(10), the first 20 val cases).

    python src/report_dev_hist.py <name> [<name> ...]      (HPC only; name = dev_delta.sh run name)

1. Classical reference: torch-equivalent strong-Wolfe L-BFGS (lr 1, 150 iterations, tol 1e-6) on the 20 val cases
   (cached in tmp/results/dev_delta/classical_wolfe_val20.npz): per case J*, IC error, gradient count per iterate.
2. Teacher line-search step lengths: on the cached val teacher pairs, every step k >= 1 = t * (two-loop direction of
   its memory); distribution of t (what a learned line search has to predict).
3. Per learner record: final J, IC error, diverged, and COST TO TARGET = gradient evaluations (incl. the first
   step's line search) to reach J <= J*_case (1 + tol), tol 1e-2 / 1e-3, vs the classical method on the same case:
   reached fraction, median cost (cases reached), speed-up = median classical / median learner over cases both reach.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_meta_b as R  # noqa: E402
from dameta.da.fourdvar import run_fourdvar  # noqa: E402
from dameta.meta.meta_b_direct import attach_history, lbfgs_two_loop  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
CFG = OmegaConf.load("src/configs/meta_b.yaml")
CFG.merge_with_dotlist(["dataset=v2", "data_prefix=l63v2d", "model=f1", "constraint=weak", "pattern=full",
                        "cost.sigma_q=0.0316227766"])
RES = Path("tmp/results/dev_delta")
TOLS = (1e-2, 1e-3)


def classical_run(n_val, a, b):
    """Classical strong-Wolfe L-BFGS on val cases [a, b) of the first n_val."""
    pb = R.problem(CFG, "val", n_val)
    idx = np.arange(a, b)
    r = run_fourdvar(pb.cost(idx), pb.x0_b[idx], "lbfgs_wolfe", 1.0, 150, 1e-6)
    x_true0 = pb.x_true[idx, 0].numpy()
    return {"J_iter": r["J_iter"], "grad_cum": r["grad_cum"],
            "rmse_x0": np.sqrt(((r["x0_a"] - x_true0) ** 2).mean(1))}


def classical(n_val):
    """Reference on the first n_val val cases: cached file, else merged chunk files (--chunk), else one run."""
    f = RES / f"classical_wolfe_val{n_val}.npz"
    if not f.exists():
        chunks = sorted(RES.glob(f"classical_wolfe_val{n_val}_chunk_*.npz"))
        if chunks:
            parts = [dict(np.load(c)) for c in chunks]
            out = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
            assert len(out["rmse_x0"]) == n_val, f"chunks cover {len(out['rmse_x0'])} of {n_val} cases"
        else:
            out = classical_run(n_val, 0, n_val)
        RES.mkdir(parents=True, exist_ok=True)
        np.savez(f, **out)
    return dict(np.load(f))


def cost_to(J, G, target):
    """[n] gradients to the first iterate with J <= target (inf if never)."""
    hit = J <= target[:, None]
    first = np.where(hit.any(1), hit.argmax(1), -1)
    return np.where(first >= 0, G[np.arange(len(J)), np.maximum(first, 0)].astype(float), np.inf)


def step_lengths():
    p = torch.load(Path("tmp/meta/dev_delta/pairs_lbfgs_wolfe_val.pt"), weights_only=False)
    p = attach_history(p, 10)
    sel = p["k"] > 0
    b = -lbfgs_two_loop(p["g"][sel], p["S"][sel], p["Y"][sel])
    t = ((p["target"][sel] * b).sum(1) / (b * b).sum(1)).numpy()
    q = np.percentile(t, [5, 25, 50, 75, 95])
    print(f"teacher step length t (k >= 1, {len(t)} val pairs): fraction t = 1: {np.mean(np.abs(t - 1) < 1e-8):.2f} | "
          f"t < 1: {np.mean(t < 1 - 1e-8):.2f} | t > 1: {np.mean(t > 1 + 1e-8):.2f} | percentiles 5/25/50/75/95 "
          + " ".join(f"{v:.3g}" for v in q))


def timing(name, n_val):
    """' | wall ...' from time_dev_hist.py (all n_val cases in one batched run, 1 thread), if measured."""
    f = RES / f"timing_val{n_val}_{name}.json"
    if not f.exists():
        return ""
    t = json.loads(f.read_text())
    return f" | wall {t['wall_s']:.1f} s ({t['iterations']} its, {1e3 * t['wall_per_iteration_s']:.0f} ms/it)"


def main(names, n_val=20):
    ref = classical(n_val)
    J_star = ref["J_iter"][:, -1]
    print(f"classical strong-Wolfe L-BFGS ({n_val} val): median J* {np.median(J_star):.2f} | IC {np.median(ref['rmse_x0']):.3f}"
          f" | gradients to its stop {np.median(ref['grad_cum'][:, -1]):.1f}" + timing("classical", n_val))
    ref_cost = {tol: cost_to(ref["J_iter"], ref["grad_cum"], J_star * (1 + tol)) for tol in TOLS}
    print("  classical cost to target: " + " | ".join(f"tol {tol:g}: {np.median(c):.1f}" for tol, c in ref_cost.items()))
    step_lengths()
    for name in names:
        f = next(iter(sorted(RES.glob(f"*_{name}_val.nc"))), None)
        if f is None:
            print(f"{name}: no record")
            continue
        ds = xr.load_dataset(f)
        if ds.sizes["ic"] != n_val:
            print(f"{name}: record has {ds.sizes['ic']} cases, not {n_val}")
            continue
        J, G = ds.J_iter.values, ds.grad_cum.values
        row = (f"{name:28s} J {np.nanmedian(J[:, -1]):9.2f} | IC {np.nanmedian(ds.rmse_x0.values):.3f} | "
               f"div {int(ds.diverged.sum())}/{len(J)}")
        for tol in TOLS:
            c = cost_to(J, G, J_star * (1 + tol))
            both = np.isfinite(c) & np.isfinite(ref_cost[tol])
            med = np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else np.nan
            sp = np.median(ref_cost[tol][both]) / np.median(c[both]) if both.any() else np.nan
            row += f" | tol {tol:g}: reached {np.isfinite(c).mean():.2f} cost {med:5.1f} speed-up x{sp:.2f}"
        print(row + timing(name, n_val))


if __name__ == "__main__":
    args = sys.argv[1:]
    nv = 20
    if args[:1] == ["--nval"]:
        nv, args = int(args[1]), args[2:]
    if args[:1] == ["--chunk"]:                   # python src/report_dev_hist.py --nval N --chunk a b
        a, b = int(args[1]), int(args[2])
        RES.mkdir(parents=True, exist_ok=True)
        np.savez(RES / f"classical_wolfe_val{nv}_chunk_{a:04d}.npz", **classical_run(nv, a, b))
    else:
        main(args, nv)
