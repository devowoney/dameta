"""
Wall-clock timing for the dev report (2026-11-19, pilot: "add a column of computation time").

    python src/time_dev_hist.py <name> [<checkpoint name>]        (HPC only, one process per method, 1 thread)
        name classical               -> classical strong-Wolfe L-BFGS (lr 1, 150 its, tol 1e-6)
        name <run> + <checkpoint>    -> frozen delta learner tmp/meta/dev_delta/<checkpoint>.pt, backtrack safeguard
    -> tmp/results/dev_delta/timing_val200_<name>.json: wall_s (all 200 val cases solved in ONE batched run, to the
       method's own stopping rule; network time included), iterations run, wall per iteration.
The per-case strong-Wolfe fallback of the classical line search is a Python loop (reference_step), counted as is.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_dev_hist import CFG, RES, R  # noqa: E402
from dameta.da.fourdvar import run_fourdvar  # noqa: E402
from dameta.meta.meta_b_direct import DirectDelta, run_direct  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
N_VAL = 200


def main(name, ckpt=None):
    pb = R.problem(CFG, "val", N_VAL)
    cost, xb = pb.cost(), pb.x0_b
    if name == "classical":
        r = run_fourdvar(cost, xb, "lbfgs_wolfe", 1.0, 150, 1e-6)
    else:
        lrn = DirectDelta.from_state_dict(torch.load(Path("tmp/meta/dev_delta") / f"{ckpt}.pt", weights_only=False))
        run_direct(R.problem(CFG, "val", 2).cost(), xb[:2], lrn, 3, 1e-6, safeguard="backtrack")   # warm-up
        r = run_direct(cost, xb, lrn, 150, 1e-6, safeguard="backtrack")
    n_it = int(np.max(r["n_grad_evals"]))
    out = {"name": name, "checkpoint": ckpt, "n_cases": N_VAL, "wall_s": float(r["wall_s"]), "iterations": n_it,
           "wall_per_iteration_s": float(r["wall_s"]) / max(n_it, 1),
           "median_grad_cum_final": float(np.median(r["grad_cum"][:, -1]))}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"timing_val{N_VAL}_{name}.json").write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
