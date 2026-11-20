"""
Unit computational costs of the ICE table (M6, pilot 2026-11-20: "total computational cost for each method, in time").

Measured on ONE node, one thread, for each (estimation model, constraint), on the 200 test cases (one batch):
    t_fwd    one forward run of the model over the window (J and the states at the observation times)
    t_grad   one gradient = forward + adjoint run (autodiff)
    t_net    per-iteration cost of each meta cell OUTSIDE the model runs (median of 20 repeats):
               c1 / c2  learner move Delta(x, g, k, L-BFGS memory of 10 pairs)
               c3       L-BFGS two-loop of its own memory + learned correction
               c4       learned one-interval TL operators + adjoint recursion + grad J_b (+ grad J_q): g^ (no model run)
The learners are the OSSE-trained seed-0 checkpoints (tmp/meta/ice_osse_meta_<model>; a missing one falls back to the
F1 learner of the same constraint: same architecture, same cost). All times in seconds per 200-case batch.
report_ice.py turns run counts into time: classical = grads x t_grad; cells 1-3 = grads x t_grad + iters x t_net;
cell 4 = forward runs x t_fwd + iters x t_net.

Usage (project root, HPC):  python src/measure_ice_cost.py   -> tmp/results/ice/unit_cost.json
"""

import json
import statistics
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_4dvar_v2 as R2  # noqa: E402
from dameta.da.reference_step import LBFGSDirection  # noqa: E402
from dameta.meta.learned_gradient import LearnedGradient, analytic_grad, forward_run  # noqa: E402
from dameta.meta.meta_a import LearnedOptimization  # noqa: E402
from dameta.meta.meta_b_direct import DirectDelta  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PREFIX = {"c1": "delta_exact", "c2": "delta", "c3": "unroll", "c4": "learned_grad"}


def timed(fn, reps=20):
    fn()                                                    # warm-up
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def checkpoint(cell, model, constraint):
    for m in (model, "f1"):
        p = ROOT / f"tmp/meta/ice_osse_meta_{m}/{PREFIX[cell]}_{constraint}_x_only_lbfgs_wolfe_lr1_{cell}_{constraint}_s0.pt"
        if p.exists():
            return p
    return None


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    out = {}
    for model in ("f1", "emulator"):
        for constraint in ("strong", "weak"):
            cfg = OmegaConf.load(ROOT / "src/configs/fourdvar_v2.yaml")
            del cfg["hydra"]
            cfg.data_dir, cfg.emulator_dir = str(ROOT / "tmp/data"), str(ROOT / "tmp/emulators")
            cfg.result_group = cfg.out_dir = "unused"
            cfg.model, cfg.constraint, cfg.pattern, cfg.split = model, constraint, "x_only", "test"
            P = R2.setup_problem(cfg, "test")
            cost, n = P["cost"], P["x0_b"].shape[0]
            eta0 = torch.zeros(n, P["ccfg"].n_obs - 1, 3) if cost.weak else None
            c = cost.pack(P["x0_b"], eta0)
            d = c.shape[1]
            with torch.no_grad():
                t_fwd = timed(lambda: cost(*cost.unpack(c)))
            t_grad = timed(lambda: cost.grad(c))
            _, g = cost.grad(c)
            gen = torch.Generator().manual_seed(0)
            S = 0.1 * torch.randn(n, 10, d, generator=gen)          # a full, valid 10-pair memory (s.y > 0)
            Y = 2.0 * S + 0.01 * torch.randn(n, 10, d, generator=gen)
            res = {"t_fwd": t_fwd, "t_grad": t_grad, "n_cases": n}
            for cell in ("c1", "c2", "c3", "c4"):
                ck = checkpoint(cell, model, constraint)
                if ck is None:
                    res[f"t_net_{cell}"] = None
                    continue
                sd = torch.load(ck, weights_only=False)
                if cell in ("c1", "c2"):
                    lrn = DirectDelta.from_state_dict(sd)
                    with torch.no_grad():
                        res[f"t_net_{cell}"] = timed(lambda: lrn.delta(c, g, 3, None, None, (S, Y)))
                elif cell == "c3":
                    lrn = LearnedOptimization.from_state_dict(sd)
                    lb = LBFGSDirection(history=10)
                    x = c.clone()
                    for i in range(11):                             # fill its memory with 10 pairs
                        lb(x + 0.01 * i, g + 0.02 * i)
                    with torch.no_grad():
                        res[f"t_net_{cell}"] = timed(lambda: lb.__class__.__call__(lb, c + 0.2, g) + lrn.delta(c, g, 3, None))
                else:
                    lrn = LearnedGradient.from_state_dict(sd)
                    fwd = forward_run(cost, c)
                    res[f"t_net_{cell}"] = timed(lambda: analytic_grad(cost, c) + lrn.grad_obs(c, 3, fwd, cost.weak))
                res[f"ckpt_{cell}"] = str(ck.relative_to(ROOT))
            out[f"{model}/{constraint}"] = res
            print(model, constraint, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in res.items()
                                      if not k.startswith("ckpt")}, flush=True)
    path = ROOT / "tmp/results/ice/unit_cost.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print("->", path)


if __name__ == "__main__":
    main()
