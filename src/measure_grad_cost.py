"""
Measure the wall-clock cost of one gradient evaluation (forward + adjoint) relative to one forward run of the 4D-Var
cost, for each D1 v2 estimation model (strong / weak, 200 test cases, batched). Used by report_m5.py to express every
optimiser's cost in forward-run equivalents. Writes tmp/results/grad_cost_ratio.json.

Usage (project root):  python src/measure_grad_cost.py
"""

import json
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_4dvar_v2 as R2  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def timed(fn, reps=20):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def main():
    torch.set_num_threads(1)
    base = OmegaConf.load(ROOT / "src/configs/fourdvar_v2.yaml")
    del base["hydra"]
    base.data_dir, base.emulator_dir = str(ROOT / "tmp/data"), str(ROOT / "tmp/emulators")
    base.result_group = base.out_dir = "x"
    out = {}
    for model in ("f1", "emulator", "truth"):
        for con in ("strong", "weak"):
            cfg = base.copy()
            cfg.model, cfg.constraint, cfg.split = model, con, "test"
            P = R2.setup_problem(cfg, "test")
            cost = P["cost"]
            n = len(P["x0_b"])
            c = cost.pack(P["x0_b"], torch.zeros(n, cost.cfg.n_obs - 1, 3) if cost.weak else None)
            with torch.no_grad():
                t_f = timed(lambda: cost(*cost.unpack(c)))
            t_g = timed(lambda: cost.grad(c))
            out[f"{model}/{con}"] = {"forward_s": t_f, "gradient_s": t_g, "ratio": t_g / t_f}
            print(f"{model:8s} {con:6s}: forward {t_f * 1e3:7.1f} ms, gradient {t_g * 1e3:7.1f} ms, ratio {t_g / t_f:.2f}")
    (ROOT / "tmp/results/grad_cost_ratio.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
