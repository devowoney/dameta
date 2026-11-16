"""
Wall-clock cost of ONE iteration of cell 8a (forward run of F + learned-operator adjoint recursion, network included)
vs one exact gradient evaluation (forward + adjoint) and one forward run, on 200 test cases (strong / full, 1 core).
Writes tmp/results/cell8_cost.json; used to report the wall-clock speed-up next to the model-run speed-up.

Usage (project root):  python src/measure_cell8_cost.py
"""

import json
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_4dvar_v2 as R2  # noqa: E402
from dameta.meta.learned_gradient import LearnedGradient, analytic_grad, forward_run  # noqa: E402

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
    for con, pat in (("strong", "full"), ("weak", "full"), ("strong", "x_only"), ("weak", "x_only")):
        for model in ("f1", "emulator"):
            ck = next((ROOT / f"tmp/meta/m5_{model}").glob(f"*_{con}_{pat}_adagrad_*_c8a_s0.pt"), None)
            if ck is None:
                continue
            cfg = base.copy()
            cfg.model, cfg.split, cfg.constraint, cfg.pattern = model, "test", con, pat
            P = R2.setup_problem(cfg, "test")
            cost = P["cost"]
            n = len(P["x0_b"])
            c = cost.pack(P["x0_b"], torch.zeros(n, cost.cfg.n_obs - 1, 3) if cost.weak else None)
            lg = LearnedGradient.from_state_dict(torch.load(ck, weights_only=False))
            t_fwd = timed(lambda: forward_run(cost, c))
            t_grad = timed(lambda: cost.grad(c))
            fwd = forward_run(cost, c)
            t_net = timed(lambda: analytic_grad(cost, c) + lg.grad_obs(c, 0, fwd, cost.weak))
            it8 = t_fwd + t_net
            key = model if (con, pat) == ("strong", "full") else f"{model}/{con}/{pat}"
            out[key] = {"forward_s": t_fwd, "gradient_s": t_grad, "cell8_network_s": t_net, "cell8_iteration_s": it8,
                        "gradient_over_cell8_iteration": t_grad / it8}
            print(f"{model:8s} {con:6s} {pat:6s}: forward {t_fwd * 1e3:6.1f} ms | exact gradient {t_grad * 1e3:6.1f} ms"
                  f" | cell-8 network {t_net * 1e3:5.1f} ms | iteration {it8 * 1e3:6.1f} ms -> gradient / cell-8 = "
                  f"{t_grad / it8:.2f}")
    (ROOT / "tmp/results/cell8_cost.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
