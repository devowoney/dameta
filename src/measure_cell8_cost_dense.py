"""
Wall-clock of ONE cell-8a iteration (F1 forward run + learned-operator adjoint recursion, network included) vs one
exact F1 gradient (forward + adjoint) and one forward run, on the DENSE-obs WEAK problem (200 test cases, 1 core,
sigma_q 0.1 / sqrt 10, checkpoints tmp/meta/m5dw_f1/*_c8a_s<seed>.pt). Same method as measure_cell8_cost.py.
Writes tmp/results/dense/cell8_cost_dense.json.

Usage (project root, env oceanai):  python src/measure_cell8_cost_dense.py
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_dense_v2 as D  # noqa: E402
from dameta.meta.learned_gradient import LearnedGradient, analytic_grad, forward_run  # noqa: E402
from measure_cell8_cost import timed  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    P = D.setup("l63v2d", "f1", "test", "weak", 0.0316227766)
    cost = P["cost"]
    n = len(P["x0_b"])
    c = cost.pack(P["x0_b"], torch.zeros(n, cost.cfg.n_obs - 1, 3))
    t_fwd = timed(lambda: forward_run(cost, c))
    t_grad = timed(lambda: cost.grad(c))
    out = {"forward_s": t_fwd, "gradient_s": t_grad, "gradient_over_forward": t_grad / t_fwd, "seeds": {}}
    for ck in sorted((ROOT / "tmp/meta/m5dw_f1").glob("*_c8a_s*.pt")):
        lg = LearnedGradient.from_state_dict(torch.load(ck, weights_only=False))
        fwd = forward_run(cost, c)
        t_net = timed(lambda: analytic_grad(cost, c) + lg.grad_obs(c, 0, fwd, cost.weak))
        it8 = t_fwd + t_net
        out["seeds"][ck.stem] = {"cell8_network_s": t_net, "cell8_iteration_s": it8,
                                 "gradient_over_cell8_iteration": t_grad / it8}
        print(f"{ck.stem}: forward {t_fwd * 1e3:.1f} ms | exact gradient {t_grad * 1e3:.1f} ms | network + recursion "
              f"{t_net * 1e3:.1f} ms | iteration {it8 * 1e3:.1f} ms -> gradient / cell-8a iteration {t_grad / it8:.2f}")
    (ROOT / "tmp/results/dense/cell8_cost_dense.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
