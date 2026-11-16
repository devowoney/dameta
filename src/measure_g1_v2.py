"""
G1 gradient fidelity on D1 v2 (M5): similarity of each method's 4D-Var sensitivity to the TRUE-dynamics sensitivity.

    G1 = cos(g_method(c), grad J_true(c))   and   |g_method - grad J_true| / |grad J_true|
with J_true the cost of the same problem (same obs, first guess, hybrid B, Q) with the TRUE model (L63, rho 28),
evaluated at the same control c: the first guess (x0_b[, eta = 0]) and the true initial state (x0_t[, eta = 0]).
Test split, medians over 200 cases (learned gradients: median over cases, mean over 3 seeds).

Methods: F1 exact adjoint (= GD, Adagrad, FD gradient, cells 1/2/5 on F1, which consume it), F1 weak constraint,
F2 emulator adjoint (= cells 1/2/5 on F2), hybrid (F1 problem, emulator adjoint), BPE (x0-gradient of L63 with the
BPE-estimated parameters theta_a, and with the prior theta_b = F1), cell 8 v1 (k = 0) and cell 8a.

Usage (project root, env oceanai):  python src/measure_g1_v2.py   -> tmp/results/g1_v2.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_4dvar_v2 as R2  # noqa: E402
from dameta.da import FourDVarCost  # noqa: E402
from dameta.dynamics.emulator import L63Emulator  # noqa: E402
from dameta.meta.learned_gradient import LearnedGradient, analytic_grad, forward_run  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def exact_grad(cost, c):
    return cost.grad(c)[1]


def scores(g, g_t):
    cos = torch.nn.functional.cosine_similarity(g, g_t, dim=1)
    rel = (g - g_t).norm(dim=1) / g_t.norm(dim=1)
    return {"cos_median": float(cos.median()), "cos_q25": float(cos.quantile(0.25)),
            "frac_cos_gt_0.9": float((cos > 0.9).double().mean()), "rel_err_median": float(rel.median())}


def main():
    torch.set_default_dtype(torch.float64)
    base = OmegaConf.load(ROOT / "src/configs/fourdvar_v2.yaml")
    del base["hydra"]
    base.data_dir, base.emulator_dir = str(ROOT / "tmp/data"), str(ROOT / "tmp/emulators")
    base.result_group = base.out_dir = "x"
    base.split = "test"
    out = {}

    def setup(model, constraint="strong"):
        cfg = base.copy()
        cfg.model, cfg.constraint = model, constraint
        P = R2.setup_problem(cfg, "test")
        true_cost = FourDVarCost(R2.truth_step(P["ds"]), P["y"], P["mask"], P["x0_b"], P["ccfg"], weak=P["weak"],
                                 b_inv=P["b_inv"])
        n = len(P["x0_b"])
        eta0 = torch.zeros(n, P["ccfg"].n_obs - 1, 3) if P["weak"] else None
        x0_t = torch.tensor(P["ds"].x_true.values[:, 0])
        pts = {"fg": P["cost"].pack(P["x0_b"], eta0), "truth": P["cost"].pack(x0_t, eta0)}
        return cfg, P, true_cost, pts

    def record(name, gfun, true_cost, pts):
        out[name] = {p: scores(gfun(c), exact_grad(true_cost, c)) for p, c in pts.items()}
        print(f"{name:28s} " + " | ".join(f"{p}: cos {v['cos_median']:.4f} rel {v['rel_err_median']:.3f}"
                                         for p, v in out[name].items()), flush=True)

    # E1: true model (identity check)
    cfg, P, tc, pts = setup("f1")
    record("E1 truth", lambda c: exact_grad(tc, c), tc, pts)
    # F1 exact adjoint (Adagrad, GD, FD, cells 1/2/5 on F1)
    record("F1 exact", lambda c: exact_grad(P["cost"], c), tc, pts)
    # hybrid: F1 problem, gradient from the emulator's adjoint
    emu = L63Emulator.load(Path(cfg.emulator_dir) / f"{cfg.surrogate_emulator}.pt")
    for p in emu.parameters():
        p.requires_grad_(False)
    sur = FourDVarCost(emu.step, P["y"], P["mask"], P["x0_b"], P["ccfg"], weak=False, b_inv=P["b_inv"])
    record("hybrid (emulator adjoint)", lambda c: exact_grad(sur, c), tc, pts)
    # BPE: x0 sensitivity of L63 with the estimated parameters theta_a (analysis) — prior theta_b = F1 row
    rec = xr.load_dataset(ROOT / "tmp/results/v2_f1_bpe/bpe_strong_full_adagrad_lr0.1_test.nc")
    theta_a = torch.tensor(rec.theta_a.values)
    from dameta.da.cost import ParamFourDVarCost  # noqa: E402
    pc = ParamFourDVarCost(P["y"], P["mask"], P["x0_b"], P["ccfg"], torch.tensor([10.0, 27.5, 8 / 3]),
                           torch.ones(3), P["ds"].attrs["dt"], P["b_inv"])
    record("BPE (theta_a)", lambda c: exact_grad(pc, torch.cat([c, theta_a], 1))[:, :3], tc, pts)
    # cell 8 v1 / 8a on F1
    for tag in ("c8", "c8a"):
        per_seed = []
        for s in range(3):
            lg = LearnedGradient.from_state_dict(torch.load(
                ROOT / f"tmp/meta/m5_f1/learned_grad_strong_full_adagrad_lr0.3_{tag}_s{s}.pt", weights_only=False))
            g = lambda c: analytic_grad(P["cost"], c) + lg.grad_obs(c, 0, forward_run(P["cost"], c), False)  # noqa
            record(f"F1 {tag} s{s}", g, tc, pts)
            per_seed.append(out.pop(f"F1 {tag} s{s}"))
        out[f"F1 {tag}"] = {p: {k: float(np.mean([d[p][k] for d in per_seed])) for k in per_seed[0][p]}
                            for p in pts}
    # F1 weak constraint
    cfg_w, P_w, tc_w, pts_w = setup("f1", "weak")
    record("F1 weak exact", lambda c: exact_grad(P_w["cost"], c), tc_w, pts_w)

    # F2 emulator (own first guess / B)
    cfg, P, tc, pts = setup("emulator")
    record("F2 exact (emulator adjoint)", lambda c: exact_grad(P["cost"], c), tc, pts)
    for tag in ("c8", "c8a"):
        per_seed = []
        for s in range(3):
            lg = LearnedGradient.from_state_dict(torch.load(
                ROOT / f"tmp/meta/m5_emulator/learned_grad_strong_full_adagrad_lr0.3_{tag}_s{s}.pt",
                weights_only=False))
            g = lambda c: analytic_grad(P["cost"], c) + lg.grad_obs(c, 0, forward_run(P["cost"], c), False)  # noqa
            record(f"F2 {tag} s{s}", g, tc, pts)
            per_seed.append(out.pop(f"F2 {tag} s{s}"))
        out[f"F2 {tag}"] = {p: {k: float(np.mean([d[p][k] for d in per_seed])) for k in per_seed[0][p]}
                            for p in pts}
    for k in ("F1 c8", "F1 c8a", "F2 c8", "F2 c8a"):
        print(f"{k:28s} " + " | ".join(f"{p}: cos {v['cos_median']:.4f} rel {v['rel_err_median']:.3f}"
                                       for p, v in out[k].items()))
    (ROOT / "tmp/results/g1_v2.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
