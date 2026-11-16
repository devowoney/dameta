"""
Step 1 of the meta-speed check (2026-11-16): how many gradients does a STRONG classical optimizer need to reach the
same 1 % cost target as the M5 table? Gives the floor against which a learned optimizer's speed-up is judged.

D1 v2, strong constraint, BICE, false model F1 or F2, same first guess / hybrid B / obs as run_4dvar_v2.
    lbfgs  per-case torch.optim.LBFGS (history 10, strong-Wolfe line search); every closure call = one gradient
           (forward + adjoint); J recorded after every call
    gn     batched Gauss-Newton: H = B^-1 + sum_i M_i^T H^T R^-1 H M_i (M_i = d x_{t_i} / d x0 from 3 tangent-linear
           runs, forward-mode AD), step = H^-1 grad J, backtracking (halving) on J if the cost increases.
           Per iteration: 1 forward + 1 adjoint (gradient) + 3 TL runs (+ forward runs for backtracking)
Target: J <= 1.01 J*, J* = best final J of classical GD / Adagrad, same model and case (as report_m5.py).
Cost reported as gradient-call counts and as wall-clock per 200-case batch (measured here, one core, against an
Adagrad exact-gradient call in the same session).

Usage (project root, env oceanai):  python src/run_floor_v2.py [--models f1 emulator] [--split test]
Writes tmp/results/floor_v2_<split>.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.autograd.forward_ad as fwAD
import xarray as xr
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_4dvar_v2 as R2  # noqa: E402
from dameta.da import FourDVarCost, run_fourdvar  # noqa: E402
from dameta.da.cost import window_trajectory  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def setup(model, split):
    base = OmegaConf.load(ROOT / "src/configs/fourdvar_v2.yaml")
    del base["hydra"]
    base.data_dir, base.emulator_dir = str(ROOT / "tmp/data"), str(ROOT / "tmp/emulators")
    base.result_group = base.out_dir = "x"
    base.model, base.split = model, split
    return base, R2.setup_problem(base, split)


def j_star(model, split):
    paths = [next((ROOT / f"tmp/results/v2_{model}").glob(f"bice_strong_full_{m}_lr*_{split}.nc"))
             for m in ("adagrad", "gd")]
    return np.nanmin(np.stack([xr.load_dataset(p).J_iter.values[:, -1] for p in paths]), 0)


def iters_to(jh, js):
    hit = jh <= 1.01 * js[:, None]
    return np.where(hit.any(1), hit.argmax(1), np.nan)


# ---- L-BFGS (per case) ------------------------------------------------------------------------------------------
def run_lbfgs(cost: FourDVarCost, x0_b, max_evals=60):
    n = len(x0_b)
    hist = np.full((n, max_evals + 1), np.nan)
    x_a = x0_b.clone()
    for i in range(n):
        ci = FourDVarCost(cost.step, cost.y[i:i + 1], cost.mask, cost.x0_b[i:i + 1], cost.cfg, b_inv=cost.b_inv[i:i + 1])
        x = x0_b[i:i + 1].clone().requires_grad_(True)
        opt = torch.optim.LBFGS([x], lr=1.0, max_iter=max_evals, max_eval=max_evals, history_size=10,
                                tolerance_grad=1e-12, tolerance_change=1e-14, line_search_fn="strong_wolfe")
        js = []

        def closure():
            opt.zero_grad()
            j = ci(x)
            j.sum().backward()
            js.append(float(j))
            return j.sum()

        opt.step(closure)
        js = np.minimum.accumulate(np.array(js[:max_evals]))   # best J reached after k gradient calls
        hist[i, 1:len(js) + 1] = js
        hist[i, len(js) + 1:] = js[-1]
        x_a[i] = x.detach()[0]
    with torch.no_grad():
        hist[:, 0] = cost(x0_b).numpy()
    return hist, x_a          # hist[:, k] = best J after k gradient calls (k = 0: first guess)


# ---- Gauss-Newton (batched) ---------------------------------------------------------------------------------------
def tl_jacobians(cost: FourDVarCost, x0):
    """[n, n_obs, 3, 3] d x_{t_i} / d x0 by 3 forward-mode TL runs (batched as 3 n cases)."""
    n = x0.shape[0]
    xr_ = x0.repeat(3, 1)
    tang = torch.eye(3, dtype=x0.dtype).repeat_interleave(n, 0)
    with fwAD.dual_level():
        traj = window_trajectory(cost.step, fwAD.make_dual(xr_, tang), cost.cfg)
        jt = fwAD.unpack_dual(traj[:, cost.t_obs]).tangent              # [3n, n_obs, 3]
    return jt.reshape(3, n, -1, 3).permute(1, 2, 3, 0)                  # [n, n_obs, 3 (state), 3 (x0 dir)]


def run_gn(cost: FourDVarCost, x0_b, max_iter=30):
    n = len(x0_b)
    x = x0_b.clone()
    rinv = 1.0 / cost.cfg.sigma_o2
    hm = cost.mask                                                      # [n_obs, 3]
    hist_grad = [cost(x0_b).detach().numpy()]                           # J after k gradient-equivalent iterations
    n_fwd_bt = 0
    for _ in range(max_iter):
        j, g = cost.grad(x)
        m = tl_jacobians(cost, x)                                       # [n, n_obs, 3, 3]
        hmm = m * hm[None, :, :, None]
        h = cost.b_inv + rinv * torch.einsum("nosi,nosj->nij", hmm, hmm)
        dx = torch.linalg.solve(h, g)
        a = torch.ones(n, dtype=x.dtype)
        with torch.no_grad():
            for _bt in range(10):                                       # backtracking: halve where J increases
                jn = cost(x - a[:, None] * dx)
                n_fwd_bt += 1
                bad = ~(jn <= j) & (a > 1e-3)
                if not bad.any():
                    break
                a = torch.where(bad, a / 2, a)
            x = x - a[:, None] * dx
            hist_grad.append(torch.minimum(cost(x), torch.tensor(hist_grad[-1])).numpy())
    return np.stack(hist_grad, 1), x, n_fwd_bt / max_iter


def timed(fn, reps=10):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["f1", "emulator"])
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    out = {}
    for model in a.models:
        cfg, P = setup(model, a.split)
        cost, xb = P["cost"], P["x0_b"]
        js = j_star(model, a.split)
        res = {}
        # reference re-run (Adagrad lr 0.3, exact gradient) for a same-session baseline
        ref = run_fourdvar(cost, xb, "adagrad", 0.3, 1000, 1e-6)
        it_ref = iters_to(ref["J_iter"], js)
        res["adagrad_ref"] = {"grads_to_target_median": float(np.nanmedian(it_ref)),
                              "reached": float(np.isfinite(it_ref).mean())}
        h_lb, x_lb = run_lbfgs(cost, xb)
        it = iters_to(h_lb, js)
        res["lbfgs"] = {"grads_to_target_median": float(np.nanmedian(it)), "reached": float(np.isfinite(it).mean()),
                        "grads_to_target_q25_q75": [float(np.nanquantile(it, q)) for q in (0.25, 0.75)]}
        h_gn, x_gn, bt = run_gn(cost, xb)
        it = iters_to(h_gn, js)
        res["gauss_newton"] = {"iters_to_target_median": float(np.nanmedian(it)),
                               "reached": float(np.isfinite(it).mean()), "backtrack_fwd_per_iter": bt,
                               "iters_to_target_q25_q75": [float(np.nanquantile(it, q)) for q in (0.25, 0.75)]}
        # IC quality of the floors (vs truth)
        xt = P["ds"].x_true.values[:, 0]
        for k, xa in (("lbfgs", x_lb), ("gauss_newton", x_gn)):
            res[k]["rmse_x0_median"] = float(np.median(np.sqrt(((xa.numpy() - xt) ** 2).mean(1))))
            res[k]["worse_basin_vs_ref"] = int((cost(xa).detach().numpy() > 1.1 * ref["J_iter"][:, -1]).sum())
        res["adagrad_ref"]["rmse_x0_median"] = float(np.median(np.sqrt(((ref["x0_a"] - xt) ** 2).mean(1))))
        # same-session timings, 200 cases, one core
        t_grad = timed(lambda: cost.grad(xb))
        t_fwd = timed(lambda: cost(xb))
        t_tl = timed(lambda: tl_jacobians(cost, xb))
        res["timing_s_per_batch"] = {"gradient": t_grad, "forward": t_fwd, "tl_jacobians_3dirs": t_tl,
                                     "gn_iteration": t_grad + t_tl + bt * t_fwd}
        g = res["adagrad_ref"]["grads_to_target_median"] * t_grad
        res["time_to_target_s"] = {
            "adagrad_ref": g,
            "lbfgs": res["lbfgs"]["grads_to_target_median"] * t_grad,
            "gauss_newton": res["gauss_newton"]["iters_to_target_median"] * res["timing_s_per_batch"]["gn_iteration"]}
        out[model] = res
        print(model, json.dumps(res, indent=1), flush=True)
    (ROOT / f"tmp/results/floor_v2_{a.split}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
