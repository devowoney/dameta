"""
Dense-observation test (pilot request 2026-11-16): same 1.1-TU window, obs at EVERY step (111 obs, D1 v2-dense)
vs the D1 v2 baseline (obs every 10 steps, 12 obs). Strong constraint, full obs, BICE, classical optimizers only.

Estimation models: f1 (L63 rho 27.5, operational false model) and truth (E1, perfect-model reference); same first
guess and hybrid B as M5 (F1 lagged background) for both.
Optimizers (all with the exact adjoint gradient):
    adagrad, gd   lr by the pre-registered val rule (select_lr.select); sparse lr = the M5 selection
                  (f1: adagrad 0.3, gd 0.001; truth: adagrad 1.0, gd 0.001); dense lr = val sweep here (grid extended
                  once at an edge)
    lbfgs         per case, history 10, strong-Wolfe (run_floor_v2 settings); 1 unit = 1 gradient call
    gn            batched Gauss-Newton (run_floor_v2.run_gn logic); 1 unit = 1 iteration
                  (= 1 gradient + 3 TL runs + backtracking forwards; converted to gradient-equivalents by timing)
Record per (dataset, model, method) on `split`: tmp/results/dense/<dataset>_<model>_<method>_<split>.nc
    J_hist (ic, k)       cost after k units (k = 0: first guess); best-so-far for lbfgs
    x_hist (ic, k, var)  iterate after k units
    x0_a (ic, var), unit_grad_eq (attr: gradient-equivalents per unit), lr (attr)
Timing (one core, 200 cases batched, same session): tmp/results/dense/timing_<dataset>_<model>.json

Weak constraint (2026-11-16, --constraint weak): control (x0, eta_1..eta_{n_obs-1}), eta added at EVERY obs time, so
the dense control has 3 + 330 entries (sparse 3 + 33). --sigma_q sets Q = sigma_q^2 I per increment (pre-registered
0.1; 0.1 / sqrt(10) keeps the model-error variance per 0.1 TU of the sparse problem). Gauss-Newton is not run (its
Hessian would need 333 TL directions per iteration). Records also hold control_a (final packed control).
Output tag: <dataset>_<model>[_weak_q<sigma_q>]

Usage (project root, env oceanai):  python src/run_dense_v2.py --model f1 [--datasets l63v2 l63v2d] [--split test]
                                    python src/run_dense_v2.py --model f1 --constraint weak [--sigma_q 0.1]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_4dvar_v2 as R2  # noqa: E402
from dameta.da import FourDVarCost, run_fourdvar  # noqa: E402
from run_floor_v2 import timed, tl_jacobians  # noqa: E402
from select_lr import select  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tmp/results/dense"
SPARSE_LR = {"strong": {"f1": {"adagrad": 0.3, "gd": 0.001}, "truth": {"adagrad": 1.0, "gd": 0.001}},
             "weak": {"f1": {"adagrad": 0.1, "gd": 0.001}, "truth": {"adagrad": 0.1, "gd": 0.0003}}}   # M5 val rule
GRID = {"adagrad": [0.03, 0.1, 0.3, 1.0, 3.0], "gd": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3], "lbfgs": [0.1, 0.3, 0.5, 1.0]}
MAX_ITER, TOL = 1000, 1e-6


def setup(dataset, model, split, constraint="strong", sigma_q=0.1):
    base = OmegaConf.load(ROOT / "src/configs/fourdvar_v2.yaml")
    del base["hydra"]
    base.data_dir, base.emulator_dir = str(ROOT / "tmp/data"), str(ROOT / "tmp/emulators")
    base.result_group = base.out_dir = "x"
    base.data_prefix, base.model, base.split = dataset, model, split
    base.background = "f1"
    base.constraint, base.cost.sigma_q = constraint, float(sigma_q)
    return R2.setup_problem(base, split)


# ---- optimizers: each returns J_hist [n, K + 1], x_hist [n, K + 1, 3] -----------------------------------------------
def opt_classical(cost, xb, method, lr):
    r = run_fourdvar(cost, xb, method, lr, MAX_ITER, TOL)
    return r["J_iter"], r["x0_iter"], r["control_final"]


def opt_lbfgs(cost: FourDVarCost, xb, max_evals=100):
    n = len(xb)
    eta0 = torch.zeros(n, cost.cfg.n_obs - 1, 3, dtype=xb.dtype) if cost.weak else None
    c0 = cost.pack(xb, eta0)                                  # packed control (weak: x0 then eta, first guess eta 0)
    jh, xh = np.full((n, max_evals + 1), np.nan), np.full((n, max_evals + 1, 3), np.nan)
    c_a = c0.numpy().copy()
    with torch.no_grad():
        jh[:, 0], xh[:, 0] = cost(*cost.unpack(c0)).numpy(), xb.numpy()
    for i in range(n):
        ci = FourDVarCost(cost.step, cost.y[i:i + 1], cost.mask, cost.x0_b[i:i + 1], cost.cfg,
                          weak=cost.weak, b_inv=cost.b_inv[i:i + 1])
        x = c0[i:i + 1].clone().requires_grad_(True)
        opt = torch.optim.LBFGS([x], lr=1.0, max_iter=max_evals, max_eval=max_evals, history_size=10,
                                tolerance_grad=1e-12, tolerance_change=1e-14, line_search_fn="strong_wolfe")
        js, xs = [], []

        def closure():
            opt.zero_grad()
            j = ci(*ci.unpack(x))
            j.sum().backward()
            js.append(float(j.detach()))
            xs.append(x.detach()[0].clone().numpy())
            return j.sum()

        opt.step(closure)
        js, xs = np.array(js[:max_evals]), np.array(xs[:max_evals])
        best = np.minimum.accumulate(js)                      # best point found after k gradient calls
        arg = np.array([int(np.argmin(js[:k + 1])) for k in range(len(js))])
        jh[i, 1:len(js) + 1], xh[i, 1:len(js) + 1] = best, xs[arg][:, :3]
        jh[i, len(js) + 1:], xh[i, len(js) + 1:] = best[-1], xs[arg[-1]][:3]
        c_a[i] = xs[arg[-1]]
    return jh, xh, c_a


def opt_gn(cost: FourDVarCost, xb, max_iter=30):
    x = xb.clone()
    rinv = 1.0 / cost.cfg.sigma_o2
    with torch.no_grad():
        jh, xh = [cost(xb).numpy()], [xb.numpy()]
    n_bt = 0
    for _ in range(max_iter):
        j, g = cost.grad(x)
        m = tl_jacobians(cost, x) * cost.mask[None, :, :, None]
        h = cost.b_inv + rinv * torch.einsum("nosi,nosj->nij", m, m)
        dx = torch.linalg.solve(h, g)
        a = torch.ones(len(x), dtype=x.dtype)
        with torch.no_grad():
            for _bt in range(10):
                jn = cost(x - a[:, None] * dx)
                n_bt += 1
                bad = ~(jn <= j) & (a > 1e-3)
                if not bad.any():
                    break
                a = torch.where(bad, a / 2, a)
            x = x - a[:, None] * dx
            jh.append(cost(x).numpy())
            xh.append(x.numpy())
    return np.stack(jh, 1), np.stack(xh, 1), n_bt / max_iter


def write(path, jh, xh, attrs, c_a=None, grad_cum=None):
    c_a = xh[:, -1] if c_a is None else c_a
    dv = {"J_hist": (("ic", "k"), jh), "x_hist": (("ic", "k", "var"), xh),
          "x0_a": (("ic", "var"), xh[:, -1]), "control_a": (("ic", "ctrl"), c_a)}
    if grad_cum is not None:                     # cumulative gradient evaluations per iterate (lbfgs_wolfe)
        dv["grad_cum"] = (("ic", "k"), grad_cum)
    xr.Dataset(dv, coords={"var": ["x", "y", "z"]}, attrs=attrs).to_netcdf(path)


def sweep_lr(dataset, model, method, constraint="strong", sigma_q=0.1):
    """Pre-registered rule on val; grid extended once at the selected edge."""
    P = setup(dataset, model, "val", constraint, sigma_q)
    grid, rows = list(GRID[method]), {}
    while True:
        for lr in grid:
            if lr in rows:
                continue
            r = run_fourdvar(P["cost"], P["x0_b"], method, lr, MAX_ITER, TOL)
            jf = r["J_iter"][:, -1]
            rows[lr] = {"lr": lr, "diverged_frac": float((~np.isfinite(jf)).mean()),
                        "J_med": float(np.nanmedian(np.where(np.isfinite(jf), jf, np.nan))),
                        "evals_med": float(np.median(r["n_grad_evals"]))}
            print(f"  val sweep {dataset} {model} {method} lr {lr:g}: {rows[lr]}", flush=True)
        best = select(list(rows.values()))["lr"]
        lrs = sorted(rows)
        if best == lrs[0] and len(grid) < len(GRID[method]) + 2:
            grid = [lrs[0] / 3] + grid
        elif best == lrs[-1] and len(grid) < len(GRID[method]) + 2:
            grid = grid + [lrs[-1] * 3]
        else:
            return best, list(rows.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="f1", choices=["f1", "truth"])
    ap.add_argument("--datasets", nargs="+", default=["l63v2", "l63v2d"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--constraint", default="strong", choices=["strong", "weak"])
    ap.add_argument("--sigma_q", type=float, default=0.1)
    ap.add_argument("--lbfgs_evals", type=int, default=100)
    ap.add_argument("--lr_adagrad", type=float, default=None)   # skip the val sweep (lr already selected)
    ap.add_argument("--lr_gd", type=float, default=None)
    # 2026-11-16: classical counterpart of the L-BFGS teacher = batched L-BFGS with a FIXED step (no line search,
    # ReferenceStep method 'lbfgs'); lr by the val rule over GRID['lbfgs']; records <tag>_lbfgsfix_<split>.nc
    ap.add_argument("--only_lbfgsfix", action="store_true")
    # 2026-11-17: consistency check of the strong-Wolfe teacher (ReferenceStep 'lbfgs_wolfe') run alone: must
    # reproduce the reference opt_lbfgs; records <tag>_lbfgswolfecheck_<split>.nc with grad_cum
    ap.add_argument("--only_wolfe_check", action="store_true")
    ap.add_argument("--splits", nargs="+", default=["test", "test_ood"])
    a = ap.parse_args()
    weak = a.constraint == "weak"
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.only_wolfe_check:
        for dataset in a.datasets:
            tag = f"{dataset}_{a.model}" + (f"_weak_q{a.sigma_q:g}" if weak else "")
            for split in a.splits:
                P = setup(dataset, a.model, split, a.constraint, a.sigma_q)
                r = run_fourdvar(P["cost"], P["x0_b"], "lbfgs_wolfe", 1.0, 150, TOL)
                write(OUT / f"{tag}_lbfgswolfecheck_{split}.nc", r["J_iter"], r["x0_iter"],
                      {"lr": 1.0, "unit_grad_eq": 1.0}, r["control_final"], r["grad_cum"])
                print(f"{tag} wolfe check {split} done: median gradient evals {np.median(r['grad_cum'][:, -1])}",
                      flush=True)
        return
    if a.only_lbfgsfix:
        for dataset in a.datasets:
            tag = f"{dataset}_{a.model}" + (f"_weak_q{a.sigma_q:g}" if weak else "")
            lr, sweep = sweep_lr(dataset, a.model, "lbfgs", a.constraint, a.sigma_q)
            print(f"{tag} fixed-step L-BFGS: selected lr {lr}", flush=True)
            for split in a.splits:
                P = setup(dataset, a.model, split, a.constraint, a.sigma_q)
                jh, xh, c_a = opt_classical(P["cost"], P["x0_b"], "lbfgs", lr)
                write(OUT / f"{tag}_lbfgsfix_{split}.nc", jh, xh, {"lr": lr, "unit_grad_eq": 1.0}, c_a)
                print(f"{tag} lbfgsfix {split} done", flush=True)
            (OUT / f"lbfgsfix_{tag}.json").write_text(json.dumps({"lr": lr, "val_sweep": sweep}, indent=1))
        return
    for dataset in a.datasets:
        lrs, sweeps = {}, {}
        for method in ("adagrad", "gd"):
            given = a.lr_adagrad if method == "adagrad" else a.lr_gd
            if given is not None:
                lrs[method] = given
            elif dataset == "l63v2" and a.sigma_q == 0.1:
                lrs[method] = SPARSE_LR[a.constraint][a.model][method]
            else:
                lrs[method], sweeps[method] = sweep_lr(dataset, a.model, method, a.constraint, a.sigma_q)
        print(dataset, a.model, "lr", lrs, flush=True)
        P = setup(dataset, a.model, a.split, a.constraint, a.sigma_q)
        cost, xb = P["cost"], P["x0_b"]
        c0 = cost.pack(xb, torch.zeros(len(xb), cost.cfg.n_obs - 1, 3) if weak else None)
        t_grad = timed(lambda: cost.grad(c0))
        t_fwd = timed(lambda: cost(*cost.unpack(c0)))
        t_tl = timed(lambda: tl_jacobians(cost, xb)) if not weak else float("nan")
        tag = f"{dataset}_{a.model}" + (f"_weak_q{a.sigma_q:g}" if weak else "")
        for method in ("adagrad", "gd"):
            jh, xh, c_a = opt_classical(cost, xb, method, lrs[method])
            write(OUT / f"{tag}_{method}_{a.split}.nc", jh, xh, {"lr": lrs[method], "unit_grad_eq": 1.0}, c_a)
            print(f"{tag} {method} done", flush=True)
        jh, xh, c_a = opt_lbfgs(cost, xb, a.lbfgs_evals)
        write(OUT / f"{tag}_lbfgs_{a.split}.nc", jh, xh, {"unit_grad_eq": 1.0}, c_a)
        print(f"{tag} lbfgs done", flush=True)
        gn_unit = float("nan")
        if not weak:
            jh, xh, bt = opt_gn(cost, xb)
            gn_unit = (t_grad + t_tl + bt * t_fwd) / t_grad
            write(OUT / f"{tag}_gn_{a.split}.nc", jh, xh, {"unit_grad_eq": gn_unit, "backtrack_fwd_per_iter": bt})
            print(f"{tag} gn done", flush=True)
        timing = {"gradient_s": t_grad, "forward_s": t_fwd, "tl_3dirs_s": t_tl, "gn_unit_grad_eq": gn_unit,
                  "lr": lrs, "val_sweeps": sweeps, "n_obs": int(P["ccfg"].n_obs)}
        sfx = "" if a.split == "test" else f"_{a.split}"          # never overwrite the test-split record (val sweeps)
        (OUT / f"timing_{tag}{sfx}.json").write_text(json.dumps(timing, indent=1))
        print(json.dumps(timing, indent=1), flush=True)


if __name__ == "__main__":
    main()
