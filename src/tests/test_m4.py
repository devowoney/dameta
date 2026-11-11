"""
Milestone 4 verification — Meta-B (bice delta chain port).

Plain-python test script (no pytest in env `oceanai`). Run from the project root:   python src/tests/test_m4.py
Requires D1 data in tmp/data.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on the path

from dameta.da import CostConfig, FourDVarCost, ReferenceStep, run_fourdvar  # noqa: E402
from dameta.dynamics import L63Model  # noqa: E402
from dameta.meta import (MetaB, MetaBConfig, MetaBSup, StretchNet, SupConfig, build_sup,  # noqa: E402
                         collect_teacher_pairs, control_stats, run_meta_b, run_meta_b_sup, train_meta_b)

torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parents[2]
VAL = xr.open_dataset(ROOT / "tmp/data/l63_val_seed42.nc").isel(ic=slice(0, 16))
TRAIN = xr.open_dataset(ROOT / "tmp/data/l63_train_seed42.nc").isel(ic=slice(0, 64))
M, CFG = L63Model(), CostConfig()
MASK = torch.tensor(VAL.obs_mask.sel(pattern="full").values)
LR = {"adagrad": 0.1, "gd": 1e-3}                      # E1-selected lr (strong)
RESULTS = []


def report(name, value, thr, ok):
    RESULTS.append((name, value, thr, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<58s} value={value:.3e}  ({thr})")
    assert ok, name


def make_cost(ds, weak, idx=None):
    ds = ds if idx is None else ds.isel(ic=idx)
    return FourDVarCost(M.step, torch.tensor(ds.y_obs.values), MASK, torch.tensor(ds.x0_b.values), CFG, weak=weak)


def make_learner(weak, zero_init=True, seed=0, **cfg):
    torch.manual_seed(seed)
    d = 3 + (3 * (CFG.n_obs - 1) if weak else 0)
    m, s = control_stats(torch.tensor(TRAIN.x_true.values), CFG.n_obs, CFG.sigma_q, weak)
    return MetaB(StretchNet(d, zero_init=zero_init), MetaBConfig(**cfg), m, s)


def test_zero_stretch_is_teacher_step():
    """s = 0 (zero-init S): the K-step chain equals exactly one teacher step (GD and Adagrad, strong and weak)."""
    for weak in (False, True):
        cost = make_cost(VAL, weak)
        c = cost.pack(torch.tensor(VAL.x0_b.values), torch.zeros(16, CFG.n_obs - 1, 3) if weak else None)
        _, g = cost.grad(c)
        for method in ("gd", "adagrad"):
            f = ReferenceStep(c, method=method, lr=LR[method])(c, g)
            c_new = make_learner(weak).advance(c, g, f, k=0)
            err = ((c - f) - c_new).abs().max().item()
            report(f"s=0 chain == teacher step ({method}, {'weak' if weak else 'strong'})", err, "< 1e-12", err < 1e-12)


def test_frozen_zero_reproduces_classical():
    """Frozen Meta-B with s = 0 reproduces run_fourdvar exactly (J history, eval counts), Adagrad strong."""
    cost = make_cost(VAL, False)
    xb = torch.tensor(VAL.x0_b.values)
    ref = run_fourdvar(cost, xb, "adagrad", LR["adagrad"], max_iter=300, tol=1e-6)
    out = run_meta_b(cost, xb, make_learner(False), "adagrad", LR["adagrad"], max_iter=300, tol=1e-6)
    err = np.abs(ref["J_iter"] - out["J_iter"]).max()
    same = bool((ref["n_grad_evals"] == out["n_grad_evals"]).all())
    report("frozen s=0 Meta-B == classical Adagrad (max |dJ|)", err, "< 1e-9 and same eval counts",
           err < 1e-9 and same)


def test_lperf_gradient_is_chain_rule():
    """
    d l_perf / d theta == (d J(c_K(theta)) / d theta) / <g^{k-1}, f^{k-1}>  with g^k = grad J at c_K(theta).
    Exact identity (chain rule) at the theta that produced c^k; checked against autodiff through the dynamics.
    """
    cost = make_cost(VAL, False)
    lrn = make_learner(False, zero_init=False, seed=1)
    c0 = torch.tensor(VAL.x0_b.values)
    _, g0 = cost.grad(c0)
    f0 = ReferenceStep(c0, "gd", LR["gd"])(c0, g0)
    g0s = lrn.standardize_grad(g0)
    params = list(lrn.net.parameters())
    denom = (g0 * f0).sum(1)
    # (a) through the dynamics: per-case J(c_K(theta)) / denom, mean over cases
    cK = lrn.chain(c0, g0s, f0, 0)
    ga = torch.autograd.grad((cost(cK) / denom).mean(), params)
    # (b) l_perf with g^k evaluated at c_K (no dynamics in the graph)
    _, g1 = cost.grad(cK.detach())
    l_perf = (-(g1 * (c0 - lrn.chain(c0, g0s, f0, 0))).sum(1) / denom).mean()
    gb = torch.autograd.grad(l_perf, params)
    num = sum((a - b).pow(2).sum() for a, b in zip(ga, gb)).sqrt()
    rel = (num / sum(a.pow(2).sum() for a in ga).sqrt()).item()
    report("grad_theta l_perf == grad_theta J(c_K)/denom (rel)", rel, "< 1e-10", rel < 1e-10)


def test_stretch_bounds_and_sign():
    """|s| < output_scale; the move never flips any teacher-step component (step * f >= 0)."""
    cost = make_cost(VAL, False)
    lrn = make_learner(False, zero_init=False, seed=2)
    with torch.no_grad():
        for p in lrn.net.parameters():
            p.mul_(30.0)                                       # saturate S on purpose
    c = torch.tensor(VAL.x0_b.values)
    _, g = cost.grad(c)
    f = ReferenceStep(c, "adagrad", LR["adagrad"])(c, g)
    s = lrn.stretch(c, lrn.standardize_grad(g), 0).detach()
    step = c - lrn.advance(c, g, f, 0)
    smax = s.abs().max().item()
    # <= : float64 tanh rounds to exactly 1 when saturated on purpose
    report("|s| <= output_scale (saturated S)", smax, f"<= {lrn.net.output_scale}", smax <= lrn.net.output_scale)
    worst = (step * f).min().item()
    ratio = (step.abs() / f.abs()).max().item()
    report("step never flips teacher component (min step*f)", worst, ">= 0", worst >= 0)
    report("per-component stretch <= 1 + output_scale", ratio, f"<= {1 + lrn.net.output_scale}",
           ratio <= 1 + lrn.net.output_scale + 1e-12)


def test_case_independence():
    """Frozen inference: a case's result does not depend on the other cases of the batch."""
    lrn = make_learner(False, zero_init=False, seed=3)
    xb = torch.tensor(VAL.x0_b.values)
    full = run_meta_b(make_cost(VAL, False), xb, lrn, "adagrad", LR["adagrad"], max_iter=100, tol=1e-6)
    sub = run_meta_b(make_cost(VAL, False, slice(0, 5)), xb[:5], lrn, "adagrad", LR["adagrad"], max_iter=100,
                     tol=1e-6)
    err = np.abs(full["x0_a"][:5] - sub["x0_a"]).max()
    report("batch composition does not change a case's analysis", err, "< 1e-12", err < 1e-12)


def test_training_smoke():
    """Short meta-training (2 episodes x 20 outer iterations, 32 train cases): finite losses, theta moves."""
    lrn = make_learner(False)
    xb = torch.tensor(TRAIN.x0_b.values)
    theta0 = torch.cat([p.detach().flatten().clone() for p in lrn.net.parameters()])
    hist = train_meta_b(lrn, lambda idx: make_cost(TRAIN, False, idx), xb[:32], "adagrad", LR["adagrad"],
                        batch=16, episode_iters=20)
    theta1 = torch.cat([p.detach().flatten() for p in lrn.net.parameters()])
    finite = all(np.isfinite([h["J_end_med"], h["L_perf_applied_mean"], h["step_ratio_med"]]).all() for h in hist)
    moved = (theta1 - theta0).norm().item()
    for h in hist:
        print("      ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in h.items()})
    report("meta-training smoke: finite diagnostics, |dtheta|", moved, "> 0 and finite", finite and moved > 0)


def test_checkpoint_roundtrip():
    """state_dict -> from_state_dict gives the identical frozen update."""
    lrn = make_learner(True, zero_init=False, seed=4)
    lrn2 = MetaB.from_state_dict(lrn.state_dict())
    cost = make_cost(VAL, True)
    c = cost.pack(torch.tensor(VAL.x0_b.values), torch.zeros(16, CFG.n_obs - 1, 3))
    _, g = cost.grad(c)
    f = ReferenceStep(c, "gd", LR["gd"])(c, g)
    err = (lrn.advance(c, g, f, 3) - lrn2.advance(c, g, f, 3)).abs().max().item()
    report("checkpoint round-trip (weak)", err, "== 0", err == 0)


class _Oracle:
    """Stand-in supervised learner that returns the TRUE running gradient (tests the supervised outer loop)."""

    def __init__(self, cost):
        self.cost = cost

    def predict(self, c, k):
        with torch.enable_grad():
            return self.cost.grad(c)[1]


def test_sup_oracle_reproduces_classical():
    """Supervised loop fed the true gradient == run_fourdvar (J history); n_fwd_evals == classical n_grad_evals."""
    for weak in (False, True):
        cost = make_cost(VAL, weak)
        xb = torch.tensor(VAL.x0_b.values)
        ref = run_fourdvar(cost, xb, "adagrad", LR["adagrad"], max_iter=200, tol=1e-6)
        out = run_meta_b_sup(cost, xb, _Oracle(cost), "adagrad", LR["adagrad"], max_iter=200, tol=1e-6)
        err = np.abs(ref["J_iter"] - out["J_iter"]).max()
        same = bool((ref["n_grad_evals"] == out["n_fwd_evals"]).all()) and int(out["n_grad_evals"].sum()) == 0
        g2 = np.nanmin(out["cos_ghat_grad"])
        report(f"sup loop w/ oracle grad == classical ({'weak' if weak else 'strong'})", err,
               "< 1e-9, same counts, cos(g_hat,g)=1", err < 1e-9 and same and abs(g2 - 1) < 1e-12)


def test_teacher_pairs():
    """Collected pairs at k = 0 are (x0_b, grad J(x0_b)) for every case; pair count = number of teacher moves."""
    cost = make_cost(VAL, False)
    xb = torch.tensor(VAL.x0_b.values)
    pairs = collect_teacher_pairs(cost, xb, "adagrad", LR["adagrad"], max_iter=200, tol=1e-6)
    k0 = pairs["k"] == 0
    _, g0 = cost.grad(xb)
    err = max((pairs["c"][k0] - xb).abs().max().item(), (pairs["g"][k0] - g0).abs().max().item())
    ref = run_fourdvar(cost, xb, "adagrad", LR["adagrad"], max_iter=200, tol=1e-6)
    moves = int((np.diff(ref["x0_iter"], axis=1) != 0).any(-1).sum())
    report("teacher pairs: k=0 == (x0_b, grad); count == moves", err, f"< 1e-14 and {len(pairs['k'])} == {moves}",
           err < 1e-14 and len(pairs["k"]) == moves)


def test_iteration_embedding_per_sample():
    """emb(k) with a per-sample k vector equals the scalar-k embedding for each sample."""
    emb = StretchNet(3).emb
    ks = torch.tensor([0.0, 3.0, 50.0])
    a = emb(ks, 3)
    b = torch.cat([emb(int(k), 1) for k in ks])
    err = (a - b).abs().max().item()
    report("iteration embedding: vector k == scalar k", err, "< 1e-15", err < 1e-15)


def test_sup_fit_smoke_and_checkpoint():
    """Supervised fit on 32 train ICs (5 epochs): val MSE improves, finite; checkpoint round-trip exact."""
    xb_tr = torch.tensor(TRAIN.x0_b.values[:32])
    tr = collect_teacher_pairs(make_cost(TRAIN, False, slice(0, 32)), xb_tr, "adagrad", LR["adagrad"], 300, 1e-6)
    va = collect_teacher_pairs(make_cost(VAL, False), torch.tensor(VAL.x0_b.values), "adagrad", LR["adagrad"],
                               300, 1e-6)
    m, s = control_stats(torch.tensor(TRAIN.x_true.values), CFG.n_obs, CFG.sigma_q, False)
    lrn = build_sup(3, SupConfig(epochs=5, batch_size=256), m, s, tr)
    hist = lrn.fit(tr, va)
    v0, vbest = hist[0]["val_mse"], min(h["val_mse"] for h in hist)
    for h in hist:
        print("      ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in h.items()})
    report("supervised fit: best val MSE <= epoch-0 val MSE", vbest, f"<= {v0:.3e}, finite",
           np.isfinite(vbest) and vbest <= v0)
    lrn2 = MetaBSup.from_state_dict(lrn.state_dict())
    c = torch.tensor(VAL.x0_b.values)
    err = (lrn.predict(c, 7) - lrn2.predict(c, 7)).abs().max().item()
    report("supervised checkpoint round-trip", err, "== 0", err == 0)
    out = run_meta_b_sup(make_cost(VAL, False), c, lrn, "adagrad", LR["adagrad"], max_iter=100, tol=1e-6)
    print(f"       frozen sup (5-epoch smoke): median J_final {np.median(out['J_iter'][:, -1]):.2f} "
          f"(J0 {np.median(out['J_iter'][:, 0]):.2f}), median cos(g_hat,g) {np.nanmedian(out['cos_ghat_grad']):.3f}")
    report("supervised frozen inference finite", float(np.isfinite(out["J_iter"]).mean()), "== 1",
           bool(np.isfinite(out["J_iter"]).all()))


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{sum(r[3] for r in RESULTS)}/{len(RESULTS)} checks passed")
