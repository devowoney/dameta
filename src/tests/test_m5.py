"""
Milestone 5 verification — learned optimization (unrolled free update rule; cells 5 and 7 of the meta grid).

Plain-python test script (no pytest in env `oceanai`). Run from the project root:   python src/tests/test_m5.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dameta.da import CostConfig, FourDVarCost  # noqa: E402
from dameta.dynamics import L63Model  # noqa: E402
from dameta.meta import LearnedOptimization, UnrollConfig, build_unroll, control_stats, run_meta_a  # noqa: E402

torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parents[2]
VAL = xr.open_dataset(ROOT / "tmp/data/l63_val_seed42.nc").isel(ic=slice(0, 8))
TRAIN = xr.open_dataset(ROOT / "tmp/data/l63_train_seed42.nc").isel(ic=slice(0, 64))
M, CFG = L63Model(), CostConfig()
MASK = torch.tensor(VAL.obs_mask.sel(pattern="full").values)
RESULTS = []


def report(name, value, thr, ok):
    RESULTS.append((name, value, thr, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<60s} value={value:.3e}  ({thr})")
    assert ok, name


def make_cost(ds, weak=False, idx=None):
    ds = ds if idx is None else ds.isel(ic=idx)
    return FourDVarCost(M.step, torch.tensor(ds.y_obs.values), MASK, torch.tensor(ds.x0_b.values), CFG, weak=weak)


def learner(use_grad=True, weak=False, zero_init=True, seed=0, **kw):
    d = 3 + (3 * (CFG.n_obs - 1) if weak else 0)
    m, s = control_stats(torch.tensor(TRAIN.x_true.values), CFG.n_obs, CFG.sigma_q, weak)
    return build_unroll(d, UnrollConfig(use_grad=use_grad, seed=seed, **kw), m, s, zero_init=zero_init)


def x0(ds, weak=False):
    xb = torch.tensor(ds.x0_b.values)
    return torch.cat([xb, torch.zeros(xb.shape[0], 3 * (CFG.n_obs - 1))], 1) if weak else xb


def test_zero_init_and_counts():
    for use_grad in (True, False):
        for weak in (False, True):
            lrn, cost = learner(use_grad, weak), make_cost(VAL, weak)
            n0 = cost.n_grad_evals
            r = lrn.unroll(cost, x0(VAL, weak), K=7)
            err = (r["x"] - x0(VAL, weak)).abs().max().item()
            calls = cost.n_grad_evals - n0
            exp = 7 if use_grad else 0
            report(f"zero-init no move; model grad calls ({'grad' if use_grad else 'no-grad'}, "
                   f"{'weak' if weak else 'strong'})", err, f"== 0 and calls {calls} == {exp}",
                   err == 0 and calls == exp and r["n_grad"] == exp)


def _fd_check(second_order):
    lrn = learner(True, zero_init=False, seed=1, K=3, second_order=second_order)
    cost, xs = make_cost(VAL), x0(VAL)
    params = list(lrn.net.parameters())
    loss = lrn.loss(cost, xs)
    grads = torch.autograd.grad(loss, params)
    torch.manual_seed(5)
    v = [torch.randn_like(p) for p in params]
    analytic = sum((g * w).sum() for g, w in zip(grads, v)).item()
    eps = 1e-6
    with torch.no_grad():
        for p, w in zip(params, v):
            p.add_(eps * w)
    lp = lrn.loss(cost, xs).item()
    with torch.no_grad():
        for p, w in zip(params, v):
            p.sub_(2 * eps * w)
    lm = lrn.loss(cost, xs).item()
    return analytic, (lp - lm) / (2 * eps)


def test_second_order_matches_fd():
    a, fd = _fd_check(True)
    rel = abs(a - fd) / abs(fd)
    report("second-order theta-gradient == finite difference (rel)", rel, "< 1e-5", rel < 1e-5)


def test_first_order_is_approximation():
    a, fd = _fd_check(False)
    rel = abs(a - fd) / abs(fd)
    report("first-order theta-gradient differs from FD but same sign", rel, "> 1e-6 and same sign",
           rel > 1e-6 and np.sign(a) == np.sign(fd))


def test_no_grad_inference_has_no_model_call():
    lrn, cost = learner(False, zero_init=False, seed=2), make_cost(VAL)
    n0 = cost.n_grad_evals
    with torch.no_grad():
        lrn.unroll(cost, x0(VAL), K=20)
    report("cell 7 inference: zero gradient evaluations", float(cost.n_grad_evals - n0), "== 0",
           cost.n_grad_evals == n0)


def test_case_independence_and_obs_flag():
    for obs in (False, True):
        lrn = learner(False, zero_init=False, seed=3, obs_input=obs)
        a = lrn.unroll(make_cost(VAL), x0(VAL), K=5)["x"].detach()
        b = lrn.unroll(make_cost(VAL, idx=slice(0, 3)), x0(VAL)[:3], K=5)["x"].detach()
        err = (a[:3] - b).abs().max().item()
        report(f"batch composition does not change a case (obs_input={obs})", err, "< 1e-12", err < 1e-12)
    d_obs = learner(False, obs_input=True).net.d_obs
    report("obs_input appends y + mask (2 x 11 x 3)", float(d_obs), "== 66", d_obs == 66)


def test_training_smoke():
    for use_grad in (True, False):
        lrn = learner(use_grad, K=5, epochs=3, batch=32)
        xb = x0(TRAIN)
        hist = lrn.fit(lambda idx: make_cost(TRAIN, idx=idx), xb, make_cost(VAL), x0(VAL))
        best = min(h["val_ratio"] for h in hist)
        print("      ", [{k: round(v, 4) if isinstance(v, float) else v for k, v in h.items()} for h in hist])
        report(f"training lowers val J(x^K)/J(x^0) below 1 ({'grad' if use_grad else 'no-grad'})", best, "< 1",
               best < 1)


def test_checkpoint_and_record():
    lrn = learner(True, weak=True, zero_init=False, seed=4, K=6)
    lrn2 = LearnedOptimization.from_state_dict(lrn.state_dict())
    cost = make_cost(VAL, True)
    xb = torch.tensor(VAL.x0_b.values)
    r1, r2 = run_meta_a(cost, xb, lrn), run_meta_a(cost, xb, lrn2)
    err = np.abs(r1["J_iter"] - r2["J_iter"]).max()
    shapes = r1["J_iter"].shape == (8, 7) and r1["cos_step_grad"].shape == (8, 6) and r1["eta_final"].shape == (8, 10, 3)
    report("checkpoint round-trip + record shapes (weak, K=6)", float(err), "== 0 and shapes", err == 0 and shapes)


def test_lookahead_pairs_and_fit():
    """Cell 2: pairs reproduce the teacher trajectory (x^0 = x0_b, f = x^k - x^{k+1}, target = x^{k+5}); fit smoke."""
    from dameta.da import run_fourdvar
    from dameta.meta import MetaB, MetaBConfig, StretchNet
    from dameta.meta.meta_b_lookahead import LookaheadConfig, collect_lookahead_pairs, fit_lookahead
    cost, xb = make_cost(VAL), x0(VAL)
    p = collect_lookahead_pairs(cost, xb, "adagrad", 0.1, m=5, max_iter=200)
    ref = torch.tensor(run_fourdvar(cost, xb, "adagrad", 0.1, 200)["x0_iter"])     # [n, 201, 3]
    k = p["k"].long()
    case = torch.cat([torch.nonzero(k == kv).flatten() * 0 + torch.arange(int((k == kv).sum())) for kv in [0]])
    k0 = k == 0
    e0 = (p["x"][k0] - xb).abs().max().item()
    e1 = (p["f"][k0] - (ref[:, 0] - ref[:, 1])).abs().max().item()
    e5 = (p["target"][k0] - ref[:, 5]).abs().max().item()
    report("lookahead pairs: x^0, teacher step, x^{k+5} == classical run", max(e0, e1, e5), "< 1e-12",
           max(e0, e1, e5) < 1e-12 and len(case) > 0)
    m, s = control_stats(torch.tensor(TRAIN.x_true.values), CFG.n_obs, CFG.sigma_q, False)
    torch.manual_seed(0)
    lrn = MetaB(StretchNet(3, output_scale=10.0), MetaBConfig(), m, s)
    tr = collect_lookahead_pairs(make_cost(TRAIN, idx=slice(0, 32)), x0(TRAIN)[:32], "adagrad", 0.1, 5, 200)
    hist = fit_lookahead(lrn, tr, p, LookaheadConfig(epochs=3, batch_size=512))
    print("      ", [{k: round(v, 4) if isinstance(v, float) else v for k, v in h.items()} for h in hist])
    from dameta.meta.meta_b_lookahead import lookahead_loss
    lrn0 = MetaB(StretchNet(3, output_scale=10.0), MetaBConfig(), m, s)          # zero-init = one teacher step
    with torch.no_grad():
        v0 = float(lookahead_loss(lrn0, p, torch.arange(p["x"].shape[0])))
    best = min(h["val_rel"] for h in hist)
    report("lookahead fit beats one teacher step on val (rel err)", best, f"< {v0:.3f}", best < v0)


def test_greedy_cell3():
    """Cell 3: zero-init no move, no gradient evaluations at inference, n_fwd counted; training smoke."""
    from dameta.meta.meta_greedy import GreedyConfig, build_greedy, run_greedy
    m, s = control_stats(torch.tensor(TRAIN.x_true.values), CFG.n_obs, CFG.sigma_q, False)
    g0 = build_greedy(3, GreedyConfig(), m, s)
    cost = make_cost(VAL)
    n0 = cost.n_grad_evals
    r = run_greedy(cost, torch.tensor(VAL.x0_b.values), g0, max_iter=20, tol=0.0, diag=False)
    move = np.abs(r["x0_a"] - VAL.x0_b.values).max()
    report("greedy zero-init: no move, 0 grad evals, n_fwd = 20", float(move),
           "== 0", move == 0 and cost.n_grad_evals == n0 and (r["n_fwd_evals"] == 20).all())
    # Mechanism check only: with input (x, k) the one-step greedy objective has no case information, so the
    # learnable average move is ~ no move (observed train ratios ~1.0); generalisation is NOT asserted here.
    g = build_greedy(3, GreedyConfig(n_epochs=2, batch=32, episode_iters=10), m, s)
    th0 = torch.cat([p.detach().flatten().clone() for p in g.net.parameters()])
    hist = g.fit(lambda idx: make_cost(TRAIN, idx=idx), x0(TRAIN), make_cost(VAL), x0(VAL))
    th1 = torch.cat([p.detach().flatten() for p in g.net.parameters()])
    ratios = [h["ratio_mean"] for h in hist if "ratio_mean" in h]
    print("       train greedy ratios", [round(r, 3) for r in ratios],
          "val scores", [round(h["val_score"], 2) for h in hist if "val_score" in h])
    report("greedy training mechanism: theta moves, finite ratios", float((th1 - th0).norm()), "> 0 and finite",
           (th1 - th0).norm() > 0 and np.isfinite(ratios).all())


def test_supervised_obs_flag():
    """Cell 4 + obs_input: pairs carry the case observations; inference with obs is case-independent."""
    from dameta.meta import SupConfig, build_sup, collect_teacher_pairs, obs_features, run_meta_b_sup
    m, s = control_stats(torch.tensor(TRAIN.x_true.values), CFG.n_obs, CFG.sigma_q, False)
    cost = make_cost(VAL)
    of = obs_features(cost, m, s)
    p = collect_teacher_pairs(cost, x0(VAL), "adagrad", 0.1, 50, obs_feats=of)
    ok_pairs = p["obs"].shape == (p["c"].shape[0], 66) and torch.equal(p["obs"][p["k"] == 0], of)
    lrn = build_sup(3, SupConfig(obs_input=True, epochs=1), m, s, p)
    a = run_meta_b_sup(cost, torch.tensor(VAL.x0_b.values), lrn, "adagrad", 0.1, 20, 0.0, diag_true_grad=False)
    b = run_meta_b_sup(make_cost(VAL, idx=slice(0, 3)), torch.tensor(VAL.x0_b.values[:3]), lrn, "adagrad", 0.1, 20,
                       0.0, diag_true_grad=False)
    err = np.abs(a["x0_a"][:3] - b["x0_a"]).max()
    report("supervised obs_input: pairs carry obs; case independence", float(err), "< 1e-12 and obs pairs",
           ok_pairs and err < 1e-12)


if __name__ == "__main__":
    for name, fn in [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]:
        fn()
    print(f"\n{sum(r[3] for r in RESULTS)}/{len(RESULTS)} checks passed")
