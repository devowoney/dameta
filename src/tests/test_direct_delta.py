"""
delta_exact, the supervised delta learned optimizer (src/dameta/meta/meta_b_direct.py, 2026-11-18).

Plain-python test script (no pytest in env `oceanai`). Run from the project root:   python src/tests/test_direct_delta.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dameta.da import CostConfig, FourDVarCost  # noqa: E402
from dameta.da.fourdvar import run_fourdvar  # noqa: E402
from dameta.dynamics import L63Model  # noqa: E402
from dameta.meta import control_stats  # noqa: E402
from dameta.meta.meta_b_direct import (DirectConfig, DirectDelta, build_direct,  # noqa: E402
                                       collect_delta_pairs, run_direct)

torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parents[2]
VAL = xr.load_dataset(ROOT / "tmp/data/l63_val_seed42.nc").isel(ic=slice(0, 16))
TRAIN = xr.load_dataset(ROOT / "tmp/data/l63_train_seed42.nc").isel(ic=slice(0, 64))
M, CFG = L63Model(), CostConfig()
MASK = torch.tensor(VAL.obs_mask.sel(pattern="full").values)


def report(name, value, thr, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<62s} value={value:.3e}  ({thr})")
    assert ok, name


def make_cost(ds, weak=False):
    return FourDVarCost(M.step, torch.tensor(ds.y_obs.values), MASK, torch.tensor(ds.x0_b.values), CFG, weak=weak)


def learner(weak=False, **kw):
    d = 3 + (3 * (CFG.n_obs - 1) if weak else 0)
    m, s = control_stats(torch.tensor(TRAIN.x_true.values), CFG.n_obs, CFG.sigma_q, weak)
    return build_direct(d, DirectConfig(**kw), m, s)


def test_init_is_gd_one_grad_per_iter():
    for weak in (False, True):
        cost = make_cost(VAL, weak)
        lrn = learner(weak)
        lrn.net.log_mag0.fill_(np.log(1e-3))         # zero-init heads: u = 1e-3 g_std -> GD with lr 1e-3 per block
        x = cost.pack(torch.tensor(VAL.x0_b.values), torch.zeros(16, CFG.n_obs - 1, 3) if weak else None)
        g = torch.randn_like(x)
        from dameta.meta.meta_b_direct import grad_scale, standardize_grad
        exp_ = -1e-3 * grad_scale(g, x.shape[1]) * standardize_grad(g, x.shape[1])
        err = (lrn.delta(x, g, 0) - exp_).abs().max().item()
        report(f"init = GD step lr 1e-3 per block ({'weak' if weak else 'strong'})", err, "< 1e-15", err < 1e-15)
        n0 = cost.n_grad_evals
        r = run_direct(cost, torch.tensor(VAL.x0_b.values), lrn, max_iter=5, tol=0.0)
        calls = cost.n_grad_evals - n0
        report("one batched gradient per iteration, no teacher", calls, "== 5", calls == 5)
        report("grad_cum[k] == k", float(np.abs(r["grad_cum"][:, :6] - np.arange(6)).max()), "== 0",
               bool((r["grad_cum"][:, :6] == np.arange(6)).all()))


def test_pairs_are_teacher_deltas():
    cost, xb = make_cost(VAL), torch.tensor(VAL.x0_b.values)
    p = collect_delta_pairs(cost, xb, "adagrad", 0.1, max_iter=30, tol=1e-6)
    ref = run_fourdvar(make_cost(VAL), xb, "adagrad", 0.1, max_iter=30, tol=1e-6)
    first = p["k"] == 0
    err = (p["x"][first] + p["target"][first] - torch.tensor(ref["x0_iter"][:, 1])).abs().max().item()
    report("pair target = x^1 - x^0 of the classical teacher run", err, "< 1e-12", err < 1e-12)


def test_gd_teacher_is_reproduced():
    """GD delta = -lr g = -RMS(g) * (lr g_std): exactly representable (u linear in g_std) -> run_direct ~ GD."""
    lr = 1e-3
    tr = collect_delta_pairs(make_cost(TRAIN), torch.tensor(TRAIN.x0_b.values), "gd", lr, max_iter=40, tol=0.0)
    va = collect_delta_pairs(make_cost(VAL), torch.tensor(VAL.x0_b.values), "gd", lr, max_iter=40, tol=0.0)
    lrn = learner(epochs=150, batch_size=256, lr=3e-3)
    hist = lrn.fit(tr, va)
    report("fit to GD deltas: best val relative error", min(h["val_rel_median"] for h in hist), "< 1e-2",
           min(h["val_rel_median"] for h in hist) < 1e-2)
    xb = torch.tensor(VAL.x0_b.values)
    r = run_direct(make_cost(VAL), xb, lrn, max_iter=40, tol=0.0)
    ref = run_fourdvar(make_cost(VAL), xb, "gd", lr, max_iter=40, tol=0.0)
    # median over cases: on the chaotic L63 cost a small per-step imitation error can switch a single case's basin
    rel = float(np.median(np.abs(r["J_iter"][:, 40] - ref["J_iter"][:, 40]) / np.abs(ref["J_iter"][:, 40])))
    report("frozen direct learner reproduces GD J(x^40) (median rel.)", rel, "< 1e-2", rel < 1e-2)


def test_free_delta_stored_graph_chain_rule():
    """delta (unsupervised, free): theta-gradient of <g(x + Delta_hat), Delta_hat_theta> through the STORED graph of
    the move (K = 5 inner rollout) == d/dtheta J(x + Delta_hat_theta) (FD)."""
    for weak in (False, True):
        cost, lrn = make_cost(VAL, weak), learner(weak, K=5)
        for prm in lrn.net.parameters():
            torch.nn.init.normal_(prm, std=0.05)
        lrn.net.log_mag0.fill_(np.log(1e-3))
        c = cost.pack(torch.tensor(VAL.x0_b.values), torch.zeros(16, CFG.n_obs - 1, 3) if weak else None)
        _, g0 = cost.grad(c)
        D = lrn.delta(c, g0, 3)                                  # stored graph of the move
        _, g1 = cost.grad(c + D.detach())                        # gradient the next iteration evaluates anyway
        lrn.net.zero_grad()
        (g1 * D).sum().backward()
        w = lrn.net.head_mag.bias
        an, h, vals = w.grad[0].item(), 1e-6, []
        for sgn in (1, -1):
            with torch.no_grad():
                w[0] += sgn * h
                vals.append(float(cost(*cost.unpack(c + lrn.delta(c, g0, 3))).sum()))
                w[0] -= sgn * h
        fd = (vals[0] - vals[1]) / (2 * h)
        rel = abs(an - fd) / abs(fd)
        report(f"free delta: stored-graph theta-gradient == FD, K 5 ({'weak' if weak else 'strong'})", rel, "< 1e-5",
               rel < 1e-5)


def test_free_delta_episode_and_teacherless_inference():
    from dameta.meta.meta_b_free import FreeDeltaConfig, free_delta_episode
    lrn = learner(K=3)
    lrn.net.log_mag0.fill_(np.log(1e-3))
    th0 = [p.detach().clone() for p in lrn.net.parameters()]
    opt = torch.optim.Adam(lrn.net.parameters(), lr=1e-3)
    r = free_delta_episode(lrn, make_cost(TRAIN), torch.tensor(TRAIN.x0_b.values), "adagrad", 0.1, 10, opt,
                           FreeDeltaConfig())
    moved = max((p.detach() - q).abs().max().item() for p, q in zip(lrn.net.parameters(), th0))
    has_perf = sum("L_perf" in d for d in r["diag"])
    report("free delta episode: theta updated, l_perf from iteration 1 on", moved, "> 0; 9 perf terms",
           moved > 0 and has_perf == 9 and "L_perf" not in r["diag"][0])
    cost = make_cost(VAL)
    n0 = cost.n_grad_evals
    out = run_direct(cost, torch.tensor(VAL.x0_b.values), lrn, max_iter=6, tol=0.0)
    report("free delta inference: learner only, 1 gradient / iteration", cost.n_grad_evals - n0, "== 6",
           cost.n_grad_evals - n0 == 6 and np.isfinite(out["J_iter"]).all())


def test_obs_input():
    """Observation input: features = [y_std * mask, mask, t / L] per case; pairs carry the case index; the output
    depends on the observations; fit and teacherless inference run with obs."""
    from dameta.meta.da_utils_obs import obs_time_features
    cost = make_cost(VAL)
    m, s = control_stats(torch.tensor(TRAIN.x_true.values), CFG.n_obs, CFG.sigma_q, False)
    f = obs_time_features(cost, m, s)
    t_ok = torch.allclose(f[:, -CFG.n_obs:], (torch.arange(0, CFG.window + 1, CFG.obs_every) / CFG.window).expand(16, -1))
    y_ok = torch.allclose(f[:, :3 * CFG.n_obs], ((cost.y - m) / s * cost.mask).flatten(1))
    report("obs features: shape 7 n_obs, values and times", float(f.shape[1]), f"== {7 * CFG.n_obs}",
           f.shape == (16, 7 * CFG.n_obs) and t_ok and y_ok)
    p = collect_delta_pairs(cost, torch.tensor(VAL.x0_b.values), "adagrad", 0.1, max_iter=5, tol=0.0)
    ok_case = bool((p["case"][p["k"] == 0] == torch.arange(16)).all()) and \
        bool((p["x"][p["k"] == 0] == torch.tensor(VAL.x0_b.values)).all())
    report("pairs carry the case index", float(ok_case), "== 1", ok_case)
    lrn = build_direct(3, DirectConfig(obs_input=True, epochs=3), m, s, n_obs=CFG.n_obs)
    for prm in lrn.net.parameters():
        torch.nn.init.normal_(prm, std=0.05)
    x, g = torch.tensor(VAL.x0_b.values), torch.randn(16, 3)
    o = lrn.obs_feats(cost)
    dd = (lrn.delta(x, g, 2, o) - lrn.delta(x, g, 2, o.roll(1, 0))).abs().max().item()
    report("Delta_hat depends on the observations", dd, "> 0", dd > 0)
    p = {**p, "obs": o}
    lrn.fit(p, p)
    lrn2 = DirectDelta.from_state_dict(lrn.state_dict())
    r = run_direct(make_cost(VAL), x, lrn2, max_iter=4, tol=0.0)
    report("obs learner: fit, checkpoint, teacherless inference", float(np.isfinite(r["J_iter"]).all()), "== 1",
           bool(np.isfinite(r["J_iter"]).all()) and lrn2.net.d_obs == 7 * CFG.n_obs)


def test_temporal_net():
    """arch temporal: token layout (weak control on the obs times), time locality of the conv, fit + inference."""
    from dameta.meta.temporal_net import TemporalDeltaNet
    for weak in (False, True):
        d = 3 + (3 * (CFG.n_obs - 1) if weak else 0)
        m, s = control_stats(torch.tensor(TRAIN.x_true.values), CFG.n_obs, CFG.sigma_q, weak)
        lrn = build_direct(d, DirectConfig(obs_input=True, arch="temporal", n_layers=1, kernel=3, epochs=3), m, s,
                           n_obs=CFG.n_obs)
        assert isinstance(lrn.net, TemporalDeltaNet)
        for prm in lrn.net.parameters():
            torch.nn.init.normal_(prm, std=0.05)
        cost = make_cost(VAL, weak)
        o = lrn.obs_feats(cost)
        x = cost.pack(torch.tensor(VAL.x0_b.values), torch.zeros(16, CFG.n_obs - 1, 3) if weak else None)
        g = torch.randn_like(x)
        D = lrn.delta(x, g, 2, o)
        ok = D.shape == x.shape and torch.isfinite(D).all()
        if weak:   # 1 conv layer, kernel 3: changing y at the LAST obs time leaves the eta directions far away unchanged
            o2 = o.clone()
            o2[:, 3 * (CFG.n_obs - 1):3 * CFG.n_obs] += 1.0
            D2 = lrn.delta(x, g, 2, o2)
            v1 = D[:, 3:3 * (CFG.n_obs - 3)] / D[:, 3:].norm(dim=1, keepdim=True)
            v2 = D2[:, 3:3 * (CFG.n_obs - 3)] / D2[:, 3:].norm(dim=1, keepdim=True)
            far = (v1 / v1.norm(dim=1, keepdim=True) - v2 / v2.norm(dim=1, keepdim=True)).abs().max().item()
            near = (D[:, -3:] / D[:, 3:].norm(dim=1, keepdim=True) - D2[:, -3:] / D2[:, 3:].norm(dim=1, keepdim=True)
                    ).abs().max().item()
            ok = ok and far < 1e-12 and near > 0
        p = collect_delta_pairs(cost, x[:, :3], "adagrad", 0.1, max_iter=4, tol=0.0)
        lrn.fit({**p, "obs": o}, {**p, "obs": o})
        lrn2 = DirectDelta.from_state_dict(lrn.state_dict())
        r = run_direct(make_cost(VAL, weak), torch.tensor(VAL.x0_b.values), lrn2, max_iter=3, tol=0.0)
        ok = ok and bool(np.isfinite(r["J_iter"]).all()) and isinstance(lrn2.net, TemporalDeltaNet)
        report(f"temporal net: shapes, time locality, fit, checkpoint, inference ({'weak' if weak else 'strong'})",
               float(ok), "== 1", bool(ok))


def _spd_binv(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, 3, 3, generator=g)
    return torch.linalg.inv(a @ a.transpose(1, 2) * 0.3 + 0.2 * torch.eye(3))


def test_cvt_coordinates():
    """coords cvt: round trip of steps, g_v == autograd dJ/dv (v0 = L^-1 (x0 - x0_b), w = eta / sigma_q), and the
    zero-initialised learner is GD in v coordinates (Delta_v = -exp(l0) g_v per block)."""
    from dameta.meta.meta_b_direct import cvt_factor
    for weak in (False, True):
        c0 = make_cost(VAL, weak)
        cost = FourDVarCost(c0.step, c0.y, MASK, c0.x0_b, CFG, weak=weak, b_inv=_spd_binv(16))
        lrn = learner(weak, coords="cvt")
        ctx = lrn.case_ctx(cost)
        L = cvt_factor(cost)
        e1 = (L @ L.transpose(1, 2) - torch.linalg.inv(cost.b_inv)).abs().max().item()
        x = cost.pack(cost.x0_b + torch.randn(16, 3), 0.05 * torch.randn(16, CFG.n_obs - 1, 3) if weak else None)
        dx = torch.randn_like(x)
        e2 = (lrn.step_to_raw(lrn.step_to_learner(dx, ctx), ctx) - dx).abs().max().item()
        _, g = cost.grad(x)
        v, gv = lrn.to_learner(x, g, ctx)
        vv = v.clone().requires_grad_(True)
        x_of_v = torch.cat([cost.x0_b + torch.einsum("nij,nj->ni", L, vv[:, :3])]
                           + ([vv[:, 3:] * CFG.sigma_q] if weak else []), 1)
        (gv_ad,) = torch.autograd.grad(cost(*cost.unpack(x_of_v)).sum(), vv)
        e3 = ((gv - gv_ad).abs().max() / gv_ad.abs().max()).item()
        lrn.net.log_mag0.fill_(np.log(1e-2))
        e4 = (lrn.delta_learner(x, g, 0, None, ctx) + 1e-2 * gv).abs().max().item() / gv.abs().max().item()
        tag = "weak" if weak else "strong"
        report(f"cvt: L L^T = B, step round trip, g_v = dJ/dv, init = GD in v ({tag})", max(e1, e2, e3, e4),
               "< 1e-10", max(e1, e2, e3, e4) < 1e-10)


def test_grad_innov():
    """grad_innov == cost.grad (J, g) + innovations of terms(); pair innovation features == inference features."""
    from dameta.meta.meta_b_direct import grad_innov, innov_features, pair_innov_features
    for weak in (False, True):
        cost = make_cost(VAL, weak)
        x = cost.pack(torch.tensor(VAL.x0_b.values), 0.05 * torch.randn(16, CFG.n_obs - 1, 3) if weak else None)
        j1, g1 = cost.grad(x)
        n0 = cost.n_grad_evals
        j2, g2, inn = grad_innov(cost, x)
        e = max((j1 - j2).abs().max().item(), (g1 - g2).abs().max().item())
        f1 = innov_features(cost, inn)
        f2 = pair_innov_features(cost, x.flip(0), torch.arange(16).flip(0)).flip(0)
        e = max(e, (f1 - f2).abs().max().item())
        jo = 0.5 * (inn ** 2).sum((1, 2)) / CFG.sigma_o2
        e = max(e, (jo - cost.terms(*cost.unpack(x))["Jo"]).abs().max().item())
        report(f"grad_innov == grad, innovations, pair features ({'weak' if weak else 'strong'})", e, "< 1e-10",
               e < 1e-10 and cost.n_grad_evals - n0 == 1)


def test_cvt_innov_lbfgs_fit():
    """cvt + fixed gradient scale + innovation features + full-batch L-BFGS: fits GD deltas (exactly representable
    for B = sigma_b^2 I), checkpoint round trip, teacherless inference with 1 gradient / iteration."""
    lr = 1e-3
    tr = collect_delta_pairs(make_cost(TRAIN), torch.tensor(TRAIN.x0_b.values), "gd", lr, max_iter=40, tol=0.0)
    va = collect_delta_pairs(make_cost(VAL), torch.tensor(VAL.x0_b.values), "gd", lr, max_iter=40, tol=0.0)
    m, s = control_stats(torch.tensor(TRAIN.x_true.values), CFG.n_obs, CFG.sigma_q, False)
    lrn = build_direct(3, DirectConfig(coords="cvt", grad_input="fixed", obs_input=True, obs_feat="innov",
                                       optimizer="lbfgs", epochs=10, lbfgs_iters=20, max_train_pairs=1500), m, s,
                       n_obs=CFG.n_obs)
    tr, va = lrn.prepare_pairs(tr, make_cost(TRAIN)), lrn.prepare_pairs(va, make_cost(VAL))
    hist = lrn.fit(tr, va)
    best = min(h["val_rel_median"] for h in hist)
    report("cvt + innov + L-BFGS fit to GD deltas: best val relative error", best, "< 1e-2", best < 1e-2)
    lrn2 = DirectDelta.from_state_dict(lrn.state_dict())
    cost = make_cost(VAL)
    n0 = cost.n_grad_evals
    r = run_direct(cost, torch.tensor(VAL.x0_b.values), lrn2, max_iter=5, tol=0.0)
    ok = cost.n_grad_evals - n0 == 5 and bool(np.isfinite(r["J_iter"]).all()) and bool(
        (lrn2.g_scale0 == lrn.g_scale0).all())
    report("cvt + innov: checkpoint, inference 1 gradient / iteration", float(cost.n_grad_evals - n0), "== 5", ok)


def test_two_loop_matches_lbfgs_direction():
    """lbfgs_two_loop over the (s, y) memory == reference_step.LBFGSDirection (history 10) along a sequence."""
    from dameta.da.reference_step import LBFGSDirection
    from dameta.meta.meta_b_direct import lbfgs_pair, lbfgs_two_loop
    gen = torch.Generator().manual_seed(1)
    n, d, m = 8, 33, 10
    ref, S, Y, err = LBFGSDirection(history=m), torch.zeros(n, m, d), torch.zeros(n, m, d), 0.0
    A = torch.randn(n, d, d, generator=gen)
    A = A @ A.transpose(1, 2) + torch.eye(d)
    c_prev = g_prev = None
    for t in range(16):
        c = torch.randn(n, d, generator=gen)
        g = torch.einsum("nij,nj->ni", A, c) + (0.5 * torch.randn(n, d, generator=gen) if t % 5 == 3 else 0)
        r = ref(c, g)
        if c_prev is not None:
            s, y = lbfgs_pair(c - c_prev, g - g_prev)
            S, Y = torch.cat([S[:, 1:], s[:, None]], 1), torch.cat([Y[:, 1:], y[:, None]], 1)
        err = max(err, ((lbfgs_two_loop(g, S, Y) - r).abs().max() / r.abs().max()).item())
        c_prev, g_prev = c, g
    report("two-loop over the memory == LBFGSDirection (16 iterations)", err, "< 1e-12", err < 1e-12)


def _wolfe_pairs(ds, n_iter=15):
    from dameta.meta.meta_b_direct import attach_history
    p = collect_delta_pairs(make_cost(ds), torch.tensor(ds.x0_b.values), "lbfgs_wolfe", 1.0, max_iter=n_iter,
                            tol=1e-6)
    return attach_history(p, 10)


def test_history_spans_wolfe_teacher_step():
    """Teacher forcing: every strong-Wolfe L-BFGS step (k >= 1) is t * (two-loop direction of its memory)."""
    from dameta.meta.meta_b_direct import lbfgs_two_loop
    p = _wolfe_pairs(VAL)
    sel = p["k"] > 0
    b = -lbfgs_two_loop(p["g"][sel], p["S"][sel], p["Y"][sel])
    t = p["target"][sel]
    alpha = (t * b).sum(1) / (b * b).sum(1)
    rel = ((t - alpha[:, None] * b).pow(2).sum(1) / t.pow(2).sum(1)).max().item()
    report(f"Wolfe teacher step = t * L-BFGS(memory) direction (median t {alpha.median():.2f})", rel, "< 1e-20",
           rel < 1e-20)


def test_hist_learner_init_and_first_step():
    """hist_m 10, base lbfgs, first_step teacher, zero heads with l0 = 0: iteration 0 = the classical strong-Wolfe
    iteration (same iterate and gradient count), then fixed-step (t = 1) L-BFGS over the learner's own memory;
    the gradient at the accepted line-search point is not counted twice."""
    from dameta.da.reference_step import LBFGSDirection
    m, s = control_stats(torch.tensor(TRAIN.x_true.values), CFG.n_obs, CFG.sigma_q, False)
    lrn = build_direct(3, DirectConfig(hist_m=10, base="lbfgs", first_step="teacher"), m, s, n_obs=CFG.n_obs)
    lrn.net.log_mag0.zero_()
    xb = torch.tensor(VAL.x0_b.values)
    r = run_direct(make_cost(VAL), xb, lrn, max_iter=6, tol=0.0)
    ref = run_fourdvar(make_cost(VAL), xb, "lbfgs_wolfe", 1.0, max_iter=2, tol=0.0)
    e1 = np.abs(r["x0_iter"][:, 1] - ref["x0_iter"][:, 1]).max()
    ok_cost = (r["grad_cum"][:, 1] == ref["grad_cum"][:, 1]).all() and (r["grad_cum"][:, 2] == r["grad_cum"][:, 1]
                                                                         ).all() and \
        (r["grad_cum"][:, 3] == r["grad_cum"][:, 1] + 1).all()
    cost, dirn, e2 = make_cost(VAL), LBFGSDirection(history=10), 0.0
    for k in range(6):
        c = torch.tensor(r["x0_iter"][:, k])
        _, g = cost.grad(c)
        dvec = dirn(c, g)
        if k >= 1:
            e2 = max(e2, np.abs(r["x0_iter"][:, k + 1] - (c - dvec).numpy()).max())
    report("hist learner: first step = Wolfe iteration, then t = 1 L-BFGS; cost bookkeeping", max(e1, e2),
           "< 1e-10", max(e1, e2) < 1e-10 and bool(ok_cost))


def test_hist_fit_checkpoint():
    from dameta.meta.meta_b_direct import attach_history  # noqa: F401  (prepare_pairs attaches the memory)
    m, s = control_stats(torch.tensor(TRAIN.x_true.values), CFG.n_obs, CFG.sigma_q, False)
    lrn = build_direct(3, DirectConfig(hist_m=10, base="grad", first_step="teacher", epochs=3, batch_size=64), m, s,
                       n_obs=CFG.n_obs)
    xb_tr, xb_va = torch.tensor(TRAIN.x0_b.values), torch.tensor(VAL.x0_b.values)
    tr = collect_delta_pairs(make_cost(TRAIN), xb_tr, "lbfgs_wolfe", 1.0, max_iter=10, tol=0.0)
    va = collect_delta_pairs(make_cost(VAL), xb_va, "lbfgs_wolfe", 1.0, max_iter=10, tol=0.0)
    tr, va = lrn.prepare_pairs(tr, make_cost(TRAIN)), lrn.prepare_pairs(va, make_cost(VAL))
    hist = lrn.fit(tr, va)
    lrn2 = DirectDelta.from_state_dict(lrn.state_dict())
    i = torch.arange(20)
    same = (lrn.loss(va, i) - lrn2.loss(va, i)).abs().item()
    r = run_direct(make_cost(VAL), xb_va, lrn2, max_iter=6, tol=0.0)
    ok = hist[0]["epoch"] == -1 and bool((tr["k"] > 0).all()) and same == 0.0 and np.isfinite(r["J_iter"]).all()
    report("hist learner: k = 0 pairs dropped, epoch -1 candidate, fit, checkpoint, inference", same, "== 0", ok)


def test_horizon_target():
    """horizon 2: target = Delta_k + Delta_{k+1} of the same case (capped at the teacher's last iterate)."""
    from dameta.meta.meta_b_direct import attach_horizon
    p = collect_delta_pairs(make_cost(VAL), torch.tensor(VAL.x0_b.values), "adagrad", 0.1, max_iter=8, tol=0.0)
    q = attach_horizon(p, 2)
    err = 0.0
    for i in range(len(p["k"])):
        c, k = p["case"][i], int(p["k"][i])
        nxt = torch.nonzero((p["case"] == c) & (p["k"] == k + 1)).flatten()
        want = p["target"][i] + (p["target"][nxt[0]] if len(nxt) else 0)
        err = max(err, (q["target"][i] - want).abs().max().item())
    report("horizon 2 target = Delta_k + Delta_{k+1} (capped)", err, "< 1e-14", err < 1e-14)


def test_backtrack_safeguard():
    """safeguard backtrack: an over-sized learner step is rejected and halved -> accepted J never increases, no
    divergence, one gradient per iteration; without it the same learner blows up."""
    m, s = control_stats(torch.tensor(TRAIN.x_true.values), CFG.n_obs, CFG.sigma_q, False)
    lrn = build_direct(3, DirectConfig(hist_m=4, base="grad", first_step="teacher"), m, s, n_obs=CFG.n_obs)
    lrn.net.log_mag0.fill_(np.log(5.0))                          # step = 5 RMS(g) per unit: far too long
    xb = torch.tensor(VAL.x0_b.values)
    bad = run_direct(make_cost(VAL), xb, lrn, max_iter=30, tol=0.0)
    r = run_direct(make_cost(VAL), xb, lrn, max_iter=30, tol=0.0, safeguard="backtrack")
    J = r["J_iter"]
    mono = bool((np.diff(J, axis=1) <= 1e-9 * np.abs(J[:, :-1])).all()) and bool(np.isfinite(J).all())
    worse_bad = float(np.nanmax(bad["J_iter"][:, -1] / bad["J_iter"][:, 0]))
    # gradients after the first (Wolfe) step: iterations 2..29 (the gradient at the accepted Wolfe point is free)
    per_it = (r["grad_cum"][:, 30] - r["grad_cum"][:, 1])
    report("backtrack: accepted J non-increasing, finite, 1 gradient / iteration (unguarded max J_end/J_0 "
           f"{worse_bad:.3g})", float(per_it.max()), "== 28", mono and bool((per_it == 28).all()))


def test_online_groups():
    """online order: every pair exactly once; inside an episode the groups follow the teacher iterations (one k per
    group, increasing), episodes = disjoint batches of cases."""
    from dameta.meta.meta_b_direct import online_groups
    p = collect_delta_pairs(make_cost(VAL), torch.tensor(VAL.x0_b.values), "adagrad", 0.1, max_iter=6, tol=0.0)
    gr = online_groups(p, 5, torch.Generator().manual_seed(0))
    allidx = torch.cat(gr).sort().values
    once = bool((allidx == torch.arange(len(p["k"]))).all())
    one_k = all(len(p["k"][g].unique()) == 1 for g in gr)
    ks = [int(p["k"][g[0]]) for g in gr]
    ordered = all(b == a + 1 or b == 0 for a, b in zip(ks, ks[1:]))
    report("online groups: each pair once, one teacher iteration per step, in order", float(len(gr)), "== 4 x 6",
           once and one_k and ordered and len(gr) == 24)


def test_checkpoint_roundtrip():
    lrn = learner()
    for prm in lrn.net.parameters():
        torch.nn.init.normal_(prm, std=0.1)
    lrn.net.log_mag0.fill_(-3.0)
    lrn2 = DirectDelta.from_state_dict(lrn.state_dict())
    x, g = torch.tensor(VAL.x0_b.values), torch.randn(16, 3)
    err = (lrn.delta(x, g, 3) - lrn2.delta(x, g, 3)).abs().max().item()
    report("checkpoint round trip", err, "== 0", err == 0.0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all direct-delta tests passed")
