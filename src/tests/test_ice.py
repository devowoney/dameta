"""
ICE-table tests (M6, 2026-11-19): classical references and the meta cells of the 4D-Var ICE table.

Classical part:
    * Adam reference step == torch.optim.Adam on a batched quadratic (same iterates), frozen cases do not move
    * run_4dvar_v2 record with the G1 hook: grad_cum present and monotone; G1 == 1 when the estimation model is the
      truth (the optimizer consumes the true gradient itself); hook time excluded from wall_s
Run on HPC only (gate job):  python src/tests/test_ice.py      (plain script, like the other dameta tests)
"""

import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dameta.da.reference_step import ReferenceStep  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _quadratic(n=4, d=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.linalg.qr(torch.randn(n, d, d, generator=g, dtype=torch.float64))[0]
    ev = torch.logspace(0, 2, d, dtype=torch.float64)
    a = q @ torch.diag_embed(ev.expand(n, d)) @ q.transpose(1, 2)
    x_star = torch.randn(n, d, generator=g, dtype=torch.float64)
    return a, x_star


# ---- classical references ------------------------------------------------------------------------------------------
def test_adam_step_matches_torch_adam():
    a, x_star = _quadratic()
    c = torch.zeros_like(x_star)
    ref = ReferenceStep(c, method="adam", lr=0.05)
    x = torch.zeros_like(x_star).requires_grad_(True)               # plain torch.optim.Adam on the same problem
    opt = torch.optim.Adam([x], lr=0.05)
    for _ in range(30):
        g = torch.einsum("nij,nj->ni", a, c - x_star)
        c = c - ref(c, g)
        opt.zero_grad()
        x.grad = torch.einsum("nij,nj->ni", a, x.detach() - x_star)
        opt.step()
        assert torch.allclose(c, x.detach(), atol=1e-12, rtol=0)
    assert ref.describe() == "Adam(lr=0.05)"


def test_adam_frozen_case_does_not_move():
    a, x_star = _quadratic(seed=1)
    c = torch.zeros_like(x_star)
    ref = ReferenceStep(c, method="adam", lr=0.1)
    active = torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=torch.float64)
    for _ in range(5):
        g = torch.einsum("nij,nj->ni", a, c - x_star)
        f = ref(c, g, active)
        assert f[1].abs().max() == 0
        c = c - f


# ---- classical runner record (G1 hook, gradient counts) ------------------------------------------------------------
def _cfg(**over):
    cfg = OmegaConf.load(ROOT / "src/configs/fourdvar_v2.yaml")
    del cfg["hydra"]
    cfg.data_dir, cfg.emulator_dir = str(ROOT / "tmp/data"), str(ROOT / "tmp/emulators")
    cfg.result_group = cfg.out_dir = "unused"
    cfg.pattern, cfg.n_cases, cfg.max_iter, cfg.g1_iter = "x_only", 6, 15, True
    for k, v in over.items():
        cfg[k] = v
    return cfg


def test_runner_record_truth_g1_is_one():
    for method, lr in [("gd", 1e-3), ("adam", 0.1), ("adagrad", 0.3), ("lbfgs_wolfe", 1.0)]:
        for constraint in ("strong", "weak"):
            _check_truth_record(method, lr, constraint)


def _check_truth_record(method, lr, constraint):
    import run_4dvar_v2 as R2
    out = R2.run(_cfg(model="truth", method=method, lr=lr, constraint=constraint, split="val"))
    gc = out.grad_cum.values
    assert (np.diff(gc, axis=1) >= 0).all() and gc[:, 0].max() == 0
    g1 = out.g1_grad.values
    assert np.nanmin(g1) > 1 - 1e-10                                # truth model: consumed gradient IS the true one
    assert np.isfinite(out.g1_move.values[:, 0]).all()
    if method == "gd":                                              # GD moves along -g exactly
        assert np.nanmin(out.g1_move.values) > 1 - 1e-10
    assert out.attrs["wall_s"] > 0 and out.attrs["g1_hook_s"] > 0


def test_runner_f1_g1_below_one_and_lbfgs_counts_line_search():
    import run_4dvar_v2 as R2
    out = R2.run(_cfg(model="f1", method="lbfgs_wolfe", lr=1.0))
    g1 = out.g1_grad.values
    print(f"   F1 G1 per iterate: max {np.nanmax(g1):.6f} median {np.nanmedian(g1):.4f} min {np.nanmin(g1):.4f}")
    assert np.nanmax(g1) < 1.0, f"G1 max {np.nanmax(g1)}"            # false model: never exactly the true gradient
    # At the first guess the F1 gradient is close to the true one (m5: G1 ~ 0.99). Near the F1 optimum it is not:
    # the F1 gradient vanishes while the true gradient still points to the true-model optimum, so G1 decays along
    # the iterations (gate 2026-11-19: median over all iterates 0.16) -> test the first iterate only.
    assert np.median(g1[:, 0]) > 0.9, f"G1 at the first guess, median {np.median(g1[:, 0])}"
    # Wolfe L-BFGS: total gradient evaluations >= outer iterations (line-search trials are counted)
    gc, ne = out.grad_cum.values[:, -1], out.n_grad_evals.values
    print(f"   grad_cum final {gc.tolist()} | n_grad_evals {ne.tolist()}")
    assert (gc >= ne - 1).all(), "line-search trials not counted"



# ---- cell 2: unsupervised learned optimizer with history input (meta_b_free history mode) --------------------------
def _cell2_problem(constraint="strong", n=5):
    import run_4dvar_v2 as R2
    from dameta.meta.meta_b_direct import DirectConfig, build_direct
    cfg = _cfg(model="f1", constraint=constraint, split="val")
    cfg.n_cases = n
    P = R2.setup_problem(cfg, "val", n)
    cost = P["cost"]
    d = 3 + (3 * (P["ccfg"].n_obs - 1) if cost.weak else 0)
    dcfg = DirectConfig(hist_m=10, base="lbfgs", first_step="teacher", hidden=32, seed=0)
    learner = build_direct(d, dcfg, torch.zeros(d, dtype=torch.float64), torch.ones(d, dtype=torch.float64),
                           n_obs=P["ccfg"].n_obs)
    with torch.no_grad():                                           # non-trivial weights: the test must see the net
        for prm in learner.net.parameters():
            prm.add_(0.05 * torch.randn_like(prm))
    c0 = cost.pack(P["x0_b"], torch.zeros(n, P["ccfg"].n_obs - 1, 3, dtype=torch.float64) if cost.weak else None)
    return cost, P, learner, c0


def test_cell2_episode_memory_matches_inference():
    """A training episode without theta update makes exactly the moves of frozen run_direct (same own L-BFGS memory,
    same teacher first step) -> the unsupervised learner is trained on the trajectory it will follow at inference."""
    from dameta.meta.meta_b_direct import run_direct
    from dameta.meta.meta_b_free import FreeDeltaConfig, free_delta_episode
    for constraint in ("strong", "weak"):
        cost, P, learner, c0 = _cell2_problem(constraint)
        n_iter = 8
        r_ep = free_delta_episode(learner, cost, c0, "lbfgs_wolfe", 1.0, n_iter, None, FreeDeltaConfig())
        r_in = run_direct(cost, P["x0_b"], learner, max_iter=n_iter, tol=0.0)
        err = float((r_ep["c"].detach() - torch.tensor(r_in["control_final"])).abs().max())
        print(f"   {constraint}: |episode - run_direct| after {n_iter} moves = {err:.2e}")
        assert err < 1e-9, err


def test_cell2_training_steps_change_theta():
    from dameta.meta.meta_b_free import FreeDeltaConfig, free_delta_episode
    cost, P, learner, c0 = _cell2_problem("weak", n=4)
    before = [prm.detach().clone() for prm in learner.net.parameters()]
    opt = torch.optim.Adam(learner.net.parameters(), lr=1e-3)
    r = free_delta_episode(learner, cost, c0, "lbfgs_wolfe", 1.0, 40, opt, FreeDeltaConfig(w_align=0.1))
    moved = sum(float((a.detach() - b).abs().max()) for a, b in zip(learner.net.parameters(), before))
    assert moved > 0 and torch.isfinite(r["c"]).all()
    assert any("L_perf" in dg for dg in r["diag"]) and any("L_align" in dg for dg in r["diag"])
    # regression (2026-11-19): converged cases (teacher step 0) must not blow l_perf up (was ~1e289)
    lp = [dg["L_perf"] for dg in r["diag"] if "L_perf" in dg]
    print(f"   l_perf per iteration: {['%.3g' % v for v in lp]}")
    assert all(np.isfinite(v) and abs(v) < 1e3 for v in lp), lp



# ---- cell 3: learned optimization (Meta-A unroll) with the L-BFGS base ---------------------------------------------
def test_cell3_untrained_lbfgs_base_is_fixed_step_lbfgs():
    """Zero-initialised correction: the unroll with base lbfgs makes exactly the moves of fixed-step L-BFGS (lr 1)."""
    from dameta.meta.meta_a import UnrollConfig, build_unroll, run_meta_a
    for constraint in ("strong", "weak"):
        cost, P, _, c0 = _cell2_problem(constraint)
        d = c0.shape[1]
        ucfg = UnrollConfig(K=12, base="lbfgs", loss="sum")
        lrn = build_unroll(d, ucfg, torch.zeros(d, dtype=torch.float64), torch.ones(d, dtype=torch.float64),
                           P["ccfg"].n_obs, hidden=32, n_layers=2, output_scale=5.0)
        r = run_meta_a(cost, P["x0_b"], lrn)
        ref = ReferenceStep(c0, method="lbfgs", lr=1.0)
        c = c0.clone()
        for _ in range(ucfg.K):
            c = c - ref(c, cost.grad(c)[1])
        err = float(np.abs(r["x0_iter"][:, -1] - c[:, :3].numpy()).max())
        print(f"   {constraint}: |unroll(base lbfgs, untrained) - fixed-step L-BFGS| after {ucfg.K} moves = {err:.2e}")
        assert err < 1e-9, err
        assert (r["grad_cum"][:, -1] == ucfg.K).all() and (r["grad_cum"][:, 0] == 0).all()


def test_cell3_training_lowers_loss():
    from dameta.meta.meta_a import UnrollConfig, build_unroll
    cost, P, _, c0 = _cell2_problem("strong", n=5)
    d = c0.shape[1]
    lrn = build_unroll(d, UnrollConfig(K=8, base="lbfgs", loss="sum", lr=1e-2), torch.zeros(d, dtype=torch.float64),
                       torch.ones(d, dtype=torch.float64), P["ccfg"].n_obs, hidden=32, n_layers=2, output_scale=5.0)
    opt = torch.optim.Adam(lrn.net.parameters(), lr=1e-2)
    l0 = float(lrn.loss(cost, c0).detach())
    for _ in range(15):
        opt.zero_grad()
        loss = lrn.loss(cost, c0)
        loss.backward()
        opt.step()
    l1 = float(lrn.loss(cost, c0).detach())
    print(f"   unroll loss (mean_m J(x^m)/J(x^0)): untrained {l0:.4f} -> after 15 steps {l1:.4f}")
    assert np.isfinite(l1) and l1 < l0


def test_cell3_backtrack_monotone_and_finite():
    """Safeguarded unroll (2026-11-20): recorded J never increases and stays finite (strong / weak, with a random
    correction so that some moves are rejected); a few training steps keep the loss finite."""
    from dameta.meta.meta_a import UnrollConfig, build_unroll, run_meta_a
    for constraint in ("strong", "weak"):
        cost, P, _, c0 = _cell2_problem(constraint, n=5)
        d = c0.shape[1]
        lrn = build_unroll(d, UnrollConfig(K=20, base="lbfgs", safeguard="backtrack", loss="sum", lr=1e-2),
                           torch.zeros(d, dtype=torch.float64), torch.ones(d, dtype=torch.float64),
                           P["ccfg"].n_obs, hidden=32, n_layers=2, output_scale=5.0)
        with torch.no_grad():
            for prm in lrn.net.parameters():
                prm.add_(0.5 * torch.randn_like(prm))
        r = run_meta_a(cost, P["x0_b"], lrn)
        J = r["J_iter"]
        rise = float(np.nanmax(np.diff(J, axis=1) / J[:, :-1]))
        print(f"   {constraint}: J {J[:, 0].mean():.2f} -> {J[:, -1].mean():.2f}, max relative rise {rise:.2e}, "
              f"grads {int(r['grad_cum'][0, -1])}")
        assert np.isfinite(J).all() and rise <= 1e-12 and int(r["grad_cum"][0, -1]) == 20
        opt = torch.optim.Adam(lrn.net.parameters(), lr=1e-2)
        for _ in range(3):
            opt.zero_grad()
            loss = lrn.loss(cost, c0)
            assert torch.isfinite(loss) and float(loss) <= 1.0 + 1e-12      # accepted J <= J(x^0)
            loss.backward()
            opt.step()


# ---- cell 4: adjoint-free learned gradient driving strong-Wolfe L-BFGS ---------------------------------------------
class _ExactJoLearner:
    """Stand-in learned gradient returning the EXACT grad J_o of a single-case cost (cell 4 plumbing test)."""
    def __init__(self, cost):
        self.cost = cost

    def grad_obs(self, c, k, fwd, weak=False):
        from dameta.meta.learned_gradient import jo_grad
        with torch.enable_grad():
            return jo_grad(self.cost, c)


def test_cell4_exact_learned_gradient_reproduces_wolfe_lbfgs():
    """With the exact grad J_o as 'learned' gradient, cell 4 (forward runs + Wolfe L-BFGS on g^) makes the moves of
    classical Wolfe L-BFGS, and its forward-run count equals the classical gradient count (line search included)."""
    from dameta.da import run_fourdvar
    from dameta.meta.learned_gradient import run_learned_gradient
    from run_4dvar_v2 import subset_cost
    for constraint in ("strong", "weak"):
        cost_all, P, _, _ = _cell2_problem(constraint, n=3)
        cfg = _cfg(model="f1", constraint=constraint)
        for i in range(3):                                          # one case at a time (stand-in learner is per case)
            cost = subset_cost(cfg, P, [i])
            xb = P["x0_b"][i:i + 1]
            r4 = run_learned_gradient(cost, xb, _ExactJoLearner(cost), "lbfgs_wolfe", 1.0, max_iter=30, tol=1e-6,
                                      diag_exact=False)
            rc = run_fourdvar(cost, xb, "lbfgs_wolfe", 1.0, max_iter=30, tol=1e-6)
            err = float(np.abs(r4["control_final"] - rc["control_final"]).max())
            same_cost = np.array_equal(r4["fwd_cum"], rc["grad_cum"])
            print(f"   {constraint} case {i}: |cell4(exact) - Wolfe L-BFGS| = {err:.2e}, "
                  f"forward runs {int(r4['fwd_cum'][0, -1])} vs gradients {int(rc['grad_cum'][0, -1])}")
            assert err < 1e-8 and same_cost, (err, r4["fwd_cum"][0, :8], rc["grad_cum"][0, :8])


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    torch.set_default_dtype(torch.float64)
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}", flush=True)
        except Exception as e:                                      # report every test, then fail the job
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}", flush=True)
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
