# Progress — branch m5-meta-grid (Milestone 5: meta-learner grid)

- 2026-11-14: Worktree created from c0a1db0 (= main after the M4 merge). Inputs copied (1.3 GB). Design recorded in
  decisions.md (6-cell grid). Implementation order: cell 5 (Meta-A) and 7 (shared unroll code), then 2, 3; 4 rerun
  with obs_input.
- 2026-11-14: Built the remaining grid cells.
  * cells 5 / 7 (learned optimization): src/dameta/meta/meta_a.py (LearnedOptimization: K-step unroll, first order
    default, second_order flag, obs_input flag, fit with best val epoch, run_meta_a fixed budget) + FreeUpdateNet.
  * cell 2 (lookahead): src/dameta/meta/meta_b_lookahead.py (teacher-trajectory pairs, m = 5, relative MSE; Meta-B
    chain + run_meta_b inference reused).
  * cell 3 (greedy): src/dameta/meta/meta_greedy.py (online episodes, J(x^{k+1})/J(x^k), stopping rule, n_fwd_evals).
  * cell 4: obs_input flag (GradNet d_obs, pairs carry obs, inference passes obs).
  * run_meta_b.py variants delta | lookahead | greedy | supervised | unroll; configs/meta_b.yaml blocks.
  Tests: test_m5.py 18/18 (zero-init/no-move, call counts, 2nd-order theta-grad == FD 3e-9, case independence, obs
  flag, training smokes, lookahead pairs == classical trajectory, checkpoint); M4 19/19 and M2 8/8 still pass.
  Observation (smoke): cell 3 greedy ratios ~1.0 even on training episodes — with input (x, k) a one-step move
  cannot be case-specific (identifiability); cell 7 smoke still lowers J(x^K)/J(x^0) to 0.94 (common drift).
- 2026-11-14: D1 v2 built (blueprint step 1). New: src/dameta/data/generate_v2.py (window 110, 12 obs, 100-step truth
  history), src/dameta/data/background.py (time-lagged false-model ensemble: 10 lags 0.1-1 TU, first guess = 0.5-TU
  member, B = member covariance + 1 % diagonal floor), src/gen_data_v2.py + configs/gen_data_v2.yaml; v1 untouched.
  Generated tmp/data/l63v2_{train,val,test,test_ood}_seed42.nc + _bg_{f1,rho275}.nc. tests/test_d1v2.py 10/10.
  First-guess error (median RMSE vs truth): 0.61-0.65 in-dist, 0.27 test_ood (v1 truth + N(0, I): 0.87).
  ISSUE (pilot decision): lagged-ensemble B is nearly rank-1 (95 % of variance in one direction; condition number
  ~300 = floor cap) and miscalibrated in-dist: chi2 = e^T B^-1 e median 7.1 / mean 10.7 (calibrated ~3); first-guess
  error along the two minor eigenvectors ~1.45 x the assumed std, along the leading one 0.56 x. test_ood calibrated
  (chi2 median 3.1). Options: larger floor / inflation / hybrid B with a static part.
- 2026-11-14: Pilot chose hybrid B. src/tune_hybrid_b.py: B_h = alpha B_ens + (1 - alpha) s2 I, (alpha, s2) by
  maximum likelihood of the val t0 innovations y_0 - x0_b ~ N(0, B_h + R) (truth-free); tmp/data/hybrid_B_seed42.json:
  f1 alpha 0.25 s 0.63, rho275 alpha 0.25 s 0.67. chi2 mean (truth diagnostic) ensemble -> hybrid: val 10.7 -> 3.2,
  test 9.7 -> 3.4, test_ood 3.6 -> 1.4 (f1; rho275 similar): calibrated in-dist, too loose on test_ood.
  Bug found: netCDF4 segfault when the same file is reopened lazily while handles are alive -> xr.load_dataset.
  Blueprint step 2-3: FourDVarCost(b_inv=...) case-specific B^-1 in J_b; ParamFourDVarCost (BPE, control
  [x0, sigma, rho, beta], L63 RK4 with control parameters, prior theta_b = F1 params, std 10 %);
  l63_tendency_theta; background.load_hybrid_background. tests/test_cost_v2.py 6/6 (FD gradients BICE 8e-8, BPE
  4e-7; BPE(theta_F1) == BICE(F1); J_o prefers rho 28 in 88 % of cases). Regressions: M2 8/8, M4 19/19, M5 18/18,
  D1v2 10/10. Next: blueprint step 4 (runner on D1 v2: false model F1/F2, hybrid B, BPE; classical baselines).
- 2026-11-14: Step 4 started. New runner src/run_4dvar_v2.py + configs/fourdvar_v2.yaml (v1 runner untouched):
  model truth (E1 ref, F1 first guess/B) | f1 | emulator; problem bice | bpe; hybrid B; records rmse_fc (estimation
  model; bpe with estimated theta), rmse_fc_m1 (true model, IC quality), J_true_final, theta_a. run_fourdvar now
  also returns control_final. Smoke (8 val cases, 50 its): all modes OK; false-model analyses barely beat the first
  guess (0.44 vs 0.43; truth model 0.34); BPE moves rho 27.5 -> 27.7. lr sweeps launched: 56 runs, 50 val cases,
  4-point grids (Adagrad 0.03-1, GD 1e-4-3e-3), tmp/results/v2_<model>[_bpe]_sweep.
- 2026-11-14: Adjoint-free baselines: src/dameta/da/gradients.py (FDGradient: central FD, 2d+1 F1 forward runs;
  HybridGradient: J from F1 + emulator adjoint gradient); run_4dvar_v2 gradient=exact|fd|hybrid (records
  fwd_per_gradient). test_cost_v2 9/9 (FD == exact to 1e-8; emulator grad vs F1 grad cos 0.999 at first guess).
  First sweep launch failed (Hydra cannot parse nested ${} in a CLI override; no run started) -> group name built in
  the shell; sweep relaunched (56 runs).
- 2026-11-14: v2 lr sweeps done (57 runs incl. edge extension; rule applied, all interior): truth Adagrad 1.0 strong
  (3.0 checked) / 0.1 weak, GD 1e-3 / 3e-4; F1 Adagrad 0.3 / 0.1, GD 1e-3 / 1e-3; F2 Adagrad 0.3 / 0.1,
  GD 1e-3 / 3e-4; F1 BPE Adagrad 0.1, GD 3e-4. Sweep signal (val, 50 cases, Adagrad strong): IC error truth 0.29,
  F1 0.45, F2 0.46; BPE F1 0.38 (parameter estimation recovers part of the model-error loss).
  Final runs launched: 72 (48 BICE truth/F1/F2 x strong/weak x full/x_only x GD/Adagrad x test/test_ood; 8 BPE;
  16 adjoint-free F1: FD and hybrid gradients), results tmp/results/v2_<model>[_bpe|_fd|_hybrid].
- 2026-11-14: Step 4 done: 72 final runs (0 diverged), src/report_v2.py -> tmp/results/v2_summary.{csv,md};
  paper-ready entry in findings.md. Key: false-model IC 0.42 vs truth-model 0.33 (first guess 0.645); F2 ~ F1;
  BPE rho 27.81 and gap vs E1 1.02 -> 0.45; FD == adjoint (7 / 73 fwd per gradient); hybrid no loss (1 fwd).
  Next: step 5 (cell 8 + meta grid on D1 v2).
