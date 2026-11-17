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
- 2026-11-14: Step 5 started. run_4dvar_v2 refactored into setup_problem / subset_cost / evaluate_v2 (bit-identical
  rerun of a F1 test record). Cell 8 built: src/dameta/meta/learned_gradient.py (one forward run -> J + innovations;
  analytic grad J_b (+ J_q); GradNet on [x, innovations, mask, emb(k)] predicts grad J_o; supervised on the emulator's
  autodiff grad J_o at states of the adjoint-free hybrid classical run; inference 1 forward / iteration).
  tests/test_cell8.py 5/5 (decomposition exact 7e-15; oracle loop == classical 1e-14). run_meta_b.py: Problem
  abstraction (dataset v1 | v2), variant learned_grad, v2 records via evaluate_v2; configs/meta_b.yaml v2 keys.
  Smoke: all variants on v1 and v2 OK. Grid launched: src/scripts/m5_grid_job.sh, 60 jobs = {f1, emulator} x
  {c1, c2, c3, c3o, c4, c4o, c5, c7, c7o, c8} x 3 seeds (strong / full, Adagrad teacher lr 0.3), seed 0 first;
  results tmp/results/m5_{f1,emulator}, learners tmp/meta/m5_*.
- 2026-11-15: M5 grid done (60 jobs, 180 records, 0 diverged; tmp/results/m5_summary_*.md). No meta cell beats
  classical Adagrad (10 gradients); cell 5 regularises against model error (better true-model forecasts); cell 7
  improves the first guess with 0 model calls; cells 3, 4 fail (identifiability, obs input does not help); cell 8
  overfits (train MSE 0.03 vs val 2.3) and fails at inference. Findings appended.
- 2026-11-15: Pilot: paper-ready reporting + intuitive impact (speed-up factor) + explicit reference in every table.
  src/measure_grad_cost.py -> tmp/results/grad_cost_ratio.json (gradient = 3.7 forward runs F1, 2.5 F2; emulator
  forward 4x slower than analytic L63). report_m5.py rewritten: every row vs reference "classical Adagrad, exact
  adjoint, same false model", cost in forward-run equivalents, speed-up x, method | ref columns, n/a when < 50 %
  reach the target. findings.md rewritten paper-ready (summary, methods, Tables 1-3 with references and speed-ups,
  findings, limitations).
- 2026-11-15: Cell 8 fix. learned_gradient.py: arch "adjoint" = learned one-interval tangent-linear operators
  A_theta(state) = I + MLP(state) combined by the EXACT adjoint recursion (linear in the innovations); data "sampled" =
  decorrelated controls x0_b + s L z (s in 0.5/1/2, LL^T = B) with emulator targets; loss "jacobian" = match each
  operator to the emulator's one-interval Jacobian at the forward-run state (gradient-level training through 11
  chaotic Jacobians was ill-conditioned: val cos 0.39); epoch selection on median relative gradient error (MSE
  dominated by a few far-from-minimum cases). test_cell8 8/8 (recursion with exact Jacobians == autodiff 1e-15,
  strong / weak; adjoint fit smoke val cos 0.997 on 64 cases). Smoke on F1 (100 train cases): 47/50 converged,
  IC 0.450 (fg 0.606), fc 1.52 ~ classical; old cell 8 diverged. Trajectory states cost 45x the collection time ->
  traj_stride 0 default. Old c8 pinned (arch mlp, data trajectory); new cell c8a launched: F1 / F2 x 3 seeds.
- 2026-11-15: Cell 8a done (F1 / F2 x 3 seeds, 18 records, 0 diverged). Val: median rel. error of grad J_o 0.059 F1 /
  0.015 F2, cos 0.9998. Test vs classical Adagrad (exact adjoint): x3.66 / x2.41 model-run speed-up, measured
  wall-clock x2.93 / x2.28 (src/measure_cell8_cost.py: network 9 ms per 200 cases); IC 0.418 / 0.419 (ref 0.425 /
  0.419), reached 1.00, forecast +-1 %, 0 worse basins; test_ood x3.42 / x2.36, IC 0.421 / 0.433. report_m5.py lists
  8 (v1) and 8a (v2). findings.md updated (summary, methods, Table 2 rows, Table 3 v1 vs v2, findings 4 and 9).
- 2026-11-16: Cell 8a smoke on weak/full and strong/x_only (F1, F2; 100 train cases): 0 diverged, 76-96 % converged,
  final J ~ classical (18.02 vs 18.00 weak F1). Full runs launched: 18 jobs = {weak/full, strong/x_only, weak/x_only}
  x {F1, F2} x 3 seeds (m5_grid_job.sh now takes [constraint] [pattern]; lr 0.3 strong / 0.1 weak).
  report_m5.py: --configs, per-config output files; FIXED a glob that would have mixed other configurations' meta
  records into the strong/full rows (strong/full table verified unchanged). measure_cell8_cost.py loops over configs.
- 2026-11-16: Cell 8a weak/full, strong/x_only, weak/x_only done (18 jobs, 54 records, 0 diverged). All configs:
  model-run speed-up x2.4-3.9, wall-clock x2.3-2.9 (consistent same-session formula, tmp/results/cell8a_speedups.json;
  measure_cell8_cost.py per config), IC within +-0.01 of the exact-adjoint reference, forecast within +-2 % (one
  -7 %), reached within 0.03. FD weak x0.05. findings.md: summary + Table 4 + finding 4 + limitations updated;
  strong/full wall-clock numbers harmonised to the same-session measurement (x2.94 / x2.26).
- 2026-11-16: src/measure_g1_v2.py -> tmp/results/g1_v2.json: G1 (cos to true-model gradient) for all M5 methods; findings Table 5.
- 2026-11-16: findings Table 6: time-to-solution + one-off meta-training time and break-even (from tmp/logs).
- 2026-11-16: Meta-speed check (pilot request). Step 1 floor: src/run_floor_v2.py -> tmp/results/floor_v2_test.json
  (test, strong/full): Adagrad 10 gradients; L-BFGS (strong Wolfe) 5.5 F1 / 6 F2 gradients (reached 1.00, IC 0.415 /
  0.402); Gauss-Newton 1 iteration (= 1 gradient + 3 TL runs ~ 3-4 gradient-equivalents; reached 0.99). Floor ~2-3x.
  Step 2: cell-5 option unroll.loss=sum (meta_a.py); sweep scripts src/scripts/speed_sweep_{job,run}.sh +
  speed_sweep_f1_val.txt; report src/report_speed.py. Val F1 (200 cases; classical val refs re-run on 200):
  selected cell 1 = Adagrad teacher lr 0.1, stretch bound 10, align 2: 7 vs 9.5 gradients (x1.36), reached 0.965,
  fc 1.96 vs 1.99. Cell 5 sum-loss: 3-6 gradients but reached 0.39-0.73 (none eligible); c5_sum_K20 taken to test.
  Final 3-seed F1/F2 test + test_ood runs launched (src/scripts/speed_sweep_final.txt).
- 2026-11-16: Final 3-seed test/test_ood of retuned cell 1 and cell 5 (sum loss K 20), F1/F2 -> findings Table 7.
- 2026-11-16: Model behaviour check (pilot): src/viz_models.py -> tmp/figs/models_{timeseries,phase,sensitivity}.png/pdf,
  tmp/results/models_behaviour.json. Same IC: F1 / F2 leave the truth (|err| > 1) after a median 0.36 / 0.35 TU; median
  |err| at 1.1 TU 1.81 / 1.85. F2 tracks F1 closely. IC sensitivity (|delta| 1e-3): separation growth 0.896 truth /
  0.888 F1 / 0.903 F2 per TU (fit 1-8 TU), median separation > 1 at 7.7 TU for all three.
- 2026-11-16: Dense-obs test (pilot): D1 v2-dense = same window 1.1 TU, obs EVERY step (111 obs), nested with D1 v2 (same
  12 obs + noise) and same F1 background (src/make_dense_obs.py -> tmp/data/l63v2d_*; fourdvar_v2.yaml flag `dataset`).
  src/run_dense_v2.py: strong/full BICE, F1 and E1, Adagrad / GD (dense lr by val rule: F1 0.1 / 1e-4, E1 0.3 / 1e-4),
  L-BFGS, Gauss-Newton; src/report_dense.py -> tmp/results/dense/summary_test.{md,csv}, tmp/figs/dense_{framework_case0,
  curves,forecast}. J* = best final J of the 4 optimizers. Dense obs: Adagrad 10 -> 28 gradients (F1), GD 125 -> 228,
  L-BFGS 6 -> 6, GN 1 iteration (~8 grad-eq with forward-mode TL). E1 IC 0.31 -> 0.20, fc@1TU 0.65 -> 0.25; F1 IC 0.42 ->
  0.38, fc@1TU (true model) 2.43 -> 2.30: model error dominates, extra obs barely help F1.
- 2026-11-16: Dense-obs WEAK constraint (pilot): run_dense_v2.py --constraint weak [--sigma_q] [--lbfgs_evals 150]
  (eta at every obs time: dense control 333 vs sparse 36; GN skipped). Configs: sparse sq 0.1, dense sq 0.1, dense sq
  0.1/sqrt10 (same variance per 0.1 TU); F1 and E1. Dense val sweeps: no lr converges in 1000 its; selected Adagrad
  0.03, GD 1e-5. report_dense.py --constraint weak -> summary_test_weak.{md,csv}, dense_*_weak figs; framework fig now
  uses L-BFGS (converged) analyses for strong and weak. F1 gradients to 1 %: Adagrad 41 -> 113 / 242, L-BFGS 9 -> 17 / 14;
  Adagrad reaches 0.60 / 0.57, GD 0.29 / 0.19. Operational F1 fc@1TU 1.24 -> 0.95 / 0.73; true-model fc of F1 analyses
  unchanged (2.3-2.7).
- 2026-11-16: Meta-learners on dense-obs WEAK (F1, sigma_q 0.1/sqrt10, control 333; teacher Adagrad lr 0.03 = classical
  val selection): src/scripts/dense_meta_{job,run}.sh (group m5dw_f1), cells c1 (retuned M5 speed config), c5 (sum loss
  K 20), c8a (samples_per_case 8 to keep ~0.9 M interval Jacobians; epochs 100 after a 200-case check showed val rel.
  err still falling at 20 epochs; final val rel. err 0.06, cos 0.9997) x 3 seeds, test + test_ood. Classical test_ood
  baseline via run_dense_v2.py --lr_adagrad 0.03 --lr_gd 1e-5 (timing json now split-suffixed; test-split val-sweep
  record kept in timing_*_q0.0316228_test_with_val_sweep.json). Meta config: new key data_prefix (renamed from my
  earlier `dataset` flag, which collided with meta_b's dataset=v2). Reports: src/report_dense_meta.py ->
  tmp/results/dense/meta_summary_{test,test_ood}.md, tmp/figs/dense_meta_curves_*; src/measure_cell8_cost_dense.py ->
  cell8a iteration 179 ms vs exact gradient 114 ms (x0.64 wall-clock).
- 2026-11-17: (1) L-BFGS teacher: ReferenceStep method 'lbfgs' = batched L-BFGS direction (two-loop, history 10, pairs
  from the moves actually applied, s.y > 0 kept), fixed step, no line search (src/dameta/da/reference_step.py;
  src/tests/test_lbfgs_teacher.py 2/2). Classical fixed-step L-BFGS: run_dense_v2.py --only_lbfgsfix, val rule ->
  lr 0.0333 (records *_lbfgsfix_*). Cell 1 with this teacher (dense_meta_job.sh METHOD=lbfgs LR=0.0333333, group
  m5dwL_f1). First run: theta became NaN in episode 5 (a few cases blew up under the x11 stretch) -> FIX in
  MetaB.meta_update: non-finite cases excluded from the meta losses, non-finite loss / grads skip the Adam step (no
  effect on finite runs; test_m4 11/11). Re-run: 3 seeds OK, 3-6 % cases diverge at inference.
  (2) Cell 5 K 50 (c5k50, 3 seeds). report_dense_meta.py: + fixed-step L-BFGS row (also in J*), c1L, c5k50 rows.
