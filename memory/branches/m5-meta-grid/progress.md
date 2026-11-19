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
- 2026-11-17: Missing-physics diagnostic (src/diag_eta_physics.py; dense weak F1 sq 0.1/sqrt10, test): eta of the
  converged L-BFGS analysis vs the known missing physics M_true(x) - M_F1(x) (= dt * 0.5 x in y): corr y 0.43 (0.50 at
  true states) but regression slope 0.05 (Q prior shrinks it ~20x); x / z components are artifacts (rms 0.003 / 0.0085
  vs missing 0.002 / 0.0024). Adagrad eta (not converged): corr <= 0.15, rms 5-10x the signal = mostly artifacts.
  User redirect: priorities are (1) amortized inference cost of the L-BFGS-taught learner over many cases,
  (2) whether a meta-learner gives more realistic analyses (recover missing physics, remove artifacts).
- 2026-11-17: Cell 1 with the L-BFGS LINE-SEARCH teacher (user request: cell 1 beats its teacher, so give it the
  strongest one). ReferenceStep method 'lbfgs_ls' = L-BFGS direction + batched backtracking Armijo (c1 1e-4, shrink
  0.5, <= 20 trials; first step lr along g/|g|, then 1); every trial forward of an active case counted (n_fwd ->
  record ls_fwd_cum [ic, k], cumulative per iterate) in run_fourdvar / run_meta_b. Classical teacher alone:
  run_dense_v2.py --only_lbfgsls -> <tag>_lbfgsls_<split>.nc (lr 1, no tuning). Cell 1: dense_meta_job.sh
  METHOD=lbfgs_ls LR=1 GROUP=m5dwLS_f1 (same c1 config), list src/scripts/dense_meta_c1LS.txt. report_dense_meta.py:
  rows lbfgsls / c1LS, cost = iterations x r + line-search forwards (cost_to). test_lbfgs_teacher.py +1 test (3/3).
- 2026-11-17: Results cell 1 / Armijo L-BFGS teacher (3 seeds, test + test_ood) -> findings "Cell 1 with the L-BFGS
  line-search teacher": 60 vs teacher 63 vs strong Wolfe 53 fwd-eq (test), 66 / 66 / 53 (test_ood); reached 0.95 vs 0.98;
  learned stretch median x1.35. No gain over a line-search teacher. M4 regression test 19/19 after the teacher change.
- 2026-11-17: User: no new method; the reference is strong-Wolfe L-BFGS (opt_lbfgs). Armijo teacher 'lbfgs_ls' REMOVED
  (code, report rows, findings subsection; its records stay in tmp/results/m5dwLS_f1 and dense/*_lbfgsls_* for
  the record only: cell 1 60 vs teacher 63 fwd-eq, learned stretch x1.35, no gain). Replaced by ReferenceStep
  'lbfgs_wolfe' = one torch.optim.LBFGS(strong_wolfe) iteration with opt_lbfgs settings (torch's _strong_wolfe; first
  trial batched, failing cases per case); test_lbfgs_wolfe_matches_torch_lbfgs checks identical iterates. Cost
  bookkeeping grad_cum [ic, k]: outer gradient not counted when the control equals the last line-search point
  (classical reuse), line-search trials counted as gradients. Consequence: a learner that moves away from the
  accepted point pays outer + line-search gradients (>= 2 per iteration) -> it must halve the iterations to win.
  Runs: run_dense_v2.py --only_wolfe_check (teacher alone, must reproduce opt_lbfgs); cell 1 K 20 / 50 / 100
  (dense_meta_job.sh c1 | c1k50 | c1k100, METHOD=lbfgs_wolfe LR=1 GROUP=m5dwW_f1, list dense_meta_c1W.txt), 3 seeds.
- 2026-11-17: User: no need for K 100 -> c1k100 dropped (seed 1 killed; seed 0 record kept in tmp/results/m5dwW_f1,
  not reported). Remaining c1 / c1k50 seed 2 relaunched via src/scripts/dense_meta_c1W_rest.txt. Wolfe check (teacher
  alone, to its stopping rule): median 40 / 43 gradient evaluations on test / test_ood.
- 2026-11-18: User: runs too long -> stopped (c1k50 s1, c1 s2, c1k50 s2 killed during training; no records). Final:
  K 20 seeds 0-1, K 50 seed 0. Findings subsection "Learned optimizer with the reference L-BFGS as teacher" written
  (paper style): 11-12 iterations as the reference, 22.5-24 vs 14 gradients, cost 89-92 vs 53 fwd-eq, learned stretch x1.3,
  K 50 = K 20. Training 7-10 h per seed (per-case strong-Wolfe fallback in a Python loop).
- 2026-11-18: Pilot: simplified delta learner ("direct", cell c1D): no delayed estimation, no chain rule, no linear
  model of J, no ALIGN; target = teacher delta x^{k+1} - x^k; same input (x, g_std, emb(k)); inference without teacher
  (1 gradient / iteration, no line search). src/dameta/meta/meta_b_direct.py (DirectNet: direction head + per-block
  log-magnitude head, Delta = -RMS_block(g) * u; teacher pairs collected once per teacher run, cached for seeds;
  best epoch on val-pair relative error), run_meta_b.py variant=direct, configs/meta_b.yaml block `direct`,
  dense_meta_job.sh cell c1D. tests/test_direct_delta.py 6/6 (GD teacher reproduced: val rel. err 1.6e-10).
  First head (plain u) failed: teacher step in gradient units spans ~4 decades (|u| 4e-5 at k 0 -> 1e-2 at k 40).
  Smoke dense weak F1, strong-Wolfe L-BFGS teacher, 50 train / 20 val cases: train rel. err ~0.9 (= no move 1.0),
  predicted step collapses to ~0 x teacher after k 3 (cos(pred, teacher) 0.92 at k < 3, 0.2 at k > 30); the
  L-BFGS direction is history-determined (cos(teacher step, -g) 0.14-0.2) and not a function of (x, g, k) ->
  regression to the mean. grad_norm_input flag does not change it. Adagrad-teacher smoke running.
- 2026-11-18: Direct-delta: best epoch now on MEDIAN val-pair relative error (mean dominated by a few pairs, kept an
  undertrained epoch). Smoke (dense weak F1, 50 train / 20 val cases; classical on the same 20 val: Adagrad 0.03 J 177.4
  at 1000 gradients, 7/20 converged; strong-Wolfe L-BFGS J 170.9 at 40 gradients, IC 0.282):
  Adagrad teacher: train rel. err 0.013, val median 0.67 (overfit at 50 cases), cos(pred, teacher) 0.83 val; frozen
  run J 173.4 at 1000 gradients (below its teacher 177.4), IC 0.368, 0 diverged.
  L-BFGS (strong Wolfe) teacher: val median 1.0, J 190.3 at 150 gradients, IC 0.662 (worse than first guess 0.413):
  not imitable from (x, g, k). Pairs size: 350 MB per 50 Adagrad train cases (needs pair subsampling for 1000).
- 2026-11-18: Pilot: dev testbed = ONE seed, 10 train cases, 20 val cases (src/scripts/dev_delta.sh, dense weak F1,
  result group dev_delta). Delta module = two versions: (1) supervised "direct" (meta_b_direct.py, unchanged);
  (2) unsupervised corrected delta = variant delta_exact (src/dameta/meta/meta_b_exact.py): teacher in the loop,
  x^{k+1} = x^k - f_teacher * max(1 + s_theta(x^k, g^k, k), 0) (StretchNet, ONE stretch, no K chain), theta updated
  at k+1 by the exact chain rule grad J(x^{k+1}) . (-d delta / d theta) (gradient the next iteration needs anyway),
  one Adam step per iteration, no ALIGN / delayed replay; zero-init -> first move = teacher step; horizon = teacher
  mean iterations; best epoch on val median J(x^K)/J(x^0) (epoch -1 = teacher alone). Inference run_meta_b, MetaB(K=1).
  A teacher-free unsupervised mode written in between was removed (pilot: the teacher stays in the loop).
  tests/test_direct_delta.py 11/11 (s = 0 == classical Adagrad exactly; chain-rule theta-gradient == FD 6e-8 / 2e-8).
  Dev: direct + L-BFGS (Wolfe) teacher on 10 cases fails again (J 3155, IC 1.33). delta_exact dev runs launched.
- 2026-11-18: Why direct (supervised) fails with the L-BFGS teacher (offline fits, 20 val cases; tmp/logs/imitation_diag.log):
  default net 10 cases train median rel 0.105 / val 0.978 (memorises, no generalisation); 50 cases 0.494 / 0.971
  (5x data: no val gain); big 512x3 nets did not train (lr 1e-3; inconclusive). Teacher step vs -g cos 0.20.
  Scaling test 10/50/200/1000 cases + history input (s^{k-1}, y^{k-1}) launched (tmp/logs/imitation_scaling.log,
  teacher pairs cached tmp/meta/dev_delta/imit_pairs_wolfe.pt).
  delta_exact + Wolfe teacher (10 cases, 50 epochs, K 50): best epoch = -1 (teacher alone; 170.90, 40.5 gradients);
  learned stretch ~0.95; score J(x^K)/J(x^0) was blind to speed (teacher converges < K) -> new delta_exact.score
  "mean" (mean_m J(x^m)/J(x^0)), default.
- 2026-11-18: Timing (local 4-core node, load ~13): dense-weak gradient 218 ms batched (10 cases) vs 276 ms single
  case -> per-op overhead (~2200 tiny torch ops + autograd backward), not arithmetic. Wolfe teacher, 50 its x 10
  cases: 70 s, of which 57 s per-case line search (585 single-case gradients); Adagrad 12 s. Pilot: use the HPC.
  New src/scripts/sb_parallel.sh (sbatch, node sl-mee-br-205, command list in parallel, 1 thread each);
  src/diag_imitation.py (collect / fit modes) replaces the local scaling script; jobs 78133 (collect 1000 train in
  20 chunks + 20 val) -> 78134 (7 fits: n 10/50/200/1000, history input). Lists in src/scripts/lists/.
- 2026-11-18: Pilot correction: `delta_exact` = SUPERVISED (loss Delta - Delta_hat, teacher update vs prediction; my
  former `direct`, renamed: variant delta_exact, config block delta_exact); `delta` = UNSUPERVISED; at inference
  NEITHER uses the teacher. Teacher-stretch "delta_exact" (meta_b_exact.py) removed. Write the update as Delta.
  New unsupervised delta (delta.output=free, default; src/dameta/meta/meta_b_free.py): module outputs Delta_hat
  (DirectNet, K-sub-step inner rollout), one meta step per iteration: w_perf <g^k, Delta_hat^{k-1}(theta)> / <g^{k-1},
  f^{k-1}> through the STORED graph of the move (no replay) + w_align rel|Delta_hat(x^k) - Delta^k|^2 + reg; teacher
  only in training. Original Meta-B kept as delta.output=stretch and pinned in the cell-1 scripts. Tests
  test_direct_delta 14/14 (stored-graph gradient == FD, K 5, 9e-9 / 2e-8), test_m4 19/19.
  Imitation scaling on HPC (jobs 78133 collect 3 min, 78134 fits 6.5 min; val = 20 cases, Wolfe teacher):
  no history: val median rel 0.978 / 0.971 / 0.964 / 0.950 for 10 / 50 / 200 / 1000 cases (cos 0.49 -> 0.56);
  + history (s^{k-1}, y^{k-1}): 50 cases 0.966; 1000 cases 0.855 (cos 0.77), lr 3e-4 0.872 (cos 0.82); still
  improving at the last epoch. -> data alone helps little; the L-BFGS memory is the main missing information.
  Dev runs (job 78179): delta_exact and delta (free) x {Wolfe, Adagrad} on 10 train / 20 val cases.
- 2026-11-18: Pilot: never use the login node; HPC only (sbatch, tracked by a background squeue wait). Dev job 78179
  was broken by my wrapper: xargs stripped the quotes, so each run got only its FIRST override (delta runs: no
  select_epoch, episode_iters 100), delta_exact crashed (KeyError K: block lacked K). Results of that job are
  invalid (delta free Wolfe J 230.9 / IC 1.32, Adagrad J 267.1 / IC 0.74 with wrong settings). Fixed: sb_parallel.sh
  xargs -d '\n'; dev_delta.sh per-run hydra.run.dir (same-second runs shared one dir); delta_exact.K: 1.
  Resubmitted as job 78193 (+ test_direct_delta on the HPC).
- 2026-11-18: Job 78193 (valid settings, 10 train / 20 val; tests passed on HPC). delta_exact: Wolfe J 3155 / IC 1.33
  (no imitation), Adagrad J 190.8 / IC 0.466 / 1000 gradients. delta (free): Wolfe 7/20 diverged, Adagrad 12/20
  diverged; diverged cases blow up at ITERATION 1 (first predicted step too large: step proportional to |g| while the
  L-BFGS first step is normalised by |g|_1; the net never sees |g|); val score (median J at k 10) hid the minority.
  Fixes: delta.grad_norm_input (log RMS_block g input), delta.select_stat q90 (diverged = inf, default).
  Job 78206: + grad-norm input for delta_exact / delta x {Wolfe, Adagrad}, delta Wolfe q90-only control.
- 2026-11-18: Job 78206 (10 train / 20 val; tests passed). grad-norm input: delta_exact Wolfe J 3155 -> 199.2 (IC
  0.541), Adagrad unchanged 190.8 (IC 0.465); delta (free) Wolfe 1/20 diverged, J 197.3, IC 0.456; Adagrad 0/20
  diverged, J 190.6, IC 0.526 (best epoch 0); q90-only control (Wolfe, no grad norm): 0/20 diverged but J 244.9,
  IC 1.08. No configuration beats its teacher (Wolfe 170.9 / IC 0.282 / 40 gradients; Adagrad 177.4 / 0.424 /
  1000) and every one ends worse than the first guess (IC 0.413). Awaiting pilot decision on the next step.
- 2026-11-18: Pilot: grad-norm input NOT accepted (removed from code and config). Hypothesis (pilot): the delta module
  learns an optimisation curve better than across different problems; cell 1 worked on the long dense-obs weak
  problem (111 obs, control 333). -> observation input (obs_input flag, delta_exact and delta): per-case features
  [y standardised by train-truth climatology and masked (3 n_obs), mask (3 n_obs), times t_i/L (n_obs)] appended to
  [x_std, g_std, emb(k)] (src/dameta/meta/da_utils_obs.py). Teacher pairs now carry the case index (own collector;
  old caches without it are re-collected). test_direct_delta + test_obs_input. Job 78258: delta_exact / delta x
  {Wolfe, Adagrad} with obs + delta Adagrad q90 control without obs (tests run in the same job).
- 2026-11-18: Job 78258 (obs input; 10 train / 20 val; classical on these cases: Wolfe 170.9 / IC 0.282 / 40 grads,
  Adagrad 177.4 / 0.424 / 1000; first guess IC 0.413). delta_exact + obs: Wolfe J 3155 -> 190.4 (IC 0.648; train rel
  0.44 vs val 0.997), Adagrad J 190.8 -> 179.6 (IC 0.466 -> 0.443; train rel 0.004 vs val 0.86: heavy overfit with
  777 obs inputs on 10 cases) = closest to its teacher so far. delta (free) + obs: Wolfe J 659 / IC 4.24, Adagrad
  J 473 / IC 0.73 (worse). delta Adagrad control without obs (q90): best epoch -1 (= untrained net), J 190.8,
  IC 0.390; every trained epoch was worse (q90 J at k 10 -> divergent). The test script had a syntax error (my
  escaped backslash) -> fixed, rerun alone as job 78276.
- 2026-11-18: Tests 18/18 on HPC (job 78276). Pilot: flat obs input (777 values) too large -> arrange in time.
  New src/dameta/meta/temporal_net.py (arch temporal, needs obs_input; pilot chose 1D conv): one token per obs time
  [ctrl_i (x0 / eta_i), g_std_i, y_i, mask_i, t_i/L, emb(k)], shared Linear embedding + n_layers residual Conv1d
  (kernel 5) along time, per-token direction head, per-block log step size from token 0 (x0) / mean of tokens >= 1
  (eta); same output law and interface as DirectNet. test_temporal_net (time locality checked). Job 78301: delta_exact
  / delta x {Wolfe, Adagrad}, temporal, hidden 32.
- 2026-11-18: Job 78301 (temporal net, hidden 32; tests 20/20 on HPC). delta_exact + Adagrad: J 177.03 / IC 0.431 /
  fc@1TU 1.015, 0 diverged, 1000 gradients, NO teacher at inference = matches classical Adagrad (177.4 / 0.424) on the
  20 val cases (flat obs: 179.6 / 0.443; no obs: 190.8 / 0.466); train rel 0.064, val rel 0.977. delta (free) + Wolfe:
  J 183.4 / IC 0.394 (below first guess 0.413; best free-delta result so far; flat obs 659 / 4.24), 0 diverged, 150
  gradients cap. delta_exact + Wolfe: fails (J 11086, 2 diverged). delta + Adagrad: J 337 / IC 1.25.
- 2026-11-18: Recap for the pilot: delta_exact Wolfe = 1 Adam step / epoch (494 pairs, batch 1024; train rel 0.86 at
  epoch 300 = does not fit its own 10 cases), Adagrad = overfit (val best at epoch 1); teacher step in gradient units
  spans ~4 decades (Hessian-spectrum fingerprint), inputs drop |g| and scale x by climatology. Built (meta_b_direct.py):
  coords cvt (control-variable transform, case Cholesky of hybrid B), grad_input fixed, obs_feat innov (grad_innov:
  J, g, innovations from one forward/adjoint run), optimizer lbfgs (full batch, strong Wolfe), max_train_pairs; free
  delta asserts raw. src/diag_cvt.py (step 1: r = log10 |Delta|/|g| per block raw vs cvt, spread, R^2 on log|g| and
  log1p k, cos(Delta, -g)). Tests test_direct_delta +3 (cvt: L L^T = B, g_v = autograd dJ/dv, init = GD in v; grad_innov;
  cvt + innov + L-BFGS fit). HPC: gate job 78424 (tests, diagnostic, 2 smokes; list dev_delta_v5_check.txt) ->
  job 78425 afterok (list dev_delta_v5.txt: {Wolfe, Adagrad} x {raw, cvt} x {Adam batch 64, L-BFGS}, temporal h32).
- 2026-11-18: Gate job 78424 passed (test_direct_delta 23/23). Step-1 diagnostic (tmp/logs/diag_cvt.log,
  tmp/results/dev_delta/diag_cvt.json): the control-variable transform does NOT shrink the spread of the teacher step
  in gradient units r = log10 |Delta|/|g| (Wolfe x0 3.2 -> 3.5 decades, eta 1.6 = 1.6: per-block constant scaling
  only shifts r by log10 sigma_q^2 = -3); it lowers cos(Delta, -g) (Adagrad 0.85 -> 0.23: Adagrad is diagonal in RAW
  coordinates). R^2 of r on log|g| 0.35-0.53 (gradient magnitude explains ~half), on log1p(k) 0.43 for Wolfe x0
  (L-BFGS first step normalised: r -4.7 at k 0 -> -2.1 at k 40), ~0 elsewhere. -> "badly conditioned" in the sense
  of coordinates is refuted; magnitude information (|g|) is the useful part. Smokes OK (Wolfe cvt L-BFGS 2 epochs x
  3 its: J 193.3 / IC 0.295, 0 diverged). Job 78425 (8 runs) running.
- 2026-11-18: Job 78425 (8 runs, 70 min; 10 train / 20 val, delta_exact temporal h32). Val results (J / IC / diverged;
  best epoch, best val rel, train rel median at that epoch). Classical on these 20: Wolfe 170.9 / 0.282, Adagrad
  177.4 / 0.424; first guess IC 0.413. Previous raw Adam-1024 baselines: Wolfe 11086 / -, Adagrad 177.0 / 0.431.
    Wolfe raw  Adam64  16003 / 27.5 / 0   (ep 109, val 0.971, train 0.918)
    Wolfe raw  L-BFGS   7297 / 5.14 / 0   (ep 2,   val 0.984; end train median 0.128)
    Wolfe cvt  Adam64  190.3 / 0.371 / 0  (ep 138, val 0.978, train 0.966)
    Wolfe cvt  L-BFGS  192.7 / 0.376 / 1  (ep 4,   val 0.981, train 0.973)
    Ada   raw  Adam64  174.0 / 0.391 / 0  (ep 0,   val 0.863, train 0.163; end train 0.005, val 1.08)
    Ada   raw  L-BFGS  173.2 / 0.395 / 0  (ep 1,   val 0.829, train 0.106; end train 0.003, val 1.11)
    Ada   cvt  Adam64  203.9 / 0.376 / 5  (ep 0,   val 0.770)
    Ada   cvt  L-BFGS  259.0 / 0.411 / 0  (ep 1,   val 0.882)
  Reading: (2) more optimisation fits the train pairs (Adagrad train 0.003, Wolfe raw L-BFGS 0.13 vs 0.86 before) but
  val does not improve (best epoch 0-1, then overfit): limit = generalisation from 10 cases, not under-training;
  L-BFGS vs Adam no difference on val. (3) cvt: no imitation gain (Wolfe val 0.98 everywhere, train cannot fit
  0.97); it only makes the Wolfe-taught learner STABLE because its initial rule is GD preconditioned by B (selected
  learners stay near init); with Adagrad cvt is worse (Adagrad is diagonal in raw coordinates). Raw Adagrad learners
  end below the teacher on these val cases (J 173.2-174.0 vs 177.4, IC 0.39 vs 0.42) at the same 1000 gradients,
  but with an almost untrained network selected on the same val cases (caveat: dev testbed, val = selection = eval).
  Wolfe teacher still not imitable from (x, g, k, obs) in any coordinates (val rel >= 0.97).
- 2026-11-18: History learner built (meta_b_direct.py: lbfgs_pair / lbfgs_two_loop (== LBFGSDirection) /
  attach_history, base_dir, run_direct own memory + first-step teacher with is_free cost bookkeeping, fit epoch -1
  candidate in hist mode; temporal_net.HistTemporalNet; dev_delta.sh env NTR). src/report_dev_hist.py: classical
  Wolfe on the 20 val cases, teacher step-length distribution, cost-to-target (tol 1e-2 / 1e-3) + speed-up per run.
  Tests +4 (two-loop == LBFGSDirection; Wolfe step = t * two-loop direction of its memory; init + first step + cost
  bookkeeping; hist fit / checkpoint). Jobs: gate 78466 -> runs 78467 (h6_lbfix, h6_{lb,g}_n{10,100}) -> report 78468.
- 2026-11-18: Gate 78466 passed (tests incl. the 4 history tests). Smoke (2 epochs) kept epoch -1: the teacher's
  median step length is exactly 1, so the untrained base-lbfgs learner has val MEDIAN rel error 0 (mean dominated by
  the few line-search-shrunk steps: train mean 1800). Untrained = fixed-step L-BFGS after a Wolfe first step: J 170.08
  (teacher 170.9), 41 gradients, 1/20 diverged. -> cancelled 78467/78468; new options delta_exact.train_loss
  (mean | logmean = mean log(1 + rel)) and delta_exact.select (median | logmean | mean); v6 resubmitted with
  select logmean (+ train_loss logmean variants): job 78469 (7 runs) -> report 78470.
- 2026-11-19: Job 78469 (7 history runs, 25 min) + report 78470 (tmp/logs/report_dev_hist.log). 20 val cases; classical
  strong-Wolfe L-BFGS: J* 170.90, IC 0.282, 15.0 / 27.0 gradients to 1 % / 0.1 % of J* (reached 1.00 by definition).
  Teacher step length t (k >= 1, 708 val pairs): t = 1 in 91 %, t < 1 in 8 % (p5 0.28) -> the line search costs
  little: L-BFGS is already ~1 gradient / iteration; the headroom of a learned line search is ~10 %.
    run (cost to 1 % | 0.1 %; reached; speed-up on cases both reach)       J / IC / div
    h6_lbfix (untrained = fixed-step L-BFGS after a Wolfe first step)  13.5 (0.90, x1.07) | 23.0 (0.85, x1.04)  170.08 / 0.321 / 1
    h6_lb_n10        12.5 (0.80, x1.08) | 15.5 (0.50, x1.06)   170.41 / 0.370 / 0   (val logmean 0.298 -> 0.247)
    h6_lb_n10_log    13.5 (0.80, x1.00) | 19.0 (0.65, x1.05)   170.53 / 0.283 / 3
    h6_lb_n100       13.0 (0.75, x1.00) | 19.0 (0.65, x1.05)   176.48 / 0.345 / 0   (0.298 -> 0.164)
    h6_lb_n100_log   13.0 (0.80, x1.08) | 16.0 (0.65, x1.25)   174.83 / 0.328 / 0   (0.298 -> 0.149)
    h6_g_n10 / n100   9.0 (0.10)        | 15.0 (0.05)          187 / 0.407 / 0      (history-only input fails)
    previous obs-input learners: dx_wolfe_tmp 48.5 (0.20, x0.12); dx5_wolfe_cvt_adam 22.5 (0.30, x0.40)
  Reading: history + L-BFGS base = first learners at L-BFGS speed (vs x0.1-0.4 before) and imitation improves with data
  (100 cases halve the val error), but none beats L-BFGS: same speed on the cases reached, fewer cases reach the
  target (0.65-0.80 vs 1.00). History as INPUT with a -g base does not work (L-BFGS direction is not learnable from
  10 cases of pairs). Imitation is capped at the teacher's iteration count; to beat L-BFGS the learner must take
  better steps than the teacher -> needs an unsupervised objective (J after the move / unrolled J) starting from
  the L-BFGS base.
- 2026-11-19: Built attach_horizon (multi-step target) and run_direct safeguard backtrack; tests +2 (horizon target,
  backtrack monotone + gradient count); dev_delta.sh env PC_OVERRIDE (reuse a pair cache). Jobs: tests 78568 ->
  v7 runs 78569 (safeguard on v6 learners; h 2 n10 / n100, h 3 n100, each + safeguarded inference) -> report 78570.
- 2026-11-19: v7 done (tests 29/29; runs 18 min; tmp/logs/report_dev_hist_v7.log). 20 val cases, classical Wolfe
  L-BFGS 15.0 / 27.0 gradients to 1 % / 0.1 % (reached 1.00). (cost to 1 % (reached, speed-up) | 0.1 % ...; J / IC / div)
    h6_lbfix 13.5 (0.90 x1.07) | 23.0 (0.85 x1.04)  170.08/0.321/1  ->  +backtrack 14.5 (1.00 x1.03) | 26.0 (0.95 x1.00)  171.00/0.304/0
    h6_lb_n10 12.5 (0.80 x1.08) | 15.5 (0.50)        170.41/0.370/0  ->  +backtrack  8.5 (0.50 x1.12) | 14.0 (0.35)        178.41/0.349/0
    h6_lb_n100_log 13.0 (0.80 x1.08) | 16.0 (0.65 x1.25) 174.83/0.328/0 -> +backtrack 11.5 (0.70 x1.00) | 16.0 (0.55 x1.12) 176.48/0.295/0
    h7_h2_n10  12.0 (0.85 x1.17) | 21.0 (0.80 x1.12)  169.49/0.305/2  ->  +backtrack 12.0 (0.80 x1.12) | 18.0 (0.60 x1.06)  171.02/0.297/0
    h7_h2_n100 11.0 (0.80 x1.23) | 16.0 (0.55 x1.12)  175.79/0.323/2  ->  +backtrack 11.0 (0.75 x1.18) | 17.0 (0.60 x1.12)  170.98/0.334/0
    h7_h3_n100 13.0 (0.75 x1.00) | 27.0 (0.55 x0.67)  175.43/0.299/1  ->  +backtrack 10.0 (0.70 x1.15) | 17.0 (0.55 x1.06)  171.82/0.368/0
  Reading: backtrack removes every divergence (0/20) at no model-run cost and makes the untrained fixed-step L-BFGS as
  reliable as the classical one (reached 1.00 / 0.95) at the same cost; it does not raise the reached fraction of the
  trained learners (they settle at other local minima: median J up to 178 vs 170.9, not slower convergence). Horizon 2
  gives the best speed on reached cases (x1.12-1.23 to 1 %) but reached 0.75-0.85; horizon 3 no better. With 20 val
  cases one case = 0.05 reached: differences between runs are mostly within noise. No setting beats L-BFGS robustly.
- 2026-11-19: Online training order built (online_groups; train_mode / online_cases; test_online_groups). Jobs: tests
  78571 -> v8 runs 78572 (online h 2 n100 / n10, online h 1 n100 control; each + backtrack inference) -> report 78573.
- 2026-11-19: v8 done (tests 30/30; runs 24 min; tmp/logs/report_dev_hist_v8.log). Online = teacher-run order, 1 Adam
  step per iteration (14100 steps n100 / 31500 n10). (cost to 1 % (reached, speed-up) | 0.1 %; J / IC / div)
    h8_on_h2_n100   11.0 (0.75 x1.18) | 24.0 (0.70 x0.92)  172.59/0.319/3  -> +backtrack  8.0 (0.55 x1.25) | 15.0 (0.55 x1.20)  177.78/0.339/0
    h8_on_h1_n100   15.0 (0.85 x0.93) | 23.0 (0.75 x1.00)  174.85/0.287/1  -> +backtrack 13.0 (0.75 x1.00) | 17.0 (0.55 x1.06)  173.40/0.354/0
    h8_on_h2_n10    11.0 (0.85 x1.27) | 18.5 (0.80 x1.27)  172.42/0.334/2  -> +backtrack 12.5 (0.90 x1.16) | 20.0 (0.85 x1.20)  170.88/0.330/0
    (offline v7 +backtrack: h2_n10 12.0 (0.80 x1.12); h2_n100 11.0 (0.75 x1.18))
  Val imitation (logmean): online h2 n100 0.534 -> 0.393 (ep 12); h1 n100 0.298 -> 0.171; h2 n10 0.517 -> 0.457 (ep 1,
  then overfits). Two-step target is harder to imitate (0.39 vs 0.17 for one step). Best balance so far: online h2 n10
  + backtrack (reached 0.90 / 0.85, x1.16 / x1.20, 0 div, J 170.88) but its selected learner is barely trained (epoch 1)
  and differences between runs are 1-3 of 20 cases -> needs the 200-case val evaluation before any claim.
- 2026-11-19: v9 = 200-val re-scoring, inference only, backtrack safeguard (job 78614: 2 min total; learners 78-95 s
  each, classical reference in 10 chunks; tmp/logs/report_dev_hist_v9_val200.log). Classical strong-Wolfe L-BFGS:
  J* 168.63, IC 0.332, 14.0 / 25.0 gradients to 1 % / 0.1 % (reached 1.00).
    v9_lbfix_sg (untrained = fixed-step L-BFGS)  13.0 (1.00 x1.08) | 25.0 (0.96 x0.96)  168.61 / 0.334 / 0 div
    v9_h2_n10_sg   (offline 2-step, 10 cases)    13.0 (0.88 x1.00) | 24.0 (0.75 x0.88)  169.01 / 0.335 / 0
    v9_h2_n100_sg  (offline 2-step, 100 cases)   10.0 (0.76 x1.10) | 19.0 (0.58 x1.05)  169.42 / 0.365 / 0
    v9_on_h2_n10_sg (online 2-step, 10 cases)    11.0 (0.89 x1.18) | 21.0 (0.83 x1.07)  168.98 / 0.345 / 0
  Reading: on 200 cases no learner beats L-BFGS robustly. The safeguarded untrained fixed-step L-BFGS equals the
  classical one in quality and reliability (1.00 / 0.96) at x1.08 / x0.96. Trained two-step learners are at most
  x1.1-1.2 faster on the cases they reach, but miss 11-24 % (1 %) / 17-42 % (0.1 %) of cases; the median final J and IC
  stay close to classical (169.0-169.4 vs 168.6; IC 0.335-0.365 vs 0.332). The 20-case ranking held qualitatively.
- 2026-11-19: Timing (src/time_dev_hist.py, job 78620; 200 val cases in ONE batched run per method, 1 thread, own core,
  to the method's stopping rule; every method ran the full 150-iteration budget because some case never met tol):
  classical strong-Wolfe L-BFGS 225.9 s (1506 ms / iteration, incl. the per-case Python fallback of the line search);
  learners with backtrack: lbfix 89.9 s (599 ms/it), h2_n10 79.1 s (528), h2_n100 86.8 s (578), online h2_n10 72.9 s
  (486). Wall x2.5-3.1 lower than classical with the same 150 iterations: no line-search trials and no per-case loop;
  the network cost is small. Report column added to report_dev_hist.py (tmp/logs/report_dev_hist_v9_val200.log).
- 2026-11-19: Branch tidy (pilot): findings.md gained the subsection "Learned Δ optimizer without the teacher at
  inference: conditioning, history, multi-step targets" (methods, 200-val table with wall clock, findings, exact
  reproduction commands incl. the Table 7 and dense-meta configurations). Then removed all generated command lists:
  src/scripts/lists/ (dev_delta_v1-v9*, imit_*, tests_delta), src/scripts/dense_meta_*.txt, speed_sweep_*.txt.
  Their content is in findings.md (Reproduction paragraphs); runners (dev_delta.sh, sb_parallel.sh, *_job.sh, *_run.sh)
  kept. Pilot question answered: learners beat fixed-step L-BFGS (and Adagrad), not line-search L-BFGS.
