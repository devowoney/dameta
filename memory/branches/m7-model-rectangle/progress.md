# Progress — branch m7-model-rectangle

- 2026-11-20: Pilot redefined the model set as a rectangle (L1, F1, F2, E1, E2). Plan approved
  (~/.claude/plans/the-previous-experiment-was-melodic-hickey.md). Worktree created from main 2a92c57; D1 and D1 v2
  truth / obs copied into tmp/data (m5 copy, md5 = m6 copy).
- 2026-11-20: Code: src/configs/models.yaml + src/dameta/dynamics/models.py (registry, perturbed_params, build_model);
  src/screen_teachers.py refactored (screen()) + `--sweep` ε mode; src/tests/test_models.py; run_4dvar_v2 /
  fourdvar_v2.yaml use the registry (model l1|f1|f2|e1|e2, legacy `emulator`; BPE prior = model's own θ);
  gen_data_v2 `registry_models` + `backgrounds_only`; tune_hybrid_b defaults f1 f2 e1 e2; src/report_models.py.
- 2026-11-20: ε screen submitted (job 79427 failed on YAML quoting of ${eps1}; fixed, job 79428).
- 2026-11-20: Screen done (job 79428; tests PASS; tmp/results/models/false_screen.{csv,md}, tmp/figs/false_screen.*).
  m(ε) is linear (≈ 0.052 σ_o per 1 %); λ₁ 0.89-0.96, all stable 20 TU. F1 = ε 3 %: θ (10.17, 28.48, 2.71),
  ‖Δθ‖ 0.52, m 0.155 (≈ v2 ρ 27.5: 0.50, 0.139). F2: rule hits the GRID EDGE (ε 10 % → m 0.52 only); m ≈ 1 needs
  ε ≈ 19 %. Waiting for pilot: extend the grid (edge rule, as for lr grids) or accept a weaker F2.
- 2026-11-20: Pilot accepted the edge extension and asked for a probabilistic inspection of the models. Extended
  screen (job 79429): F1 = ε 3 % θ (10.17, 28.48, 2.71), m 0.155, energy distance to L1 0.021 = noise floor 0.021
  (same climate, trajectory error only; W1_z 0.42); F2 = ε 18 % θ (11.04, 30.91, 2.94), ‖Δθ‖ 3.10, m 0.958,
  error q05/q50/q95 0.21/0.86/1.54, λ₁ 1.02 vs 0.89, z mean 26.1 vs 23.6, energy 0.33 (16x floor): distinct
  climate. models.yaml eps1 0.03 / eps2 0.18. train_emulator `+registry_model`. E1 / E2 training jobs 79430 / 79431
  (10k steps); chain 79432 (afterok): backgrounds f1 f2 e1 e2 + hybrid B, QC e1 e2, report_models, tests.
- 2026-11-20: E1 / E2 trained (best val rmse@50 0.030 / 0.038, ~330 s each); backgrounds f1 f2 e1 e2 + hybrid B
  (f2 / e2 at grid edge α 1, χ² 6.3); QC e1 e2; rectangle report (job 79432, figure fix 79433: ECDF panel); tests PASS.
  Findings entry written (branch findings.md). Open for pilot: small horizontal axis (Problem 2 signal), background
  confound on the vertical axis.
- 2026-11-20: Pilot: no unrequested result write-ups. Asked for an ensemble (truth-free) view of F1 / F2 / E1 / E2:
  src/ensemble_models.py. Members = model forecast over τ 0.1 TU from x_true(t0 − τ) + N(0, 2 I), control = forecast
  of the un-noised state, error = member − own control. Studies: 100 windows (test 0–99, 1 run each) and one fixed
  window (test case 0, 100 runs). Figures tmp/figs/ens_{windows,fixed}_{growth,decorr,summary,dist}.*, data
  tmp/results/models/ensemble_*.npz (jobs 79438 failed dtype, 79439, figure revision rerun).
- 2026-11-23: Pilot: findings.md deleted; ensemble τ = 0.5 TU (50 steps, = v2 first-guess lag), horizon 5 TU kept.
  Rerun job 80332 (config tau 50 verified in the npz); figures tmp/figs/ens_* regenerated.
- 2026-11-23: Pilot: F and E error growth identical -> emulators must be visibly imperfect. Pilot chose: screen
  both knobs (less training data, weaker architecture), pre-registered pick m_h (emulator vs own teacher,
  one-obs-interval, / σ_o) closest to 0.15 (= F1 model error), stable 20 TU, λ₁ > 0; same setting for E1 and E2.
  train_emulator `+n_train_ic`; src/screen_emulators.py. Grid on teacher F1 (jobs 80336-80343): d100, d30, d10,
  w16, w8, w8d1, d30w16, d10w8d1; screen job 80344 (afterok).
- 2026-11-23: Emulator screen done (tmp/results/models/emulator_screen_{f1,f2}.md). Pick by rule: d10w8d1 (10 train
  ICs, MLP 1 x 8). E1 m_h 0.142, E2 (same setting, job 80345) m_h 0.126; both stable, λ₁ > 0. Saved as
  tmp/emulators/{e1,e2}_d10w8d1.pt; registry still points to e1 / e2 (well-trained) pending pilot sign-off.
- 2026-11-23: Pilot approved d10w8d1: models.yaml e1 / e2 -> e1_d10w8d1 / e2_d10w8d1 (e1.pt / e2.pt kept).
  Backgrounds e1 / e2 regenerated, hybrid B re-tuned (e1: α 0.95, s² 10.0 = upper grid edge), ensemble figures
  rerun (job 80349).
