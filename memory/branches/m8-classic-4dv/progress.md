# Progress — branch m8-classic-4dv

- 2026-11-25: Plan approved (~/.claude/plans/rectangular-model-is-redefined-linked-rain.md). Worktree + branch
  m8-classic-4dv from 3b9a0e1; data / emulators copied from m7.
- 2026-11-25: Code: src/dameta/eval/landscape.py (hessian_batch, spectrum, cos_truth); run_4dvar_v2 `LandscapeRecorder`
  (flags landscape_iter, hess_every) + sol_deg / traj_deg in evaluate_v2; src/tests/test_landscape.py;
  src/scripts/m8_lists.py; src/report_classic_4dv.py (select / report). Smoke job 81642.
- 2026-11-25: Fixed a pre-existing m7 bug in run_4dvar_v2 `registry()` (empty registry_overrides → plain dict passed
  to OmegaConf → every run failed). Smoke job 81648: test_landscape 13/13 PASS (Hessian vs FD rel. err ≤ 1e-9, F/E,
  strong/weak), test_models PASS; 16 tiny runs OK (sol_deg = √3 rmse_x0; κ NaN iff n_neg > 0). Fixed wall_s
  (analysis Hessian no longer subtracted).
- 2026-11-25: val lr sweep, 240 runs (gd / adagrad / adam × 5 lr × 4 models × 2 schemes × 2 obs), jobs
  81685-81696 (12 nodes, 3 runs in parallel each, 36 GB).
- 2026-11-25: Sweep done (240 runs). 8 picks on the lower grid edge (gd lr 1e-3: e1/f1 full; adam 0.03: f1 / e1
  weak or x-only) → extension gd 3e-4, adam 0.01 (src/scripts/lists/m8_grid_extra.json, jobs 81705-81712). After
  extension no pick on an edge; tmp/results/m8/{selected_lr.json, lr_sweep.md}.
- 2026-11-25: Test runs, 64 (4 models × 2 schemes × 2 obs × gd / adagrad / adam / L-BFGS), 200 cases,
  landscape_iter=true, jobs 81713-81724.
- 2026-11-25: Test done (64/64, no errors; 1 diverged case: F2 strong full L-BFGS). Report job 81740:
  tmp/results/m8/classic_4dv.{md,csv}, 48 figures tmp/figs/m8_{cos_truth,lam_min,lam_max,kappa_H,n_neg,eig_a}_<opt>.
- 2026-11-25: Pilot asked for method-comparison figures (all optimizers superposed): src/fig_classic_4dv.py (jobs
  81918-81920) -> tmp/figs/m8cmp_{cost,icerr,fcerr,traj}_<scheme>_<obs>, m8cmp_icerr_a, m8cmp_geom_<model> (strong
  only); trajectory cache tmp/results/m8/traj_cmp.npz. New methods (meta) in tmp/results/m8/test/<model>/ are picked
  up automatically (next colour slot).
- 2026-11-25: Pilot feedback: per-variable errors → m8cmp_{icerr,fcerr}_xyz_<scheme>_<obs>, m8cmp_icerr_a_xyz; window
  marked by band + dashed end line + "window 0.6 TU" label; traj figure shows the noisy observations (job 82065).
- 2026-11-25: Pilot feedback: per-variable figures share one y range per figure; norm-only error figures removed
  (m8cmp_icerr_<c>_<p>, m8cmp_icerr_a, m8cmp_fcerr_<c>_<p>, code + files); new m8cmp_geom_xyz_<model> (J on the
  x-y / x-z / y-z planes through the L-BFGS analysis, strong). Jobs 82156, 82161.
- 2026-11-25: Pilot redefinition of the 4D-Var experiment (see decisions): code `background.time_shift`
  (background.py, gen_data_v2), gen_data_v2.yaml (obs_every 20, σ_o² 4, history 105), run_4dvar_v2 `cost.b_infl` and
  σ_o² from the data, report header from the data, m8_lists `binfl`, report_classic_4dv `binfl`. Data regenerated
  (job 82279), hybrid B retuned with the extended s² grid (job 82293). B-inflation val smoke, 48 L-BFGS runs
  (jobs 82294-82305) -> tmp/results/m8/binfl.md.
- 2026-11-25: binfl smoke done (48/48; job 82297 stuck on 207 -> resubmitted 82909 on 216); table tmp/results/m8/binfl.md. Waiting for the pilot's b_infl choice before the lr sweep / test.
- 2026-11-25: Pilot: b_infl 4, longer windows, 10-case feasibility first. Data tmp/data_w{100,160} (job 82931); feasibility smoke 192 runs (3 windows × 4 models × 2 × 2 × gd 0.01 / adagrad 0.3 / adam 0.3 / L-BFGS; jobs 82932-82943) -> tmp/results/m8/feas.md
- 2026-11-25: Feasibility smoke done (192/192, 0 diverged; job 82936 stuck on 207 -> 83127 on 204). tmp/results/m8/feas.md. Waiting for the pilot's window choice.
- 2026-11-25: Pilot asked for window-impact figures: src/fig_window_4dv.py (feasibility runs, 3 windows) -> tmp/figs/m8win_{profile,geom,nonpd,lammin_a,icerr_xyz,icerr_a_xyz}_* (28 figures, job list src/scripts/lists/m8_figwin.txt).
- 2026-11-26: Pilot fixed the window at 1.6 TU: data moved (tmp/data = 1.6 TU), configs / m8_lists / fig_window_4dv paths updated. Next: val lr sweep + 200-case test on the confirmed configuration (awaiting go).
- 2026-11-26: Pilot: lr sweep on 10 val cases only (m8_lists sweep 12 10; old m8_grid_extra.json -> m8_grid_extra_o10s2.json, fresh grid). 240 runs, jobs 83626-83637 -> tmp/results/m8/sweep, select -> selected_lr.json, lr_sweep.md.
- 2026-11-26: Pilot stopped the 10-case lr sweep (216/240 done, sigma_q 0.1 -> invalid for weak). Per-model diagonal Q implemented (CostConfig.q_var, FourDVarCost.jq, cost.q=model, src/estimate_q.py; job 83664); test_landscape fixed (weak control size from n_obs) 13/13 PASS (job 83669). Nodes 205/208-216 drained, 202/203 draining. Weak vs strong 10-case smoke, 64 runs (m8_lists feasq; jobs 83671 on 206, 83674 on 204) -> tmp/results/m8/feasq.md, src/fig_weak_strong.py -> tmp/figs/m8wq_*.
- 2026-11-26: Pilot: static truth-free B. Cancelled flow-B weak/strong smoke (21/64 -> tmp/results/m8/feasq_flowB). estimate_b.py + load_static_background + background_cov flag (job 83760: B PD, test_landscape 13/13, 1-case checks OK). Weak vs strong smoke rerun, 64 runs, hess_every=5, job 83763 on 204 -> feasq.md, m8wq_* figures.
- 2026-11-26: Static-B weak vs strong smoke done (64/64, 0 diverged; job 83763, 1h56). Post-processing list re-created as m8_post_feasq.txt (the m8_feasq_* cleanup glob had deleted it), job 83966 -> tmp/results/m8/feasq.md, tmp/figs/m8wq_* (10 figures).
- 2026-11-26: Pilot: 1-case lr sweep with the current config (1.6 TU, static B x 4, per-model Q); servers in maintenance, only 204/206/207. Old sweep archived tmp/results/m8/sweep_flowB_sq01. 240 runs: 200 on 204 (job 83996, 12 parallel), 20 on 206 (83997), 20 on 207 (83998) -> select -> selected_lr.json, lr_sweep.md.
- 2026-11-26: 1-case lr sweep done (240/240, jobs 83996-83998); select -> tmp/results/m8/selected_lr.json, lr_sweep.md; 7 picks on a grid edge (gd 0.1 upper: e1/f1/f2 strong x_only, f1 weak x_only; adagrad 3 upper: e2 strong full; adam 0.03 lower: f1 strong/weak x_only).
- 2026-11-26: Pilot accepted the 1-case lr picks as they are (no edge extension); recorded in decisions.md.
- 2026-11-26: M8-R plan approved (~/.claude/plans/now-we-have-we-tender-corbato.md). Code: gen_data_v2 `truth_model`, estimate_b `--models` (merge), m8_lists modes qtune / rect_smoke / rect (STAGES, Q_GRID), src/select_q.py, src/report_rect_4dv.py, src/fig_rect_4dv.py. Data job 84049: bg_l1 in tmp/data, tmp/data_tF1, tmp/data_tF2 (truth F1 / F2, bg f1 / f2), static B l1 / f1 / f2 (PD, chi2 test 3.3 / 3.7 / 3.4); test_landscape 13/13. Q tuning 126 runs, jobs 84050-84052 (only 204 / 206 / 207 up).
- 2026-11-26: Q tuning done (126 runs; job 84050 stuck on 204 for memory -> cancelled, lines rerun as 84144 / 84145 on 206 / 207). Shared x_b = x_true(t0 − 5) checked (max diff 0). Mean pick: F1 1.78, F2 100 (EDGE), E1 1.78, E2 17.8 (truth L1); E1 0.32 (truth F1), E2 0.18 (truth F2). Pilot: extend + median -> 16 extra runs job 84164, select_q job 84165.
- 2026-11-26: Q re-selected (median + plateau rule; jobs 84164 extension, select_q). M8-R smoke (120 runs, 10 val cases, landscape_iter) jobs 84190 (206, 80 lines) / 84191 (204, 40 lines); 207 full.
- 2026-11-27: M8-R smoke done (120/120, no errors; 1 diverged case: F1 weak x_only GD, truth L1). report_rect_4dv / fig_rect_4dv smoke OK -> tmp/results/m8r/rect_smoke.{md,csv}, rect_smoke_delta.csv, tmp/figs/m8r_{land,geom}_*. Waiting for the pilot's go for the 200-case test.
- 2026-11-27: Emulator-vs-teacher check over many ICs: src/fig_emulator_ics.py (job 84362) -> tmp/figs/m8emu_{traj_<e>,err,map}.png. Bad-emulator screen (1, 3 or 10 train ICs, teacher F1 / F2, M3 recipe 10k steps): d3w4 (2x4), d3w4d1, d10w4d1, d3w8d1, d3w6d1, d1w6d1 (jobs 84366-84402) -> tmp/emulators/e{1,2}_<name>.pt, tmp/figs/m8emu_*_<name>.png.
- 2026-11-27: Pilot picked d1w6d1 as bad emulators: models.yaml e1b / e2b (aliases e1g / e2g -> e1 / e2). M8-R STAGES + e1b / e2b, LR_MODEL (e1b -> e1, e2b -> e2), m8_lists `--only`. Q tuning 92 runs (jobs 84403 / 84405 / 84407), select_q (84441; previous picks unchanged, backups q_chi2_seed42_pre_bad.json): e1b 0.316 / e2b 3.16 (truth L1), e1b 0.562 (F1), e2b 0.1 (F2).
- 2026-11-28: M8-R smoke for e1b / e2b, 64 runs (jobs 84443-84445, 0 diverged); report + figures (84446) -> tmp/results/m8r/rect_smoke.{md,csv} (184 runs), tmp/figs/m8r_*. README table: E1b / E2b rows.
- 2026-11-28: Pilot asked for quantitative emulator quality: src/model_quality.py (m = one-obs-interval error, w = window RMS error from true x0, both / sigma_o; vs teacher and vs L1; current M8 data 1.6 TU, obs every 20, sigma_o 2), test 200 cases, job 84572 -> tmp/results/models/quality_test.{json,md}.
- 2026-11-28: Pilot redefined bad emulator target: m ~ 0.45 or w ~ 3 vs teacher (~3x good). Screen of 16 new candidates (1 train IC, depth 1; width 2-5 @10k steps, width 6 @1k/2k/3k/5k steps; jobs 84581-84584) + model_quality --tag badscreen (84585) -> tmp/results/models/quality_test_badscreen.{json,md}. Awaiting pilot pick (models.yaml e1b / e2b unchanged).
- 2026-11-28: Diagnostic figures for d1w6d1s5000 (both teachers, with good E1 / E2 alongside): src/fig_emulator_ics.py (job 84590) -> tmp/figs/m8emu_{err,map,traj_e1,traj_e1b,traj_e2,traj_e2b}_d1w6d1s5000.png (renamed: --tag needs the leading underscore).
- 2026-11-28: Pilot adopted d1w6d1s5000 as bad emulators: models.yaml e1b / e2b -> e{1,2}_d1w6d1s5000 (Q tuning / M8-R smoke for e1b / e2b were done with d1w6d1: stale). model_quality.py: false models use L1 as own reference. Full rectangle table (L1, F1, F2, E1, E2, E1b, E2b; job 84592) -> tmp/results/models/quality_test_rect.{json,md}, rectangle_table.md.
- 2026-11-28: Re-tune Q + rerun M8-R smoke for e1b / e2b (now d1w6d1s5000). Old e1b / e2b results archived: tmp/results/m8r_d1w6d1/ (qtune, smoke, qtune.md, rect_smoke.*), <data>/q_chi2_seed42_d1w6d1.json, tmp/figs/m8r_d1w6d1/. Q tuning 92 runs (jobs 84594-84597; 84598 / 84599 stuck pending on 204 / 207 -> resubmitted 84607 on 202, 84608 on 216), select_q (84600, dependency moved).
- 2026-11-28: select_q (84600): e1b / e2b picks truth L1 0.316 / 3.16 (unchanged), e1b truth F1 1.0 (was 0.562), e2b truth F2 0.562 (was 0.1); no edge, 0 diverged. M8-R smoke e1b / e2b 64 runs: jobs 84618 (216), 84620 (202), 84623 (214), 84624 (215) (sb_submit crashed on 202's 192G request; parts 2 / 3 written by hand and checked = same 64 lines); report + figures job 84625.
- 2026-11-28: M8-R smoke e1b / e2b (d1w6d1s5000) done: 64/64, 0 diverged; report + figures (84625) -> tmp/results/m8r/rect_smoke.{md,csv}, rect_smoke_delta.csv (184 runs), tmp/figs/m8r_*. README E1b / E2b rows not yet updated.
- 2026-11-28: README: E1b / E2b description (d1w6d1s5000) and the 4 smoke rows (L-BFGS, full obs) updated from rect_smoke.md.
