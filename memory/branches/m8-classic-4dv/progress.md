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
