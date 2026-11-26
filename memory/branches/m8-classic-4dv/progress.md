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
