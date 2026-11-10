# Branch m2-classical-4dvar — progress (branch-local; fold into memory/progress.md on main after merge)

- 2026-11-10: Worktree created from main 4cbb6d2. Restored src/dameta/da/ (cost.py, reference_step.py, fourdvar.py,
  __init__.py), written before isolation. D1 regenerated in the worktree's tmp/data (deterministic, seed 42, 4 splits).
- 2026-11-10: Wrote da/cost.py, da/reference_step.py (bice port), da/fourdvar.py, run_4dvar.py + configs/fourdvar.yaml,
  select_lr.py, tests/test_m2.py (8/8 pass, 39 s). First full val run (strong/full/Adagrad lr 0.1, 200 cases, 132 s):
  0 diverged, median J_final 16.51 (chi2 expectation m/2 = 16.5), rmse_x0 0.458 vs background 0.868.
  lr sweeps (50 val cases, strong + weak, 64 runs) launched in background, logs tmp/logs/sweep_{strong,weak}.log.
- 2026-11-10: lr sweeps done (64 + 4 runs, rule applied, all interior). Final E1 runs: 8 configs x {val, test, test_ood}
  x 200 cases (24 runs), 0 diverged. report_e1.py -> tmp/results/e1/summary.{csv,md}, evals_to_target.csv,
  tmp/figs/m2_e1_{forecast_rmse,convergence}.{png,pdf}. Tests: M1 13/13, M2 8/8. Milestone 2 DONE (pending pilot
  review). Open: which baseline anchors the 10x cost gate (GD vs Adagrad).
