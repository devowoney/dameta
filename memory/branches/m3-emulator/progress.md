# Branch m3-emulator — progress (branch-local; fold into memory/progress.md on main after merge)

- 2026-11-10: Worktree created from main 31fb43c. D1 regenerated in worktree tmp/data; E1 results copied from the
  m2 worktree (tmp/results/e1) as the reference for E3 / gap-closure.
- 2026-11-10: Wrote dynamics/emulator.py (L63Emulator: SiLU MLP tendency + RK4; TeacherConfig/teacher_rollout for
  deliberate degradation), dynamics/train_emulator.py (rollout loss + curriculum), train_emulator.py + configs/emulator.yaml,
  eval/emulator_metrics.py (free forecast, Lyapunov, TL fidelity, G1), eval_emulator.py, run_4dvar.py dynamics=emulator
  (+ J_true_final = P1, rmse_fc_m1). tests/test_m3.py 5/5 pass. Clean emulator training launched (10k steps).
- 2026-11-10: Clean emulator trained (447 s, best val rmse@0.5TU 0.0286); diagnostics + G1 done. Degradation candidates measured (A/B/B'/C). E3-clean lr sweeps running (50 val cases).
- 2026-11-10: E3-clean lr sweep done (24 runs; same lr as E1). Final E3-clean runs on test/test_ood launched (16 runs).
- 2026-11-10: E3-clean final (16 runs, 0 diverged) + report_e3.py. E3-clean ~ E1 (gap tiny) -> degraded emulator required for gap-closure claim. Awaiting pilot degradation choice.
- 2026-11-10: Pilot chose B' (xz 0.95) main, B (xz 0.9) stress. Training emulators xz095, xz090.
- 2026-11-10: Emulators trained: xz095 (best val 0.0304 vs its teacher, 707 s), xz090 (0.0337, 702 s). lr sweeps running.
- 2026-11-10: Degraded emulators xz095/xz090 diagnostics + E3 final (16 + 8 runs, 0 diverged). report_e3 fixed to pair with E1 by configuration (lr may differ). Large gaps (2.09 / 3.77 @1TU). Found Q mis-specification for degraded emulators -> pilot decision.
- 2026-11-11: Pilot committed m3-emulator; decided to keep Q fixed (sigma_q = 0.1). Milestone 3 closed.
- 2026-11-11: Consolidated IC-estimation report: src/report_ic.py -> tmp/report (3 tables with bootstrap CIs + paired Wilcoxon, 3 figures, CVD-validated palette). Paper-ready section appended to branch findings.
