# Progress

## 2026-11-09
- Initialized project per vibeCodingInit.md: created memory/, src/, run/, tmp/, empty .env (chmod 600), .gitignore.
- `git init -b main` — no commits made (awaiting pilot authorization).
- Started Phase A: M-O-R questions sent to pilot.
- Phase A: Motivation captured (Lorenz-63; generalization, adjoint cost, emulator gradient fidelity). Objective/Result pending.
- Phase A revised by pilot: single problem = cost of iterative 4D-Var optimization; meta = learned optimizer
  (not meta-trained emulator). 2x2 design {M1 L63, M2 emulator} x {classical, meta}, strong & weak constraint.
- Phase A: meta = learned update rule; two configs Meta-A (optimized K-step correction operator) vs Meta-B (imitate classical optimizer). Ambiguities sent to pilot.
- Phase A: Meta-A/B ambiguities resolved except g_theta inputs (await pilot reference code).
- Phase A: metrics captured (forecast RMSE > window, cost, phys. consistency/grad fidelity, generalization to unseen IC). Proposed concrete PC/GF metrics in findings.md.
- Phase A CLOSED: core metrics G2,P1,P3; target = paper; next models L96 (forced) / multi-layer QG. Phase B started (North Star Q).
- Phase B Q1 North Star captured (X=5%, N=10x + scientific claim). Clarifying 'missing dynamics' & 'delta inner loss'.
- North Star clarified: missing dynamics = intrinsic + deliberate; gap-closure ratio accepted; 'delta inner loss' awaits reference code. Next: Phase B Q2 Integrations.
- Phase B Q2 Integrations: SLURM GPU, TensorBoard, local git only. Rules: git user-only; publication-ready findings.md per development. Next: Q3 Source of Truth.
- Phase B Q3 Source of Truth: seeded L63 generation, xarray NetCDF/Zarr in tmp/data/, IC split, emulator on same trajectories. Next: Q4 Delivery Payload.
- Phase B Q4 Delivery Payload confirmed. Next: Q5 Behavioral Rules.
- Phase B.1 CLOSED: behavioral rules confirmed. Starting B.2 prior-art research + data schema draft.
- Data schema draft v0 (D1 dataset, D2 emulator, D3 analysis output) written to original.md. Research agent running.
- B.2 research done -> findings.md 'Related work' (paper-ready, [V]/[U] tagged). Pending: reference code, DA definitions, schema sign-off.
- 2026-11-10: Read bice reference (delta inner loss). Mapped to Meta-B; method draft + L63 adaptation in findings.md. D2 emulator proposal v0 in original.md.
- 2026-11-10: D2 RK4 approved; Meta-B protocol (train on train ICs, frozen test); supervised Meta-B defined; bice port authorized.
- 2026-11-10: bice optimizer/Hydra/SLURM surveyed; implementation plan v0 in task_plan.md; DA default proposal sent.
- 2026-11-10: Plan v0 + DA config approved. Milestone 1 started (L63, data gen, gradient tests).
- 2026-11-10: Milestone 1 DONE. src/dameta/{dynamics/l63.py,data/generate.py}, src/gen_data.py, src/configs/gen_data.yaml, src/tests/test_m1.py (11/11 pass). D1 in tmp/data (seed 42). Lyapunov 0.904±0.012. Flag: test ICs within 0.033 of train ICs (interpolation).
- 2026-11-10: Added held-out region split test_ood (flag heldout_region); data regenerated (4 splits); tests 13/13. Memory updated.
- 2026-11-10: Pilot decision: held-out split stays on main; Milestone 2 continues in git worktree branch
  `m2-classical-4dvar` (.claude/worktrees/). Branch notes live in memory/branches/<branch>/ (see original.md
  "Memory protocol"); shared memory files are edited on main only.

### Branch m2-classical-4dvar (merged by pilot)
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
- 2026-11-10: Pilot: report cost gate vs GD and Adagrad; claim anchored on Adagrad (recorded in branch decisions/findings).
- 2026-11-10: Pilot merged m2-classical-4dvar into main; branch memory folded into shared files.
