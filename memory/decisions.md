# Decisions

| Date | Decision | Rationale |
|---|---|---|
| 2026-11-09 | Use spec file layout; memory file named `original.md` (spec text; tree diagram says `origin.md`) | Spec body names `original.md` explicitly |
| 2026-11-09 | `tmp/`, `.env`, `.venv/` git-ignored | tmp is disposable; secrets never versioned |
| 2026-11-09 | No CLAUDE.md created | Explicit spec rule |
| 2026-11-09 | First testbed = Lorenz-63 | Pilot choice; cheap, exact adjoint available as ground truth for gradient fidelity |
| 2026-11-09 | Meta module = learned iterative optimizer for 4D-Var, NOT meta-trained emulator | Pilot clarification; problem is cost of iterative minimization |
| 2026-11-09 | 2x2 design: {M1 physical L63, M2 AD emulator} x {classical 4D-Var, 4D-Var+meta}; strong & weak | Pilot design |
| 2026-11-09 | Two meta configs: Meta-A outcome-optimized K-step operator; Meta-B imitation of classical optimizer | Pilot; compare outcome-driven vs behaviour-driven learning of the optimizer |
| 2026-11-09 | Meta-A loss=J(x0^K); Meta-B teacher=GD (Adagrad candidate); one meta learner per dynamics model | Pilot answers |
| 2026-11-09 | Core consistency/fidelity metrics = G2, P1, P3 | Pilot choice |
| 2026-11-09 | North Star gate X=5% forecast RMSE tolerance, N=10x cost reduction | Pilot |
| 2026-11-09 | Emulator error = intrinsic + deliberate degradation; metric = gap-closure ratio | Pilot; controlled, reproducible model error |
| 2026-11-09 | Git commit/push reserved to user; agent touches git only on explicit request | Pilot rule |
| 2026-11-09 | Each development -> publication-ready markdown update (findings.md) | Paper target |
| 2026-11-09 | Seeded generated data; emulator trained on same trajectories; split by IC | Pilot; reproducibility |
| 2026-11-09 | Test ICs excluded from emulator and meta training | Avoid leakage in generalization claim |
| 2026-11-09 | Behavioral rules 1-7 adopted (SLURM approval, gradient checks, reproducibility, integrity, stack, tone, no-unasked actions) | Pilot confirmed |
| 2026-11-10 | Meta-B = bice Δ inner-loss mechanism; Meta-A = free K-step unrolled rule with dynamics calls, loss J(x0^K) | Resolves A/B distinction from reference code |
| 2026-11-10 | Port bice optimization/Hydra/SLURM/delta-chain into src/, adapted to L63 MLP | Pilot authorized; reuse prior work |
| 2026-11-10 | Plan v0 + DA config v0 approved | Pilot sign-off |
| 2026-11-10 | Harder held-out test = lobe-transition region r_xy<3 (test_ood), in-dist r_xy>=4; x-only obs as secondary OOD | Pilot asked for harder test; test ICs were within 0.033 of train ICs |
| 2026-11-10 | Held-out region split kept on main; M2 onward developed in git worktrees (branch per milestone) | Pilot; isolate milestone work, user merges |
| 2026-11-10 | Branch memory in memory/branches/<branch>/; shared memory files edited on main only | Avoid merge conflicts on append-only memory files |
| 2026-11-10 | Weak-constraint control = (x0, eta_1..eta_10), eta added to the state at obs times t_i, i >= 1 | Approved DA config v0 |
| 2026-11-10 | Cost unit = 1 gradient evaluation (forward + adjoint run per case); per-case stopping on relative dJ < tol | Fair per-case iteration count in a batched run |
| 2026-11-10 | lr selection rule (fixed before results): min median final J on val; exclude lr with >1% diverged; tie-break median n_grad_evals | Truth-free like 4D-Var; pre-registered to avoid tuning bias |
| 2026-11-10 | lr sweep on first 50 val cases (CPU, 4 cores shared); chosen lr re-run on full val/test/test_ood | Login node limits; SLURM needs pilot OK |
| 2026-11-10 | Final lr: Adagrad 0.1 (all); GD 0.001 (strong), 3e-4 (weak) | Pre-registered rule on val sweep |
| 2026-11-10 | Cost metric = gradient evaluations to reach J <= (1+tol) J*, tol 1 % and 5 % (J* = best final J per case) | Curves plateau before the 1e-6 stopping rule |
| 2026-11-10 | North Star cost gate: report vs BOTH GD and Adagrad; the claim (<= 1/10 evals) is anchored on Adagrad, the strongest classical baseline (target ~7 evals to 1 % of J*, strong/full test) | Pilot decision; reviewers expect the strongest baseline. FOLD INTO original.md North Star on main after merge |
| 2026-11-10 | Deliberate degradation implemented in the emulator's TRAINING TEACHER (biased sigma/rho/beta, xz_scale on the -x*z term of dy/dt), trajectories from the same D1 train ICs; network and D1 untouched | Controlled, reproducible model error; keeps the pilot rule "same trajectories" (same ICs) |
| 2026-11-10 | E3 analyses judged with the true model: J_true_final (P1) and rmse_fc_m1 (forecast with M1 from the analysis); rmse_fc = forecast with the emulator | Separate IC quality from emulator forecast error; P1 defined in North Star |
| 2026-11-10 | Emulator training: rollout loss, curriculum n_roll 1 -> 10, Adam + cosine decay, best-val checkpoint (val = D1 val ICs) | Standard for stable multi-step emulators; test splits never used |
| 2026-11-10 | E3-clean lr = E1 lr (Adagrad 0.1; GD 1e-3 strong / 3e-4 weak), re-selected by the same rule on a 3-point grid (all interior) | Pre-registered rule; emulator does not shift the optimum step |
| 2026-11-10 | Degradation: main = B' (teacher xz_scale = 0.95, emulator 'xz095'); stress test = B (xz_scale = 0.9, emulator 'xz090'); clean emulator = control | Pilot decision; B' model error ~ obs error, B exceeds it |
| 2026-11-10 | Degraded-emulator lr check on full obs only (Adagrad {0.03,0.1,0.3}; GD strong {3e-4,1e-3,3e-3}, weak {1e-4,3e-4,1e-3}); x-only reuses the full-obs choice | x-only lr matched full-obs lr exactly in E1 and E3-clean; CPU budget |
| 2026-11-10 | Degraded-emulator lr: Adagrad 0.3 strong / 0.1 weak; GD 3e-4 (strong grid edges extended to 1.0 and 1e-4 to confirm interior) | Pre-registered rule |
| 2026-11-11 | Keep Q fixed (sigma_q = 0.1, approved DA config v0) for all emulators; no per-emulator Q re-estimation | Pilot decision; mis-specified Q is part of the degraded-emulator test |
| 2026-11-11 | Report statistics: median + 95 % bootstrap CI (2000, seed 0); E3 vs E1 paired by case (Wilcoxon); cost = evals to 1.01 J*, J* = best final J over GD/Adagrad with the same dynamics | One pre-fixed definition for all tables; supersedes the 'own J*' cost used earlier for E3 |
