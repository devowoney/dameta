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
