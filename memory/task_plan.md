# Task Plan

## Done
- [x] Held-out region split test_ood (2026-11-10)
- [x] Scaffold; Phase A (M-O-R); Phase B.1 (5 questions); B.2 prior art; bice reference read + mapped

## Phase C implementation plan v0 (2026-11-10) — APPROVED by pilot 2026-11-10
Layout (everything code under src/; artifacts to tmp/):
  src/dameta/dynamics/l63.py        M1: L63 + RK4, float64; TL/adjoint via autodiff
  src/dameta/dynamics/emulator.py   M2: MLP tendency + RK4; degradation flag
  src/dameta/data/generate.py       D1: seeded truth/obs/background -> xarray NetCDF
  src/dameta/da/cost.py             J = J_b + J_o (strong); + J_q with eta (weak)
  src/dameta/da/reference_step.py   port bice ReferenceStep (GD / Adagrad teacher)
  src/dameta/da/fourdvar.py         classical outer loop (E1/E3), eval-count + wall-clock instrumented
  src/dameta/meta/networks.py       MLP S_theta + iteration embedding
  src/dameta/meta/meta_b.py         port bice delta chain (l_align / l_perf), supervised variant, frozen inference
  src/dameta/meta/meta_a.py         free update rule unrolled K steps with dynamics calls, loss J(x0^K)
  src/dameta/eval/metrics.py        forecast RMSE / VPT, cost, G2, P1, P3 (backward Lyapunov vectors), gap-closure
  src/run.py + src/configs/         Hydra entry (project_root resolver, seeds, SLURM env detect)
  src/scripts/sbatch_*.sh           Odyssey_GPU, mamba activate oceanai — submit only with pilot OK
  src/tests/                        adjoint dot-product, FD vs AD (M2), s=0 chain == one teacher step, reproducibility
Milestones:
  M1 L63 + data gen + gradient tests [DONE]    M4 Meta-B (port): E2-B, E4-B
  M2 classical strong/weak 4D-Var: E1 [branch m2-classical-4dvar]          M5 Meta-A: E2-A, E4-A
  M3 emulator train + E3 + G1                  M6 metrics, figures, SLURM sweeps, findings.md entry per milestone
