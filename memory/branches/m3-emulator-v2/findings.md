# Branch m3-emulator v2 — findings (paper-ready; fold into memory/findings.md after merge)

## Parameter-biased emulator and emulator quality checks (2026-11-13)
### Methods (paper-ready draft)
The deliberately imperfect emulator is now trained on trajectories of a Lorenz-63 model with a biased parameter
(ρ = 27.5 instead of 28; alternative σ = 10.5 instead of 10), started from the same training initial conditions, with
the unchanged M3 recipe (MLP tendency + RK4, multi-step rollout loss, 1 → 10-step curriculum, 10 000 Adam steps, best
validation-rollout checkpoint). Candidate biases were screened against the true model on validation ICs by the
model error accumulated over one observation interval (the error 4D-Var sees between observations). Emulator quality
is assessed on held-out test ICs with three checks against the true model M1 and against the emulator's own biased
teacher: (1) free rollouts from the same true initial state; (2) separation of two initial states 10⁻³ apart
propagated by the same model; (3) 100-member ensembles — (a) members taken at different times of a long true
trajectory and propagated 20 TU (attractor statistics, 1-D Wasserstein distances of the pooled marginals), (b) members
perturbed by N(0, 0.1²) around 20 test states and propagated 10 TU (spread growth).

### Teacher screen (val ICs, teacher vs M1)
| teacher | one-obs-interval model error std (x, y, z) | RMSE from true x0 @0.5 / 1 TU | λ₁ |
|---|---|---|---|
| exact L63 (clean) | 0 | 0 / 0 | 0.893 |
| xz_scale 0.95 (old main) | 0.36, 0.84, 0.47 | 0.84 / 2.37 | 0.898 |
| **ρ 27.5 (new main candidate)** | 0.12, 0.28, 0.15 | 0.33 / 0.85 | 0.870 |
| ρ 27 | 0.24, 0.57, 0.29 | 0.63 / 1.67 | 0.872 |
| ρ 28.5 | 0.12, 0.29, 0.15 | 0.33 / 0.86 | 0.914 |
| **σ 10.5 (milder alternative)** | 0.12, 0.05, 0.06 | 0.12 / 0.30 | 0.906 |
| σ 11 | 0.24, 0.10, 0.12 | 0.22 / 0.60 | 0.886 |
| β 2.5 / 2.8 | 0.04-0.14 | 0.71 / 1.38, 0.55 / 1.15 | 0.871 / 0.904 |
(observation error σ_o = 1.41; tmp/results/emulator/teacher_screen.md)

### Quality checks (200 test ICs; medians; tmp/results/emulator/emulator_qc.md)
| check | metric | ρ 27.5 | σ 10.5 | clean | xz 0.95 |
|---|---|---|---|---|---|
| 1 same IC | RMSE emulator vs M1 @0.1 / 0.5 / 1 / 2 TU | 0.23 / 0.37 / 0.77 / 1.94 | 0.08 / 0.11 / 0.33 / 0.50 | 0.007 / 0.02 / 0.05 / 0.15 | 0.54 / 0.88 / 2.12 / 3.62 |
| 1 same IC | of which intended (teacher vs M1 @1 TU) / learning (emulator vs teacher @1 TU) | 0.75 / 0.05 | 0.33 / 0.06 | 0 / 0.05 | 2.15 / 0.07 |
| 1 same IC | valid prediction time vs M1 [TU] | 1.60 | 3.09 | 4.59 | 1.13 |
| 2 pair | separation growth rate M1 / emulator [1/TU] | 0.92 / 0.86 | 0.92 / 0.90 | 0.92 / 0.91 | 0.92 / 0.89 |
| 2 pair | time to separation 1, M1 / emulator [TU] | 6.24 / 6.37 | 6.24 / 6.25 | 6.24 / 6.22 | 6.24 / 6.30 |
| 3a ensemble | stable over 20 TU | yes | yes | yes | yes |
| 3a ensemble | Wasserstein x, y, z vs M1 | 0.60, 0.59, 0.64 | 0.19, 0.18, 0.25 | 0.71, 0.73, 0.25 | 0.26, 0.28, 1.12 |
| 3b ensemble | spread @1 / 2 / 4 / 10 TU, M1 = 0.23 / 0.75 / 4.80 / 8.42 | 0.21 / 0.71 / 5.68 / 8.83 | 0.23 / 0.81 / 5.28 / 7.53 | 0.23 / 0.72 / 4.50 / 8.28 | 0.23 / 0.71 / 5.09 / 9.00 |

- Both new emulators learn their teacher as well as the clean one learns L63 (emulator-vs-teacher RMSE at 1 TU
  0.05-0.06): their error relative to the true model is the intended parameter bias, ~1/3 (ρ 27.5) and ~1/7
  (σ 10.5) of the old xz 0.95 error at 1 TU.
- All emulators reproduce the sensitivity to initial conditions of L63 (separation growth 0.86-0.91 vs 0.92 /TU;
  same time to O(1) separation), are stable over 20 TU, and reproduce the ensemble-spread growth and saturation.
- Attractor statistics: with 100 members × 10 TU the Wasserstein noise floor is ~0.7 (clean emulator); ρ 27.5 and
  σ 10.5 are at or below it; xz 0.95 shows a clear shift in z (1.12; z std 9.11 vs 8.54).
