# Branch m5-meta-grid — findings (paper-ready; fold into memory/findings.md after merge)

## M5 step 4 — classical baselines on D1 v2: model-error-isolated BICE / BPE and adjoint-free gradients (2026-11-14)

### Methods (paper-ready draft)
Truth = Lorenz-63 (σ 10, ρ 28, β 8/3); observations of the truth every 0.1 TU with N(0, 2 I) noise over a 1.1-TU window
(the Lyapunov time of the false model; 12 observation times). All estimation uses a FALSE model: F1, a knowledge
numerical model (L63 with ρ = 27.5), or F2, an MLP emulator trained on F1 trajectories. The first guess is one member
of a time-lagged ensemble of the false model (forecast launched from the true state 0.5 TU before t0, valid at t0);
the background covariance is hybrid, B = 0.25 B_ens + 0.75 s² I, with B_ens the covariance of 10 lagged members
(0.1–1.0 TU) and (α, s²) fitted by maximum likelihood of the validation innovations at t0 (truth-free; χ² 3.2 on
val, 3.4 on test). BICE estimates the initial condition (strong / weak constraint, σ_q 0.1); BPE jointly estimates
[x0, σ, ρ, β] with F1's structure and a Gaussian parameter prior centred on F1's values (std 10 %). E1 (true model
inside 4D-Var, same first guess and B as F1) is the perfect-model reference. Learning rates by the pre-registered
validation rule. Adjoint-free baselines: central finite-difference gradients of the F1 cost (2d + 1 forward runs per
gradient) and a hybrid using J from F1 and the gradient from the emulator's adjoint (1 F1 forward run + 1 emulator
gradient). Source: src/run_4dvar_v2.py; tables tmp/results/v2_summary.{csv,md} (src/report_v2.py).

### Results (test, 200 ICs, Adagrad, medians; fc = forecast RMSE at 1 TU after the window, vs truth)
| estimation | problem / constraint | evals to 1 % | IC RMSE (first guess 0.645) | fc with estimation model | fc with true model | paired gap vs E1 | P1 J_true |
|---|---|---|---|---|---|---|---|
| E1 truth (reference) | BICE strong | 8.5 | 0.327 | 0.71 | 0.71 | — | 18.0 |
| F1 ρ 27.5 | BICE strong | 10 | 0.425 | 1.55 | 2.47 | +1.02 | 23.8 |
| F2 emulator | BICE strong | 10 | 0.419 | 1.48 | 2.50 | +1.05 | 23.7 |
| F1 | BICE weak | 49 | 0.432 | 1.35 | 2.53 | +1.21 | 23.1 |
| F1 | **BPE** [x0, σ, ρ, β] | 95.5 | 0.413 | 1.39 | **1.73** | **+0.45** | 21.2 |
| F1 FD gradient (7 fwd / grad) | BICE strong | 10 | 0.425 (identical, Δx0 4e-10) | 1.55 | 2.47 | +1.02 | 23.8 |
| F1 hybrid gradient (1 fwd + emulator adjoint) | BICE strong | 6 (reached 0.88) | 0.416 (Δx0 0.04) | 1.45 | 2.46 | +1.02 | 23.6 |
BPE estimates (test, median [IQR]): ρ 27.81 [27.53, 28.06] (truth 28, prior 27.5), σ 10.24, β 2.672.
test_ood (strong/full): first guess already 0.267; F1 / F2 BICE analyses DEGRADE it to 0.42 / 0.43 (E1 0.187), BPE
keeps it at 0.195 (ρ 27.91) with gap vs E1 0.10. GD needs 90–660 evaluations (Adagrad 5–96).

### Findings
- **Model error dominates the inverse problem.** With the correct model 4D-Var halves the first-guess error
  (0.645 → 0.327); with either false model it only reaches 0.42 and the analysis is "false-model consistent":
  forecasting it with the true model is worse (2.47) than with the false model (1.55). Out of distribution, where the
  lagged first guess is already good (0.27), false-model 4D-Var makes the initial condition worse (0.42).
- **F2 ≈ F1:** the emulator reproduces the knowledge model inside 4D-Var (IC 0.419 vs 0.425, gap 1.05 vs 1.02, same
  cost) — a good emulator is a faithful stand-in for the numerical model.
- **BPE recovers part of the missing physics:** jointly estimating the parameters moves ρ from 27.5 towards 28
  (median 27.81; 27.91 test_ood) and halves the true-model forecast gap (+1.02 → +0.45; test_ood +0.57 → +0.10), at
  ~10× more gradient evaluations.
- **Adjoint-free baselines:** finite differences reproduce the adjoint analysis exactly (Δx0 ~ 1e-10) at 7 (strong) /
  73 (weak) forward runs per gradient — exact but scaling with the control size. The hybrid (F1 cost + emulator
  adjoint) loses nothing in estimation quality (IC 0.416, fc 2.46 vs 2.47) with one F1 forward run per gradient, but
  its gradient is slightly inconsistent with J, so fewer cases reach the 1 % cost target (0.82–0.92). This is the
  reference the learned adjoint-free gradient (cell 8) must match.
