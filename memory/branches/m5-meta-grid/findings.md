# Branch m5-meta-grid — findings (paper-ready; fold into memory/findings.md after merge)

# Milestone 5 — Learned optimizers for 4D-Var under model error: a model-error-isolated benchmark (2026-11-15)

## Summary
- With an imperfect model (knowledge model F1, ρ 27.5 instead of 28, or its emulator F2), 4D-Var stalls at an IC error
  of 0.42 versus 0.33 with the true model; the analysis fits the false model and forecasts worse with the true one.
- Joint parameter estimation (BPE) recovers ρ ≈ 27.8 and halves the true-model forecast gap, at ~10× the cost.
- Adjoint-free classical 4D-Var (finite differences, or the emulator's adjoint with the numerical model's cost) loses
  **no estimation quality**, at about half the speed of the exact-adjoint reference on L63.
- **No meta learner beats the reference** (classical Adagrad with the exact adjoint), which already converges in ~10
  gradients on this problem. Best speed-ups: ×1.13 (cell 1, Δ learned optimizer, F1) and ×1.30 / ×1.25 (cell 5, K-20
  learned optimization) — both reach the cost target in fewer cases (0.84, 0.62 vs 0.99).
- Cell 5 regularises against model error (true-model forecast −4 % / −8 % vs the reference); cell 7 improves the
  first guess with **no model run** (IC 0.645 → 0.52); learners without gradient input (cells 3, 4) and the
  adjoint-free learned gradient (cell 8, as built) fail.

## 1. Methods (paper-ready draft)
**Truth and observations.** The truth is Lorenz-63 (σ 10, ρ 28, β 8/3; RK4, Δt 0.01). Observations of the truth,
y = x + N(0, 2 I), are available every 0.1 TU over an assimilation window of 1.1 TU — the Lyapunov time of the false
model (λ₁ = 0.90) — i.e. 12 observation times; forecasts are verified over 5 TU beyond the window. Initial conditions
are split by IC (train 1000 / val 200 / test 200 / test_ood 200, the latter in the held-out lobe-transition region).

**False models.** All estimation uses a model different from the truth: F1, a knowledge numerical model (L63 with
ρ = 27.5), or F2, an MLP emulator (tendency network + RK4) trained on F1 trajectories (rollout loss; quality-checked
against the truth for same-IC rollouts, perturbed-pair separation and 100-member ensemble statistics). The true
model appears only in the evaluation and in a perfect-model reference (E1).

**First guess and B.** The first guess is one member of a time-lagged ensemble of the false model: the forecast
launched from the true state 0.5 TU before t₀ and valid at t₀ — a balanced false-model state, never the truth. The
background covariance is hybrid, B = α B_ens + (1 − α) s² I, with B_ens the covariance of 10 lagged members (lags
0.1–1.0 TU) and (α, s²) = (0.25, 0.40) for F1 / (0.25, 0.45) for F2, fitted by maximum likelihood of the validation
innovations at t₀ (truth-free; χ² of the first-guess error 3.2 on val and 3.4 on test, calibrated value 3).

**Inverse problems.** BICE estimates the initial condition (strong constraint; weak constraint with model-error
increments, σ_q = 0.1). BPE estimates [x₀, σ, ρ, β] jointly with F1's structure and a Gaussian parameter prior
centred on F1's values (10 % std).

**Optimisers.** Classical: GD and Adagrad, learning rate by a pre-registered validation rule (lowest median final
cost; ties → fewer evaluations; grid extended at the edges). Adjoint-free classical baselines: central finite-
difference gradients of F1's cost (2d + 1 forward runs per gradient), and a hybrid gradient (cost from F1, gradient
from the emulator's adjoint). Meta-learners — inputs always the running state x^k, the iteration k and, where stated,
the running gradient ∇J(x^k) (never the background):

| cell | type | training | inputs | inference cost per iteration |
|---|---|---|---|---|
| 1 | learned optimizer (per-iteration rule) | unsupervised: Δ inner loss on J | x, ∇J, k | 1 gradient |
| 2 | learned optimizer | supervised: 5-step teacher lookahead | x, ∇J, k | 1 gradient |
| 3 | learned optimizer | unsupervised: one-step J decrease | x, k (± observations) | 1 forward run |
| 4 | learned optimizer | supervised: teacher's gradient | x, k (± observations) | 1 forward run |
| 5 | learned optimization (K = 20 unrolled steps) | unsupervised: J(x^K) | x, ∇J, k | 1 gradient (20 total) |
| 7 | learned optimization (K = 20) | unsupervised: J(x^K) | x, k (± observations) | none |
| 8 | adjoint-free learned gradient (+ Adagrad) | supervised: emulator ∇J_o | x, innovations, k | 1 forward run |
Cells 1–2 rescale the Adagrad teacher step (bounded stretch, K = 20 network applications per gradient); cells 3, 5, 7
use a free update (bounded MLP output × climatological scale); cell 8 predicts only the observation-term gradient
(∇J_b, ∇J_q are analytic) from the innovations of one forward run. Each learner is trained on the training ICs of its
own false model (3 seeds, best epoch on validation) and applied frozen to test and test_ood. Hyper-parameters: cell 1
uses the configuration selected in Milestone 4 (D1 v1); the others use defaults.

**Metrics and cost.** IC RMSE vs truth; forecast RMSE at 1 TU after the window with the true model (initial-condition
quality) and with the estimation model (operational); cost = model runs until J ≤ 1.01 J* (J* = best final cost of
classical GD / Adagrad on the same model and case), in forward-run equivalents of that model, using the measured cost
of one gradient evaluation (forward + adjoint): 3.7 forward runs for F1, 2.5 for F2 (200 cases batched, one core);
speed-up = reference cost / method cost; "worse basins" = cases ending with a final cost > 1.1 × the reference's.
Sources: src/run_4dvar_v2.py, src/run_meta_b.py (dataset=v2), src/report_v2.py, src/report_m5.py,
src/measure_grad_cost.py; records tmp/results/v2_*, tmp/results/m5_*; tables tmp/results/{v2,m5}_summary*.

## 2. Results

### Table 1 — Effect of model error, parameter estimation and adjoint-free gradients (test, strong / full obs)
Reference: classical Adagrad with the exact adjoint, BICE, on F1. Cost in wall-clock per 200-case batch (measured per
gradient: true model 99 ms, F1 102 ms, F2 283 ms; per F1 forward run 28 ms), because the rows use different models.
| method | gradients to 1 % | wall-clock | speed-up vs ref | IC RMSE (first guess 0.645) | true-model fc @1 TU | gap vs E1 |
|---|---|---|---|---|---|---|
| **reference: F1, BICE, Adagrad, exact adjoint** | 10 | 1.02 s | ×1 | 0.425 | 2.47 | +1.02 |
| E1: true model inside 4D-Var (upper bound) | 8.5 | 0.84 s | ×1.20 | 0.327 | 0.71 | — |
| F2 emulator, BICE, Adagrad | 10 | 2.83 s | ×0.36 (emulator slower than analytic L63) | 0.419 | 2.50 | +1.05 |
| F1, BICE, weak constraint | 49 | 4.98 s | ×0.20 | 0.432 | 2.53 | +1.21 |
| F1, BPE [x₀, σ, ρ, β] | 95.5 | 9.70 s | ×0.10 | 0.413 | **1.73** (−30 %) | **+0.45** |
| F1, finite-difference gradient (no adjoint, 7 fwd) | 10 | 1.95 s | ×0.52 | 0.425 (identical) | 2.47 | +1.02 |
| F1 cost + emulator adjoint (hybrid, no F1 adjoint) | 6 (reached 0.88) | 1.86 s | ×0.55 | 0.416 | 2.46 | +1.02 |
| F1, BICE, GD (exact adjoint) | 122 | 12.3 s | ×0.08 | 0.434 | 2.45 | +1.07 |
BPE estimates: ρ 27.81 [IQR 27.53–28.06] (truth 28, prior 27.5), σ 10.24, β 2.672. test_ood (lagged first guess
already 0.267): F1 / F2 BICE degrade it to 0.421 / 0.428 (E1 0.187); BPE keeps 0.195 (ρ 27.91, gap vs E1 0.57 → 0.10).

### Table 2 — Meta-learners vs the reference, same false model (test, strong / full obs; 3 seeds)
Reference: classical Adagrad with the exact adjoint on the same false model — F1: 37 forward-run equivalents (10
gradients), reached 0.99, IC 0.425, true-model fc 2.47; F2: 24 (10 gradients), 1.00, 0.419, 2.50.
| cell | method | false model | cost (fwd-eq) method / ref | speed-up | reached method / ref | IC RMSE method / ref | true-model fc method / ref | worse basins /200 |
|---|---|---|---|---|---|---|---|---|
| 1 | learned optimizer, unsup, gradient (Δ) | F1 | 32 / 37 | **×1.13** | 0.84 / 0.99 | 0.453 ± 0.027 / 0.425 | 2.86 / 2.47 (+16 %) | 27 |
| 1 | | F2 | 26 / 24 | ×0.94 | 0.92 / 1.00 | 0.432 ± 0.014 / 0.419 | 2.52 / 2.50 (+1 %) | 13 |
| 2 | learned optimizer, sup, gradient (lookahead) | F1 | 48 / 37 | ×0.77 | 0.51 / 0.99 | 0.440 / 0.425 | 3.72 / 2.47 (+51 %) | 77 |
| 2 | | F2 | — | n/a (reached 41 %) | 0.41 / 1.00 | 0.452 / 0.419 | 3.85 / 2.50 (+54 %) | 88 |
| 5 | learned optimization K 20, gradient | F1 | 28 / 37 | **×1.30** | 0.62 / 0.99 | 0.472 ± 0.006 / 0.425 | **2.37 / 2.47 (−4 %)** | 42 |
| 5 | | F2 | 20 / 24 | **×1.25** | 0.62 / 1.00 | 0.453 ± 0.023 / 0.419 | **2.30 / 2.50 (−8 %)** | 42 |
| 7 | learned optimization K 20, no gradient | F1 | 0 / 37 | no model run | 0.07 / 0.99 | 0.519 ± 0.010 / 0.425 | 3.98 / 2.47 (+61 %) | 133 |
| 7 | | F2 | 0 / 24 | no model run | 0.06 / 1.00 | 0.505 ± 0.022 / 0.419 | 4.03 / 2.50 (+61 %) | 133 |
| 7+obs | same + observations as input | F1 / F2 | 0 | no model run | 0.10 / 0.08 | 0.541 / 0.535 | 3.59 / 3.61 | 120 / 121 |
| 3, 3+obs | learned optimizer, unsup, no gradient | F1 / F2 | — | n/a (reached 6–12 %) | ≤ 0.12 | 18–38 (diverge) | 9.2–9.8 | ~198 |
| 4, 4+obs | learned optimizer, sup, no gradient | F1 / F2 | — | n/a (reached 6–12 %) | ≤ 0.12 | 6–19 (diverge) | 8.7–9.6 | ~195 |
| 8 | adjoint-free learned gradient | F1 / F2 | — | n/a (reached 7–8 %) | 0.07 | 3.9 / 3.3 (diverge) | 7.1 / 6.2 | ~190 |
test_ood (reference F1: 44 fwd-eq, IC 0.421, true-model fc 1.47): cell 1 ×0.79 / ×0.73 (F1 / F2) with +87 % / +57 %
forecast error; cell 5 reaches the target in only 28–38 % of cases but gives IC 0.33 (vs 0.42); cell 7 IC 0.32–0.33
with no model run, forecasts +220–270 %. Full tables: tmp/results/m5_summary_{test,test_ood}.md.

### Table 3 — Why the adjoint-free learned gradient (cell 8) fails (seed 0, validation)
| false model | training pairs | train MSE (std units) | val MSE (mean predictor = 1) | val cos(ĝ_o, ∇J_o) | inference cos(ĝ, ∇J) |
|---|---|---|---|---|---|
| F1 | 234 007 (1000 cases) | 0.05 | 2.49 | 0.29 | −0.20 |
| F2 | 258 129 (1000 cases) | 0.015 | 2.21 | 0.17 | −0.10 |

## 3. Findings
1. **Model error dominates the inverse problem.** With either false model, 4D-Var reaches an IC error of 0.42 versus
   0.33 with the true model (first guess 0.645), and the analysis is false-model consistent: its true-model forecast
   (2.47) is worse than its false-model forecast (1.55). Where the lagged first guess is already good (test_ood,
   0.27), false-model 4D-Var makes it worse (0.42).
2. **The emulator is a faithful stand-in for the knowledge model** inside 4D-Var (Table 1: IC 0.419 vs 0.425, same
   gap to E1, same number of gradients); on L63 it is slower per model run than the analytic model (×0.36).
3. **Parameter estimation recovers part of the missing physics**: BPE moves ρ from 27.5 to 27.8 and cuts the
   true-model forecast error by 30 % (gap to E1 +1.02 → +0.45), at ~10× the cost of BICE.
4. **4D-Var without the model's adjoint loses nothing in estimation quality.** Finite differences reproduce the
   exact-adjoint analysis to 10⁻¹⁰ at ×0.52 speed (7 forward runs per gradient; the cost grows with the control size,
   73 runs for weak constraint); the emulator's adjoint with the numerical model's cost gives IC 0.416 and forecast
   2.46 (reference 0.425, 2.47) at ×0.55 speed, with fewer cases reaching the strict 1 % cost target (0.88).
5. **Meta-learners do not accelerate a well-conditioned 4D-Var.** The reference converges in ~10 gradients; the best
   speed-ups are ×1.13 (cell 1, F1; ×0.94 on F2) and ×1.25–1.30 (cell 5), each paid with fewer cases reaching the
   target and more basin switches. Supervised imitation of a 5-step lookahead (cell 2) is slower and worse.
6. **Stopping early regularises against model error.** The fixed-budget learned optimisation (cell 5) does not fit
   the false model fully and its analyses forecast better with the true model (−4 % F1, −8 % F2; test_ood IC 0.33
   vs 0.42).
7. **An amortised correction without any model run is possible but weak**: cell 7 improves the first guess from
   0.645 to 0.52 (0.32 on test_ood) with zero model runs, far from 4D-Var quality.
8. **Without the gradient, a learned optimizer cannot locate the case's minimum** (cells 3, 4: errors grow to 6–38),
   even with the raw observations as input.
9. **The adjoint-free learned gradient fails as built** (Table 3): an unstructured network on 72 innovation features
   memorises ~240 correlated states per case (train MSE 0.02–0.05 vs validation 2.2–2.5) and its gradient is
   anti-aligned at inference.

## 4. Limitations and next steps
- Strong constraint, full observations only for the meta grid; weak / x_only pending.
- Cell 1 hyper-parameters were selected on D1 v1; the other cells use defaults — no meta tuning on D1 v2 yet.
- Cost excludes network evaluations and meta-training cost (amortised over many analyses).
- Cell 8 next: an adjoint-structured architecture (state-dependent operators A_i(x), ĝ_o = Σ_i A_i(x)ᵀ d_i, linear in
  the innovations like a true adjoint) and decorrelated training states sampled around trajectories (targets from
  the emulator's autodiff need no truth).

### Verification
tests: test_d1v2 10/10, test_cost_v2 9/9, test_cell8 5/5, test_m5 18/18, test_m4 19/19, test_m2 8/8; v2 runner
refactor bit-identical; all 72 classical and 180 meta records verified readable, 0 diverged.
