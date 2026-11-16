# Branch m5-meta-grid — findings (paper-ready; fold into memory/findings.md after merge)

# Milestone 5 — Learned optimizers for 4D-Var under model error: a model-error-isolated benchmark (2026-11-15)

## Summary
- With an imperfect model (knowledge model F1, ρ 27.5 instead of 28, or its emulator F2), 4D-Var stalls at an IC error
  of 0.42 versus 0.33 with the true model; the analysis fits the false model and forecasts worse with the true one.
- Joint parameter estimation (BPE) recovers ρ ≈ 27.8 and halves the true-model forecast gap, at ~10× the cost.
- Adjoint-free classical 4D-Var (finite differences, or the emulator's adjoint with the numerical model's cost) loses
  **no estimation quality**, at about half the speed of the exact-adjoint reference on L63.
- **An adjoint-free learned gradient (cell 8a) matches the exact-adjoint reference in quality at ×2.3–2.9 lower
  wall-clock cost** (×2.4–3.9 fewer model-run equivalents) in all four configurations (strong / weak constraint ×
  full / x-only observations), on test and test_ood, for both false models: one forward run per iteration, learned
  tangent-linear operators combined by the exact adjoint recursion; IC error within ±0.01 of the reference, forecast
  within ±2 % (one case −7 %), reached fraction within 0.03, ≤ 4 extra basin switches out of 200 (Table 4).
- **No learned optimizer beats the reference** (classical Adagrad with the exact adjoint), which already converges
  in ~10 gradients on this problem. Best speed-ups: ×1.13 (cell 1, Δ learned optimizer, F1) and ×1.30 / ×1.25 (cell 5, K-20
  learned optimization) — both reach the cost target in fewer cases (0.84, 0.62 vs 0.99).
- Cell 5 regularises against model error (true-model forecast −4 % / −8 % vs the reference); cell 7 improves the
  first guess with **no model run** (IC 0.645 → 0.52); learners without gradient input (cells 3, 4) and the first,
  unstructured version of the learned gradient (cell 8 v1) fail.

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
| 8 v1 | adjoint-free learned gradient (+ Adagrad), MLP | supervised: emulator ∇J_o | x, innovations, k | 1 forward run |
| 8a (v2) | adjoint-free learned gradient (+ Adagrad), adjoint-structured | supervised: emulator one-interval Jacobians | forward-run states, residuals | 1 forward run |
Cells 1–2 rescale the Adagrad teacher step (bounded stretch, K = 20 network applications per gradient); cells 3, 5, 7
use a free update (bounded MLP output × climatological scale); cell 8 predicts only the observation-term gradient
(∇J_b, ∇J_q are analytic) from one forward run: v1 maps (x, innovations, k) to ∇J_o with an MLP trained on states of
an adjoint-free hybrid classical run; v2 (8a) learns the 3 × 3 tangent-linear operator of the false model over one
observation interval as a function of the state, A_θ(x) = I + MLP(x), and forms ∇J_o with the exact adjoint recursion
λ_L = r_L, λ_i = r_i + A_θ(x_{t_i})ᵀ λ_{i+1}, ∇_{x0}J_o = λ_0 (∇_{η_i}J_o = λ_i), r_i = H_iᵀ R⁻¹(H x_{t_i} − y_i), on the
states and residuals of the same forward run. A_θ is trained to match the emulator's one-interval Jacobians (autodiff)
at the forward-run states of 64 B-shaped controls per training case around the first guess (x0_b + s L z, LLᵀ = B,
s ∈ {0.5, 1, 2}); the best epoch minimises the median relative validation error of ∇J_o. It has no iteration input,
since a local tangent-linear operator does not depend on the optimiser iteration. Each learner is trained on the training ICs of its
own false model (3 seeds, best epoch on validation) and applied frozen to test and test_ood. Hyper-parameters: cell 1
uses the configuration selected in Milestone 4 (D1 v1); the others use defaults.

**Metrics and cost.** IC RMSE vs truth; forecast RMSE at 1 TU after the window with the true model (initial-condition
quality) and with the estimation model (operational); cost = model runs until J ≤ 1.01 J* (J* = best final cost of
classical GD / Adagrad on the same model and case), in forward-run equivalents of that model, using the measured cost
of one gradient evaluation (forward + adjoint): 3.7 forward runs for F1, 2.5 for F2 (200 cases batched, one core);
speed-up = reference cost / method cost; "worse basins" = cases ending with a final cost > 1.1 × the reference's.
Network evaluations are not counted in this model-run speed-up; for cell 8a, whose network runs at every iteration,
the measured wall-clock speed-up is also reported (one cell-8a iteration = forward run + network: 35 ms on F1 and
123 ms on F2 per 200 cases, vs 103 ms and 279 ms for an exact gradient; src/measure_cell8_cost.py).
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
| **8a** | **adjoint-free learned gradient v2 (no adjoint)** | F1 | 10 / 37 | **×3.66 (wall-clock ×2.94)** | 1.00 / 0.99 | 0.418 ± 0.003 / 0.425 | 2.45 / 2.47 (−1 %) | 0 |
| **8a** | | F2 | 10 / 24 | **×2.41 (wall-clock ×2.26)** | 1.00 / 1.00 | 0.419 ± 0.000 / 0.419 | 2.51 / 2.50 (+0 %) | 0 |
| 8 v1 | adjoint-free learned gradient v1 (MLP) | F1 / F2 | — | n/a (reached 7–8 %) | 0.07 | 3.9 / 3.3 (diverge) | 7.1 / 6.2 | ~190 |
test_ood (reference F1: 44 fwd-eq, IC 0.421, true-model fc 1.47): cell 8a ×3.42 / ×2.36 (F1 / F2) with IC 0.421 /
0.433 (ref 0.421 / 0.428), reached 0.97, forecast ±1 %, 1 worse basin; cell 1 ×0.79 / ×0.73 with +87 % / +57 %
forecast error; cell 5 reaches the target in only 28–38 % of cases but gives IC 0.33 (vs 0.42); cell 7 IC 0.32–0.33
with no model run, forecasts +220–270 %. Full tables: tmp/results/m5_summary_{test,test_ood}.md.

### Table 3 — Adjoint-free learned gradient: unstructured (v1) vs adjoint-structured (v2 = 8a)
Reference: classical Adagrad with the exact adjoint (F1: 10 gradients = 37 forward-run equivalents = 1.03 s per 200
cases; F2: 24 fwd-eq = 2.79 s). Validation fidelity on seed 0; inference over 3 seeds, test.
| false model | version | training data | val median rel. error of ∇J_o | val median cos | speed-up model runs / wall-clock | IC RMSE (ref) | reached |
|---|---|---|---|---|---|---|---|
| F1 | v1 MLP(x, innovations, k) | 234 007 trajectory states | ~1 (train MSE 0.05, val 2.49) | 0.29 | n/a (diverges) | 3.9 (0.425) | 0.07 |
| F1 | **v2 learned TL operators + adjoint** | 64 000 sampled controls | **0.059** | **0.9998** | **×3.66 / ×2.94** | **0.418** (0.425) | 1.00 |
| F2 | v1 | 258 129 trajectory states | ~1 (train MSE 0.015, val 2.21) | 0.17 | n/a (diverges) | 3.3 (0.419) | 0.07 |
| F2 | **v2** | 64 000 sampled controls | **0.015** | **0.99998** | **×2.41 / ×2.26** | **0.419** (0.419) | 1.00 |
Inference cos(ĝ, ∇J) over all iterations: v2 0.59 (F1) / 0.70 (F2) — near the minimum ∇J_o and ∇J_b nearly cancel,
so small relative errors in ĝ_o dominate the cosine without affecting the analysis; v1 −0.20 / −0.10.

### Table 4 — Adjoint-free learned gradient (cell 8a) in every configuration (3 seeds; test, test_ood in brackets)
Reference for every row: classical Adagrad with the EXACT adjoint, same false model, constraint and observation
pattern (Adagrad lr 0.3 strong / 0.1 weak). Speed-up = reference cost / cell-8a cost; wall-clock speed-up =
(reference iterations × measured gradient time) / (cell-8a iterations × measured cell-8a iteration time), same session
(src/measure_cell8_cost.py: cell-8a iteration = forward run + 9 ms network per 200 cases). Classical adjoint-free
baselines on F1 for comparison: finite differences ×0.52 strong / ×0.05 weak (73 forward runs per gradient), hybrid
(F1 cost + emulator adjoint) ×0.29–0.55.
| constraint / obs | false model | speed-up model runs | speed-up wall-clock | reached 8a / ref | IC RMSE 8a / ref | true-model fc 8a / ref | worse basins |
|---|---|---|---|---|---|---|---|
| strong / full | F1 | ×3.66 [×3.42] | ×2.94 [×2.75] | 1.00 / 0.99 [0.97 / 0.98] | 0.418 / 0.425 [0.421 / 0.421] | 2.45 / 2.47 [1.45 / 1.47] | 0 [1] |
| strong / full | F2 | ×2.41 [×2.36] | ×2.26 [×2.22] | 1.00 / 1.00 [0.97 / 0.99] | 0.419 / 0.419 [0.433 / 0.428] | 2.51 / 2.50 [1.67 / 1.65] | 0 [1] |
| weak / full | F1 | ×3.82 [×3.87] | ×2.70 [×2.73] | 0.93 / 0.95 [0.91 / 0.94] | 0.423 / 0.432 [0.314 / 0.314] | 2.52 / 2.53 [2.03 / 2.18] | 0 [1] |
| weak / full | F2 | ×2.58 [×2.53] | ×2.42 [×2.37] | 0.96 / 0.96 [0.94 / 0.94] | 0.430 / 0.430 [0.328 / 0.328] | 2.49 / 2.50 [2.20 / 2.21] | 1 [2] |
| strong / x_only | F1 | ×3.66 [×3.66] | ×2.75 [×2.75] | 1.00 / 1.00 [0.97 / 0.98] | 0.518 / 0.517 [0.304 / 0.308] | 3.14 / 3.17 [1.45 / 1.47] | 0 [0] |
| strong / x_only | F2 | ×2.53 [×2.45] | ×2.45 [×2.37] | 1.00 / 1.00 [0.97 / 0.98] | 0.529 / 0.527 [0.320 / 0.321] | 3.14 / 3.21 [1.45 / 1.45] | 0 [0] |
| weak / x_only | F1 | ×3.65 [×3.78] | ×2.60 [×2.69] | 0.98 / 0.98 [0.91 / 0.92] | 0.507 / 0.507 [0.283 / 0.285] | 2.92 / 2.90 [1.80 / 1.77] | 0 [4] |
| weak / x_only | F2 | ×2.50 [×2.53] | ×2.28 [×2.30] | 0.98 / 0.98 [0.94 / 0.94] | 0.523 / 0.524 [0.294 / 0.294] | 2.94 / 2.95 [1.73 / 1.75] | 0 [1] |
Sources: tmp/results/m5_summary{,_weak_full,_strong_x_only,_weak_x_only}_{test,test_ood}.md,
tmp/results/cell8a_speedups.json; 54 + 18 records, 0 diverged.

## 3. Findings
1. **Model error dominates the inverse problem.** With either false model, 4D-Var reaches an IC error of 0.42 versus
   0.33 with the true model (first guess 0.645), and the analysis is false-model consistent: its true-model forecast
   (2.47) is worse than its false-model forecast (1.55). Where the lagged first guess is already good (test_ood,
   0.27), false-model 4D-Var makes it worse (0.42).
2. **The emulator is a faithful stand-in for the knowledge model** inside 4D-Var (Table 1: IC 0.419 vs 0.425, same
   gap to E1, same number of gradients); on L63 it is slower per model run than the analytic model (×0.36).
3. **Parameter estimation recovers part of the missing physics**: BPE moves ρ from 27.5 to 27.8 and cuts the
   true-model forecast error by 30 % (gap to E1 +1.02 → +0.45), at ~10× the cost of BICE.
4. **4D-Var without the model's adjoint loses nothing in estimation quality — and a learned adjoint makes it cheaper
   than the exact one.** The adjoint-structured learned gradient (8a) reaches the same analysis as the exact-adjoint
   reference (IC 0.418 vs 0.425 F1, 0.419 vs 0.419 F2; forecast ±1 %; all cases reach the target, no extra basin
   switches) at ×2.9 (F1) / ×2.3 (F2) lower wall-clock cost, because each iteration needs one forward run of the
   false model instead of a forward + adjoint run. The result holds in all four configurations — including the weak
   constraint, where the learned recursion also gives the gradient with respect to the model-error increments, and
   x-only observations — and out of distribution: wall-clock ×2.3–2.9, IC error within ±0.01, forecast within ±2 %
   (Table 4). Finite differences, the exact adjoint-free alternative, become ×0.05 for the weak constraint (73
   forward runs per gradient), whereas the learned gradient's cost does not grow with the control size.
   The classical adjoint-free baselines are exact but slower: finite differences reproduce the
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
9. **Structure, not data volume, makes the learned gradient work** (Table 3): an unstructured network on 72
   innovation features memorises ~240 correlated states per case and diverges at inference; learning only the local
   tangent-linear operator (a property of the dynamics, not of the case) and combining it with the exact adjoint
   recursion generalises from 64 decorrelated controls per case (validation error 1.5–6 %). Training the operators
   through the full 11-interval product instead of matching local Jacobians is ill-conditioned (val cos 0.39).

## 4. Limitations and next steps
- Strong constraint, full observations only for the meta grid; weak / x_only pending.
- Cell 1 hyper-parameters were selected on D1 v1; the other cells use defaults — no meta tuning on D1 v2 yet.
- Cost excludes network evaluations and meta-training cost (amortised over many analyses).
- Cell 8a is supervised on the EMULATOR's Jacobians (F2 is the emulator itself; for F1 the emulator stands in for
  the missing adjoint). On L63 a forward run is cheap, so the network (9 ms per 200 cases) is a third of an F1 forward
  run; for large models the network cost is relatively smaller and the saving tends to the adjoint/forward cost
  ratio. All four constraint / observation configurations are covered (Table 4).
- The operator network has no iteration input (deviation from the x / ∇J / k input rule; pilot to confirm).

### Verification
tests: test_d1v2 10/10, test_cost_v2 9/9, test_cell8 8/8, test_m5 18/18, test_m4 19/19, test_m2 8/8; v2 runner
refactor bit-identical; all 72 classical and 180 meta records verified readable, 0 diverged.

### Table 5 — G1 gradient fidelity vs the TRUE dynamics (test, strong / full; src/measure_g1_v2.py, 2026-11-16)
G1 = cos(g_method, ∇J_true) with J_true = same problem (obs, first guess, hybrid B) with the true L63 (ρ 28), at the
same control: first guess x0_b [true IC x0_t]; median over 200 cases (learned gradients: mean over 3 seeds).
| method | G1 cos @x0_b | G1 cos @x0_t | rel. err @x0_b |
|---|---|---|---|
| E1 true model | 1 | 1 | 0 |
| F1 exact adjoint (= GD, Adagrad, FD; cells 1/2/5 on F1) | 0.992 | 0.881 | 0.55 |
| F1 weak constraint (full control) | 0.983 | 0.720 | 0.55 |
| F2 emulator adjoint (cells 1/2/5 on F2) | 0.990 | 0.862 | 0.58 |
| hybrid (F1 problem, emulator adjoint) | 0.990 | 0.862 | 0.58 |
| BPE, x0-sensitivity of L63(θ_a) | **0.997** | 0.819 | **0.46** |
| cell 8 v1 F1 / F2 | 0.13 / 0.15 | 0.14 / 0.13 | 1.25 / 1.08 |
| cell 8a F1 / F2 | 0.990 / 0.990 | 0.877 / 0.855 | 0.55 / 0.58 |
Every false-model gradient (incl. cell 8a) shares the same ~0.99 / ~0.87 similarity to the true sensitivity: the
learned adjoint adds no error beyond the false model's own. Only BPE (recovered ρ) improves it at the first guess.
Near x0_t ∇J_true is small, so relative errors there are large (2–4) and the cosine is noisier.

### Table 6 — Total time: time-to-solution per 200-case batch and one-off training (test, strong / full; 2026-11-16)
Time-to-solution = iterations to J ≤ 1.01 J* × measured time per iteration on one core for 200 cases (gradient: true
0.099 s, F1 0.102 s, F2 0.283 s; cell-8a iteration F1 0.035 s, F2 0.125 s; network cost of cells 1/2/5 not included,
small). Gain vs the F1 Adagrad exact-adjoint reference (1.02 s). One-off training time = meta-training on 1000
training ICs (+ validation-epoch selection), mean over 3 seeds, from tmp/logs/m5_*_val.log (log span minus the
inference wall time; jobs ran concurrently, so this is an upper bound; cell 8a excludes the Jacobian-pair collection);
emulator training 637 s. Break-even = cases needed before the saved time pays back the training time.
The record attribute wall_s is NOT comparable across methods (batched runs continue to max_iter for the slowest
case; meta runs also compute the exact gradient for diagnostics).
| method | time / 200 cases | gain vs F1 ref | gain vs same-model ref | one-off training | break-even |
|---|---|---|---|---|---|
| E1 true model | 0.84 s | ×1.20 | — | — | — |
| F1 Adagrad exact (ref) | 1.02 s | ×1 | ×1 | — | — |
| F1 GD | 12.3 s | ×0.08 | ×0.08 | — | — |
| F1 weak constraint | 4.98 s | ×0.20 | — | — | — |
| F1 finite differences | 1.95 s | ×0.52 | ×0.52 | — | — |
| hybrid (emulator adjoint) | 1.86 s | ×0.55 | ×0.55 | 637 s (emulator) | never |
| F2 emulator Adagrad | 2.83 s | ×0.36 | ×1 | 637 s (emulator) | never |
| BPE | 9.70 s | ×0.10 | — | — | — |
| cell 1 F1 / F2 | 0.89 / 3.00 s | ×1.15 / ×0.34 | ×1.13 / ×0.94 | 1671 / 1994 s | ~2.6 M cases / never |
| cell 2 F1 / F2 | 1.34 s / n/a | ×0.76 / n/a | ×0.77 / n/a | 830 / 810 s | never |
| cell 5 F1 / F2 | 0.78 / 2.31 s | ×1.31 / ×0.44 | ×1.30 / ×1.25 | 1678 / 2511 s | ~1.4 M / ~1.0 M cases |
| cell 7 F1 / F2 | ≈ 0 (network only) | — (IC 0.52, not converged) | — | 99 / 183 s | — |
| cell 8a F1 / F2 | **0.35 / 1.25 s** | **×2.91** / ×0.82 | **×2.94 / ×2.26** | 685 (+637 emulator) / 657 s | ~0.4 M / ~83 k cases |

### Table 7 — Meta-speed check: classical floor and speed-retuned learners (strong / full; 2026-11-16)
Gradients (forward + adjoint) to J ≤ 1.01 J*, median; learners mean over 3 seeds. Selection on F1 val only (rule:
reached ≥ ref − 0.05, true-model fc ≤ 1.05 × ref, then fewest gradients): cell 1 → Adagrad teacher lr 0.1, stretch
bound 10, 2 align steps (val 7 vs 9.5 gradients, ×1.36); cell 5 with the speed loss mean_m J(x^m)/J(x^0) reached the
target in only 39–73 % of val cases for every K / step scale (none eligible; K 20 carried to test as the most reliable).
The F2 runs reuse the F1-selected settings (no F2 val tuning).
| method | F1 test: gradients (speed-up) | reached | IC | true fc @1TU | worse basins | F2 test: gradients (speed-up) | reached | true fc |
|---|---|---|---|---|---|---|---|---|
| reference Adagrad lr 0.3 | 10 (×1) | 0.995 | 0.425 | 2.47 | 0 | 10 (×1) | 1.00 | 2.50 |
| floor: L-BFGS, strong Wolfe (no training) | **5.5 (×1.82)** | 1.00 | 0.415 | — | 0 | **6 (×1.67)** | 1.00 | — |
| floor: Gauss-Newton | 1 iteration ≈ 3–4 gradient-eq. (×~3; ×1.3 measured with autodiff TL) | 0.99 | 0.418 | — | 1 | 1 iteration (×~3; ×1.0 measured) | 0.99 | — |
| cell 1, M5 (D1 v1 settings) | 8.8 (×1.13) | 0.84 | 0.453 | 2.86 | 27 | 10.7 (×0.94) | 0.92 | 2.52 |
| **cell 1, retuned** | **7.8 (×1.28)** | **0.97** | **0.423** | 2.53 (+2 %) | 5 | 6.3 (×1.58) | 0.67 | 3.23 (+29 %) |
| cell 5, M5 (J(x^K), K 20) | 7.7 (×1.30) | 0.62 | 0.472 | 2.37 | 42 | 8 (×1.25) | 0.62 | 2.30 |
| cell 5, speed loss K 20 | 6.3 (×1.58) | 0.71 | 0.455 | 2.41 | 27 | 6.7 (×1.50) | 0.70 | 2.38 |
test_ood (F1): reference 12 gradients; retuned cell 1 9.8 (×1.22) but reached 0.90, fc 2.38 vs 1.47; cell 5 speed loss
13.2, reached 0.44. Sources: tmp/results/speed_{f1,emulator}_{val,test,test_ood}.{md,csv}, floor_v2_test.json.
**Findings.** (i) The headroom over Adagrad is small: a classical quasi-Newton method needs 5.5–6 gradients (×1.7–1.8)
with no training and full reliability; Gauss-Newton needs one iteration (~×3 in model-run equivalents). (ii) Retuning on
D1 v2 makes cell 1 a valid speed-up on F1 (×1.28, same reach, IC and forecast as the reference — the M5 setting failed
the quality rule), closing ~50 % of the Adagrad → L-BFGS gap, but it does not reach the classical floor, and the
setting does not transfer to F2 or out of distribution without re-tuning. (iii) A speed-weighted loss makes cell 5
descend faster (×1.5–1.6) but a fixed-budget learned update reaches the 1 % target in only ~70 % of cases.
