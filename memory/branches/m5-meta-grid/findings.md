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

## Addendum (2026-11-16) — model behaviour and dense observations (strong / full, test, classical only)
**Models.** From the same IC, F1 and F2 depart from the truth by |err| > 1 after a median 0.36 TU (i.e. inside the
window); F2 reproduces F1 closely. All three have the same IC-error growth rate (0.89–0.90 per TU) and predictability
(~7.7 TU from |δ| = 10⁻³). Figures: tmp/figs/models_{timeseries,phase,sensitivity}.
**Dense observations (111 obs over the same 1.1-TU window, nested with D1 v2).** Target J ≤ 1.01 J*, J* = best final
J of Adagrad / GD / L-BFGS / Gauss-Newton per case; medians over 200 cases; table tmp/results/dense/summary_test.md.
| obs | model | Adagrad | GD | L-BFGS | GN (grad-eq) | IC RMSE (L-BFGS) | true-model fc @1 TU (L-BFGS) |
|---|---|---|---|---|---|---|---|
| 12 | F1 | 10 | 125 | 6 | 8.0 | 0.415 | 2.43 |
| 111 | F1 | 28 | 228 | 6 | 8.3 | 0.376 | 2.30 |
| 12 | E1 | 9 | 99 | 5 | 16.2 | 0.306 | 0.65 |
| 111 | E1 | 7 | 112 | 5 | 17.1 | 0.196 | 0.25 |
Dense obs make first-order optimizers on F1 ~3× slower (Adagrad 10 → 28 gradients, reached 0.82) while L-BFGS stays at
6: the room for a learned optimizer grows against Adagrad (×4.7 floor) but not against L-BFGS. With the true model
the extra obs cut the 1-TU forecast error 2.6×; with F1 only 5 % — the analysis is limited by model error, not by
observations. GN costs ~8 gradient-equivalents per iteration here because the 3 tangent-linear runs use forward-mode AD.

### Dense observations, weak constraint (test, full obs; eta at every obs time; table summary_test_weak.md)
| config (control size) | model | Adagrad (reached) | GD (reached) | L-BFGS | IC RMSE (L-BFGS) | fc @1 TU est. model / true model (L-BFGS) |
|---|---|---|---|---|---|---|
| sparse, σ_q 0.1 (36) | F1 | 41 (0.83) | 122 (0.74) | 9 | 0.406 | 1.24 / 2.48 |
| dense, σ_q 0.1 (333) | F1 | 113 (0.60) | 591 (0.29) | 17 | 0.313 | 0.95 / 2.68 |
| dense, σ_q 0.1/√10 (333) | F1 | 242 (0.57) | 148 (0.19) | 14 | 0.332 | 0.73 / 2.31 |
| sparse, σ_q 0.1 | E1 | 45.5 (0.87) | 294 (0.72) | 9 | 0.320 | 0.65 |
| dense, σ_q 0.1 | E1 | 188 (0.53) | 627 (0.27) | 20 | 0.268 | 0.65 |
| dense, σ_q 0.1/√10 | E1 | 304 (0.48) | 672 (0.16) | 14 | 0.210 | 0.32 |
Weak constraint with dense observations is the first configuration where first-order optimizers break down: Adagrad
needs 113–304 gradients and reaches the 1 % target in only about half the cases; GD in under 30 %. L-BFGS needs 14–20 (all
cases). Against Adagrad, a learned optimizer now has about 8–17× room; against L-BFGS the floor is still about 15 gradients.
The η increments improve the OPERATIONAL F1 forecast (1.24 → 0.73 with the scaled Q) but not the true-model forecast of the
F1 analysis: η fits F1's tendency error in-window, which does not carry into a true-model forecast. With the true model
(no model error) σ_q 0.1 lets η fit observation noise (forecast 0.65 vs 0.25 in strong); the scaled Q is better (0.32).

### Meta-learners on the long problem: dense obs, weak constraint (F1, σ_q 0.1/√10, control 333; 3 seeds)
Cost = F1 forward-run equivalents to J ≤ 1.01 J* (1 gradient = 3.82 forward runs; J* = best classical final J);
median over cases reaching the target, so it must be read with "reached". Tables: tmp/results/dense/meta_summary_*.md.
| method | test: cost / reached / IC / fc@1TU F1 | test_ood: cost / reached / IC / fc@1TU F1 |
|---|---|---|
| Adagrad (exact adjoint, M5 reference) | 924 / 0.57 / 0.447 / 0.92 | 1518 / 0.54 / 0.225 / 1.26 |
| L-BFGS | 53 / 1.00 / 0.332 / 0.73 | 53 / 1.00 / 0.260 / 1.03 |
| cell 1 (Δ learned optimizer) | 178 ± 2 / 0.78 / 0.377 / 0.88 | 268 ± 15 / 0.68 / 0.277 / 1.57 |
| cell 5 (learned optimization, K 20) | 25 ± 2 / 0.53 / 0.382 / 0.82 | 38 ± 2 / 0.17 / 0.253 / 1.43 |
| cell 8a (adjoint-free learned gradient) | 225 ± 8 / 0.57 / 0.448 / 0.92 | 390 ± 4 / 0.54 / 0.224 / 1.25 |
Fixed budget of 20 gradients (test): fraction at target Adagrad 0.06, cell 1 0.17, cell 5 0.43–0.48, L-BFGS 0.65;
median excess J/J* − 1: 0.064, 0.052, 0.011–0.012, 0.0035. On the long problem the learned optimizers now clearly beat their
teacher: cell 1 reaches the target in 0.78 vs 0.57 of the cases at ~5× fewer model runs (consistent over seeds), cell 8a
reproduces Adagrad (same reached / IC / forecast) with forward runs only (×4.1 fewer model runs) but is ×0.64 in
wall-clock on L63 (the learned adjoint recursion over 110 intervals costs more than the exact L63 adjoint). Cell 5 is the
fastest method of all for the first ~10 gradients (below the L-BFGS curve) but its fixed K = 20 stops it at ~1 % above J*,
and it does not generalise to test_ood (reached 0.17, 53 worse basins). No learner beats L-BFGS at reaching the 1 % target.

#### L-BFGS teacher and K = 50 (same problem; fraction of cases at the 1 % target after N gradients)
| N | Adagrad | L-BFGS fixed step (teacher) | L-BFGS line search | cell 1, Adagrad teacher | cell 1, L-BFGS teacher | cell 5 K 20 | cell 5 K 50 |
|---|---|---|---|---|---|---|---|
| test 50 | 0.12 | 0.17 | 0.93 | 0.41 | 0.59 | 0.45 | 0.51 |
| test 100 | 0.17 | 0.53 | 0.98 | 0.51 | 0.72 | 0.45 | 0.51 |
| test_ood 50 | 0.05 | 0.05 | 0.95 | 0.18 | 0.61 | 0.07 | 0.06 |
| test_ood 100 | 0.12 | 0.34 | 0.97 | 0.45 | 0.75 | 0.07 | 0.06 |
A better teacher gives a better learned optimizer: cell 1 on the L-BFGS direction learns the step length (median
stretch near its ×11 cap: an effective lr ~0.37 vs the stable fixed lr 0.033) and beats its own teacher (×2.8 fewer model
runs to the target) and the Adagrad-teacher cell 1 (reached 0.84 vs 0.78; IC 0.350 vs 0.377), and it generalises
out of distribution (test_ood reached 0.84, ×11.6 vs Adagrad), but 3–6 % of cases diverge (no line-search safeguard)
and it stays ~2.4× costlier than L-BFGS with a line search. Cell 5 with K = 50 does not close its gap: final cost
unchanged, 0.60 reached vs 0.53 on test, and test_ood worse (0.21 reached, 71 worse basins, IC 0.38 > first guess
0.27). The learned unroll is fast but does not generalise.

#### Learned optimizer with the reference L-BFGS as teacher (same problem; 2026-11-18)
**Method.** The teacher of cell 1 is now the reference method itself: one iteration of L-BFGS with a strong-Wolfe
line search, with the settings of the classical baseline (history 10, unit initial step, c₁ = 10⁻⁴, c₂ = 0.9, at most
25 trials). At each outer iteration the teacher computes the L-BFGS direction from the curvature pairs of the moves
actually applied, runs the line search from the current control, and returns its accepted step f; the learned network then
rescales f component-wise exactly as before (K sub-steps, stretch in [0, 11]; Δ inner loss; training on the 1000 training
cases, best epoch on validation, frozen inference on test and test_ood). With a zero stretch the learner is identical
to the reference; a unit test checks that the teacher alone reproduces the iterates of torch.optim.LBFGS. Cost is
counted in gradient evaluations (forward + adjoint; 1 gradient = 3.82 F1 forward runs): every line-search trial
counts as one gradient, and the outer gradient is not counted when the control is the point just accepted by the line
search, whose cost and gradient are already known (as in the classical method). Two inner chain lengths K were
tested (20, the default, and 50). Runs were stopped after 2 seeds (K 20) and 1 seed (K 50), since the seeds agreed
closely; one-off training took 7–10 h per seed (upper bound, concurrent runs).

**Iterations versus gradient evaluations.** An *iteration* is one update of the control; its *cost* is the number of
gradient evaluations it needs. For the reference, the line search usually accepts its first trial, whose gradient is
then reused at the next iteration, so one iteration costs about one gradient (12 iterations, 14 gradients to the target).
A learner that moves the control away from the accepted point must evaluate a new gradient there and, in addition, run
the teacher's line search, so each iteration costs at least two gradients.

**Table — dense observations, weak constraint, F1 (control dimension 333); medians over cases, learners averaged over
seeds.** Target: J ≤ 1.01 J*.
| method (seeds) | iterations to target | gradients to target | cost (F1 fwd-eq) | reached | worse basins /200 | IC RMSE | fc @1 TU (F1) | fc @1 TU (true) | learned stretch |
|---|---|---|---|---|---|---|---|---|---|
| *test* | | | | | | | | | |
| L-BFGS, strong Wolfe (reference) | 12 | 14 | 53 | 1.00 | 0 | 0.332 | 0.73 | 2.31 | — |
| cell 1, reference teacher, K 20 (2) | 11–11.5 | 22.5–24 | 89 ± 3 | 0.94 | 0 | 0.344 | 0.72 | 2.30 | ×1.29–1.33 |
| cell 1, reference teacher, K 50 (1) | 11 | 23 | 88 | 0.93 | 2 | 0.343 | 0.75 | 2.30 | ×1.35 |
| *test_ood* | | | | | | | | | |
| L-BFGS, strong Wolfe (reference) | 12 | 14 | 53 | 0.99 | 0 | 0.260 | 1.03 | 2.01 | — |
| cell 1, reference teacher, K 20 (2) | 11–12 | 24 | 92 ± 0 | 0.94 | 5 | 0.253 | 1.05 | 2.06 | ×1.29–1.32 |
| cell 1, reference teacher, K 50 (1) | 12 | 24 | 92 | 0.94 | 5 | 0.240 | 1.05 | 2.02 | ×1.36 |
For comparison (same problem, cost in F1 fwd-eq on test): cell 1 with the Adagrad teacher 178, with the fixed-step L-BFGS
teacher 129, Adagrad 934.

**Findings.**
1. *A learned optimizer does not improve on its teacher when the teacher already chooses its step length well.* With
   the reference L-BFGS as teacher, the learner needs as many iterations as the reference (11–12 vs 12) but 1.7× more
   model runs (×0.58–0.60 speed-up on test and test_ood), and it reaches the target in fewer cases (0.93–0.94 vs
   0.99–1.00). The analysis quality is unchanged (IC RMSE and forecasts within the seed spread of the reference).
2. *What the learner learns is a step-length correction.* The network converges to a nearly uniform over-stretch of
   the accepted step (median ×1.29–1.36 across runs; interquartile range within ×1.27–1.45) and does not change the search direction. The Δ inner loss
   rewards the first-order decrease of the cost along the move, which favours longer steps; it does not see the
   gradient evaluation that is lost by leaving the line-search point. Against teachers with a poorly tuned step
   (Adagrad, fixed-step L-BFGS) the same correction gave a ×2.8–5 gain; against a line-search teacher there is no step-length
   error left to correct.
3. *A longer inner chain does not help.* K = 50 gives the same iterations, cost and reach as K = 20.
4. *Consequence for amortisation.* Because this learner still requires the teacher's gradient and line search at
   every iteration, its per-case inference cost is bounded below by that of the reference; no number of cases can
   pay back its training. A cost reduction at inference would require a learner that replaces gradient evaluations
   (fewer model runs per iteration), not one that rescales the step of a converged classical method.
Sources: src/dameta/da/reference_step.py (method lbfgs_wolfe), src/scripts/dense_meta_job.sh (c1, c1k50; METHOD=lbfgs_wolfe),
records tmp/results/m5dwW_f1/, teacher check tmp/results/dense/*_lbfgswolfecheck_*.nc, tables
tmp/results/dense/meta_summary_{test,test_ood}.md.

#### Learned Δ optimizer without the teacher at inference: conditioning, history, multi-step targets (dev testbed; 2026-11-18/19)
**Setting.** Same long problem (dense observations, weak constraint, F1, σ_q 0.1/√10, control 333). Development testbed:
one seed, 10 or 100 training cases, selection on the first 20 validation cases; final check on the first 200 validation
cases (inference only). The learner outputs the update Δ̂ itself (no teacher at inference, one gradient per iteration):
`delta_exact` = supervised imitation of the teacher update (relative error |Δ̂ − Δ|² / |Δ|²), `delta` = unsupervised.
Reference: L-BFGS with strong-Wolfe line search (the classical baseline above; same settings). Cost to target = gradient
evaluations (line-search trials included) to J ≤ J*(1 + tol), J* = the reference's final J of the same case; speed-up =
median reference cost / median learner cost over the cases both reach.

**1. Conditioning and training (supervised, 20 val cases).** Hypotheses tested: (a) badly scaled inputs (state
standardised by climatology, |g| removed), (b) under-training (494 teacher pairs, 1 Adam step per epoch), (c) ocean-code
scaling unsuited to L63. Diagnostic on the teacher pairs: the teacher step in gradient units, r = log10 |Δ|/|g|, spans
3.2–3.5 decades (x0) and 1.6–2.2 (η) per control block; the 4D-Var control-variable transform (v₀ = L⁻¹(x₀ − x₀ᵇ), B = LLᵀ;
w = η/σ_q) does not narrow it (it only shifts η by log10 σ_q²) and lowers cos(Δ, −g) (Adagrad 0.85 → 0.23); the gradient
magnitude explains 35–53 % of the variance of r. Training with Adam (batch 64, ×10–15 more steps) or full-batch L-BFGS
fits the training pairs (relative error 0.003 Adagrad teacher, 0.13 L-BFGS teacher, vs 0.86 before) but the validation
error does not improve (best epoch 0–1, ≥ 0.83). With the L-BFGS teacher the imitation error stays ≥ 0.97 in every
coordinate system; scaling with 10 / 50 / 200 / 1000 training cases gives 0.98 / 0.97 / 0.96 / 0.95 (0.86 with the
previous move and gradient change as input). → The limit is information (the L-BFGS memory), not conditioning or
optimisation.

**2. History input (supervised; 20 val cases; reference 15.0 / 27.0 gradients to tol 1 % / 0.1 %).** Input = the
L-BFGS memory: the m = 10 most recent pairs (s = x^{k+1} − x^k, y = g^{k+1} − g^k), one token per observation time with
the state and gradient, 1-D convolution along time (no observations). Output relative to a base direction: −g, or the
L-BFGS two-loop direction −Hg of the same memory (the untrained learner is then fixed-step L-BFGS). Iteration 0 = one
classical strong-Wolfe iteration (counted), the learner from k = 1. Every reference step after the first is exactly
t × (two-loop direction of its memory), with t = 1 in 91 % of the steps (t < 1 in 8 %, 5th percentile 0.28): the line
search mostly accepts its first trial, so the reference already costs about one gradient per iteration.
| learner | to 1 %: gradients (reached, speed-up) | to 0.1 % | final J / IC / diverged |
|---|---|---|---|
| untrained, base −Hg (fixed-step L-BFGS) | 13.5 (0.90, ×1.07) | 23.0 (0.85, ×1.04) | 170.1 / 0.321 / 1 |
| base −Hg, 10 cases | 12.5 (0.80, ×1.08) | 15.5 (0.50, ×1.06) | 170.4 / 0.370 / 0 |
| base −Hg, 100 cases, log loss | 13.0 (0.80, ×1.08) | 16.0 (0.65, ×1.25) | 174.8 / 0.328 / 0 |
| base −g (history as input only) | 9.0 (0.10) | 15.0 (0.05) | 187 / 0.407 / 0 |
| observation input (previous version) | 22.5–48.5 (0.20–0.30, ×0.12–0.40) | — | 190–11086 |
The memory makes the learner as fast as the reference (×0.1–0.4 before); without the −Hg base it cannot rebuild the
L-BFGS direction. Selection note: the median validation error is 0 for the untrained −Hg learner (median t = 1), so the
best epoch is chosen on the mean of log(1 + relative error).

**3. Multi-step targets, online training, safeguard.** Two-step target: the move at k is supervised by the teacher
displacement Δ_k + Δ_{k+1} = x^{k+2} − x^k (one gradient per two teacher steps: ceiling ×2); three steps likewise.
Online variant: the teacher run is replayed in iteration order (teacher forcing, identical to a live run), one Adam step
per iteration on the moves made at k once their target is known. Inference safeguard (backtrack): the gradient
evaluation at x^{k+1} also returns J(x^{k+1}); a move that does not decrease J is rejected and halved, with no extra model
run (memory, stopping rule and returned analysis = last accepted iterate). Imitation error (mean log(1 + rel)): 0.17
one step vs 0.39 two steps (100 cases). On 20 val cases the safeguard removed every divergence; three steps were no
better than two; online and offline were within noise (1–3 cases).

**Table — 200 validation cases, inference only, all learners with the safeguard.** Reference: J* 168.63, IC 0.332,
14.0 / 25.0 gradients to tol 1 % / 0.1 % (reached 1.00). Wall clock: all 200 cases in one batched run per method,
1 thread, own core; every method ran its 150-iteration budget (some case never met the stopping tolerance).
| method | to 1 %: gradients (reached, speed-up) | to 0.1 % | final J / IC / diverged | wall, 200 cases (per iteration) |
|---|---|---|---|---|
| L-BFGS, strong Wolfe (reference) | 14.0 (1.00) | 25.0 (1.00) | 168.63 / 0.332 / 0 | 225.9 s (1506 ms) |
| untrained fixed-step L-BFGS (Wolfe first step) | 13.0 (1.00, ×1.08) | 25.0 (0.96, ×0.96) | 168.61 / 0.334 / 0 | 89.9 s (599 ms) |
| two-step, offline, 10 cases | 13.0 (0.88, ×1.00) | 24.0 (0.75, ×0.88) | 169.01 / 0.335 / 0 | 79.1 s (528 ms) |
| two-step, offline, 100 cases | 10.0 (0.76, ×1.10) | 19.0 (0.58, ×1.05) | 169.42 / 0.365 / 0 | 86.8 s (578 ms) |
| two-step, online, 10 cases | 11.0 (0.89, ×1.18) | 21.0 (0.83, ×1.07) | 168.98 / 0.345 / 0 | 72.9 s (486 ms) |

**Findings.**
1. *Learned optimizers beat L-BFGS without a line search, not L-BFGS with one.* Against fixed-step L-BFGS (and Adagrad)
   the learned step length gives ×2.8–5 fewer model runs (previous subsection). Against the strong-Wolfe reference no
   learner is faster in gradient evaluations by more than ×1.0–1.2, and the trained ones reach the target in 11–24 %
   (1 %) / 17–42 % (0.1 %) fewer cases: on their own trajectories they settle in nearby, slightly worse local minima
   (median final J within 0.5 % of the reference, IC 0.335–0.365 vs 0.332).
2. *There is little room to beat it here.* The line search accepts its first trial in 91 % of the steps, so the
   reference spends about one gradient per iteration; imitation is bounded by its iteration count, and a two-step target
   (ceiling ×2) is harder to imitate (error 0.39 vs 0.17) and gains ≤ ×1.2.
3. *A free safeguard makes teacher-free inference as reliable as the reference.* Rejecting and halving moves that raise J
   (no extra model run) removes all divergences; fixed-step L-BFGS with a classical first step and this safeguard matches
   the reference in quality and reach (1.00 / 0.96) at the same gradient count.
4. *Wall clock.* All teacher-free methods are ×2.5–3.1 faster than the reference in our implementation, because they skip
   the line-search trials and its per-case fallback loop; the untrained safeguarded fixed-step L-BFGS already gets ×2.5,
   so this gain comes from dropping the line search, not from learning; the network itself is cheap.
5. *Next step if pursued:* steps better than the teacher's need an unsupervised objective (J after the move, or summed
   over a few unrolled moves), starting from the safeguarded fixed-step L-BFGS learner.

**Reproduction (dev testbed).** `bash src/scripts/dev_delta.sh <name> delta_exact lbfgs_wolfe "<overrides>"` (dense weak
F1, val split; env NTR = train cases (10), NVAL = val cases (20), PC_OVERRIDE = teacher-pair cache), on the HPC through
`sbatch src/scripts/sb_parallel.sh <command list> <project root>`. Common history overrides: `delta_exact.hist_m=10
delta_exact.first_step=teacher delta_exact.hidden=32 delta_exact.batch_size=64 delta_exact.select=logmean
delta_exact.base=lbfgs`; then untrained `epochs=0`; 10 cases `epochs=300`; 100 cases (NTR=100) `epochs=100
train_loss=logmean`; two / three steps `horizon=2|3 train_loss=logmean`; online `train_mode=online online_cases=10|100`;
inference with the safeguard `checkpoint=<learner .pt> delta_exact.safeguard=backtrack` (NVAL=200 for the final table).
Conditioning runs: `delta_exact.obs_input=true delta_exact.arch=temporal delta_exact.hidden=32` with `batch_size=64` or
`optimizer=lbfgs` (Adagrad teacher: `max_train_pairs=3000`), and `coords=cvt grad_input=fixed obs_feat=innov`.
Reports: `python src/diag_cvt.py`; `python src/report_dev_hist.py [--nval 200] <names>` (classical reference in chunks:
`--nval 200 --chunk a b`); timing `python src/time_dev_hist.py classical | <name> <checkpoint>`. Imitation scaling:
`python src/diag_imitation.py collect|fit`. Logs tmp/logs/{dev_delta_*,report_dev_hist*,diag_cvt,timing_val200_*}.log,
records tmp/results/dev_delta/, learners tmp/meta/dev_delta/.
Earlier list files (dense meta runs, speed sweep) are reproduced by `bash src/scripts/dense_meta_job.sh <cell> <seed>
"test test_ood" [overrides]` for cells c1, c1k50, c5, c5k50, c8a (`learned_grad.epochs=100`), seeds 0–2, and by
`bash src/scripts/speed_sweep_job.sh <model> <name> <teacher lr> <seed> "<overrides>" [splits]` with the configurations of
Table 7.
Table 7 configurations (`speed_sweep_job.sh f1|emulator <name> <teacher lr> <seed> "<overrides>"`; final runs seeds 0–2,
splits "test test_ood"). Cell 1 common overrides: `variant=delta delta.output=stretch delta.episode_iters=100
delta.w_perf_delta=2.5 delta.grad_trans_steps=2 delta.meta_lr=1e-2 delta.n_epochs=3 delta.select_epoch=true
delta.train_seed=<seed>`; val grid (seed 0): teacher lr {0.3, 0.1} × `net.output_scale` {10, 20} × `delta.grad_align_steps`
{0, 2} (+ `delta.select_k=5` variant); selected `c1_lr01_os10_a2` = lr 0.1, `delta.grad_align_steps=2 net.output_scale=10`.
Cell 5: `variant=unroll unroll.seed=<seed> unroll.loss=sum|final unroll.K=<K>`; val grid K {5, 10, 20} sum, K 10 final,
K 10 sum with `unroll.step_scale=0.3` (teacher lr 0.3); carried to test `c5_sum_K20` (loss sum, K 20).
