# Branch m2-classical-4dvar — findings (paper-ready; fold into memory/findings.md on main after merge)

## Milestone 2 — classical strong/weak-constraint 4D-Var (E1)

### Conditioning of the L63 4D-Var problem (Hessian at the background, 20 val cases, 2026-11-10)
| Setting | median λ_max | max λ_max | median λ_min | min λ_min | median cond. |
|---|---|---|---|---|---|
| strong, full obs | 48.2 | 1.69e4 | 1.11 | −7.68e3 | 46.3 |
| strong, x-only | 6.0 | 515 | 1.13 | −5.93e3 | 3.96 |
| weak, full obs | 274 | 7.47e4 | −0.004 | −3.42e4 | 263 |
| weak, x-only | 125 | 2.35e3 | 1.11 | −2.64e4 | 103 |

Curvature is heavy-tailed across cases (λ_max spans > 2 orders of magnitude) and some cases are non-convex at the
background (negative eigenvalues). A single fixed gradient-descent step size cannot be both stable for the stiffest
cases (lr < 2/λ_max ≈ 1e-4) and efficient for typical ones (≈ 2/48 ≈ 0.04). The weak-constraint problem is further
ill-conditioned by the Q⁻¹ = 100 block on η. This is the case-dependent step-size problem that a learned,
per-component stretch of the teacher step (Meta-B) is designed to address.

### Learning-rate selection (val, first 50 cases; rule pre-registered: min median final J, ≤1 % diverged)
| Setting | Adagrad lr (median J, median evals) | GD lr (median J, median evals) |
|---|---|---|
| strong / full | **0.1** (15.83, 622) | **0.001** (16.87, 1000 = budget) |
| strong / x-only | **0.1** (5.46, 324) | **0.001** (5.39, 1000) |
| weak / full | **0.1** (16.12, 644) | **0.0003** (17.08, 1000) |
| weak / x-only | **0.1** (5.23, 340) | **0.0003** (5.37, 1000) |
Grids: strong lr ∈ {1e-3, 3e-3, 1e-2, 2e-2, 4e-2, 0.1, 0.3, 1} (+ {1e-4, 3e-4} for GD); weak lr ∈ {1e-4, 3e-4, 1e-3,
3e-3, 7e-3, 3e-2, 0.1, 0.3}. All selected values are interior to
their grid (strong-GD grid extended down to 1e-4 to confirm). Full tables: `python src/select_lr.py`.

Observations (paper-ready):
- Adagrad is robust: no divergence at any tested lr, a single lr (0.1) is optimal in all four settings.
- Fixed-step GD is fragile in the non-convex, heavy-tailed L63 problem: larger steps do not diverge at first but stop
  early at higher cost (strong/full, lr 0.01: "converged" in 204 evals at J = 18.1, forecast RMSE at 1 TU ×1.8 worse),
  and from lr ≥ 0.03 (weak) 46–98 % of cases diverge. The best GD step is small and exhausts the 1000-iteration budget.
- The χ² consistency check holds at the optimum: median J_min ≈ m/2 (16.5 for 33 full-state obs; 5.5 for 11 x-only obs).

### Methods (paper-ready draft)
The strong-constraint cost is J(x₀) = ½‖x₀ − x₀ᵇ‖²_{B⁻¹} + ½ Σᵢ ‖H M_{0→tᵢ}(x₀) − yᵢ‖²_{R⁻¹} over the 11 observation
times of the 1-TU window; the weak-constraint control adds model-error increments ηᵢ applied to the state at the
observation times t₁…t₁₀, penalized by ½ Σᵢ ‖ηᵢ‖²_{Q⁻¹} (Q = 0.01 I). Gradients are exact (reverse-mode
differentiation through the RK4 model); one gradient evaluation = one forward plus one adjoint integration and is the
unit of computational cost. Two classical optimizers are used, both as E1 baselines and as Meta-B teachers: fixed-step
gradient descent and Adagrad (ReferenceStep ported from bice). All cases of a split are minimized in parallel, but
each case stops individually when its relative cost decrease falls below 10⁻⁶, or after 1000 iterations. Step sizes
were selected on validation data with a rule fixed before the sweep (lowest median final J, ≤ 1 % divergence).

### Results — E1 (physical model M1), 200 cases per split (tmp/results/e1/summary.md)
| split | constraint / obs | method (lr) | median J (χ² ref m/2) | median IC RMSE (bg) | fc RMSE @1 TU (bg) | VPT [TU] (bg) | evals to ≤1 % of J* |
|---|---|---|---|---|---|---|---|
| test | strong / full | Adagrad (0.1) | 16.5 (16.5) | 0.429 (0.895) | 0.542 (3.83) | 2.27 (0.61) | **70** |
| test | strong / full | GD (0.001) | 17.0 | 0.429 | 0.704 | 2.13 | 261 |
| test | weak / full | Adagrad (0.1) | 16.4 | 0.428 | 0.558 | 2.25 | **138** |
| test | weak / full | GD (3e-4) | 17.0 | 0.525 | 0.756 | 2.17 | 617 (54 % reach) |
| test | strong / x-only | Adagrad (0.1) | 5.43 (5.5) | 0.608 | 1.34 | 1.58 | **40** |
| test | strong / x-only | GD (0.001) | 5.47 | 0.611 | 1.36 | 1.46 | 549 |
| test_ood | strong / full | Adagrad (0.1) | 17.1 | 0.494 (0.853) | 0.975 (7.66) | 2.24 (0.74) | **125** |
| test_ood | strong / full | GD (0.001) | 18.3 | 0.517 | 1.94 | 1.72 | 219 |
| test_ood | weak / full | Adagrad (0.1) | 17.0 | 0.518 | 1.11 | 2.18 | **190** |
| test_ood | weak / full | GD (3e-4) | 17.7 | 0.559 | 1.48 | 1.79 | 454 (64 % reach) |
J* = best final J of either method on the same case. VPT threshold = 0.4 × climatological std = 5.62.
Full table (all patterns, val): tmp/results/e1/summary.md; evals-to-target: tmp/results/e1/evals_to_target.csv.
Figures: tmp/figs/m2_e1_forecast_rmse.{png,pdf}, tmp/figs/m2_e1_convergence.{png,pdf}.

### Observations (paper-ready)
1. No divergence in any final run (0/200 in all 24 configurations); median J_min matches the χ² expectation m/2,
   so the analyses are statistically consistent with B and R.
2. 4D-Var reduces the IC error by ~50 % (0.43 vs 0.90) and extends the valid prediction time from 0.6 to ≈ 2.2 TU.
3. Adagrad dominates fixed-step GD: equal or better accuracy at 2–14× fewer gradient evaluations (to 1 % of J*).
   GD's best step is small and stalls (zig-zag in the weak problem), reaching the target in only 20–64 % of the
   weak-constraint cases.
4. The held-out lobe-transition region is harder: background forecast error doubles (7.66 vs 3.83 at 1 TU), Adagrad
   needs ~1.8× more evaluations, and the analysis forecast error at 1 TU roughly doubles (0.98 vs 0.54) — a genuine
   extrapolation test for the meta learners.
5. Weak ≈ strong constraint for M1, as expected in a perfect-model twin (no model error to absorb); the weak
   formulation will matter for the emulator (E3/E4). Adagrad's first weak-constraint step increases J
   (step lr·sign(g) = 0.1 = σ_q on η) before recovering.
6. Cost metric: convergence curves plateau long before the 10⁻⁶ stopping rule fires, so "evaluations to reach
   (1 + tol) J*" is the fair cost measure for the North Star gate. Wall-clock is batch-level (≈ 0.5 s per case per
   1000 iterations on CPU) and not per-case; it will be reported per configuration.

### Cost-gate reference (resolved by pilot, 2026-11-10)
The North Star cost gate is reported against both classical optimizers, and the claim is anchored on the strongest
one, Adagrad: a meta optimizer must reach J ≤ 1.01 J* in ≤ 1/10 of Adagrad's gradient evaluations (≈ 7 vs 70 for
strong/full test; ≈ 14 vs 138 weak/full; ≈ 13 vs 125 strong/full test_ood). GD targets (≈ 26 / 62 / 22) are reported
alongside.
