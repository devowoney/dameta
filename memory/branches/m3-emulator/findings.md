# Branch m3-emulator — findings (paper-ready; fold into memory/findings.md on main after merge)

## Milestone 3 — emulator M2, gradient fidelity G1, 4D-Var through the emulator (E3)

### Emulator (methods, paper-ready draft)
M2 replaces the Lorenz-63 vector field by a multilayer perceptron h_φ (3 → 64 → 64 → 3, SiLU activations, inputs
standardized by the training-state statistics, outputs scaled by the training tendency standard deviation) inside
the same fourth-order Runge–Kutta scheme (Δt = 0.01), so that 4D-Var gradients through M2 come from automatic
differentiation of a smooth map. It is trained on the 1000 D1 training trajectories (601 states each) with a
multi-step rollout loss (standardized MSE averaged over n_roll steps; curriculum n_roll = 1 → 10 over the first
half of training), Adam with cosine decay (10⁻³ → 10⁻⁵, 10 000 steps, batch 512), gradient clipping, and the
best checkpoint selected on the D1 validation trajectories (50-step rollout RMSE). Training takes 7.5 min on CPU.
Deliberate model error is introduced only through the teacher model generating the training trajectories (same
initial conditions): biased parameters and/or a scaled −xz term in dy/dt.

### Clean emulator diagnostics (test / test_ood, medians [IQR]; tmp/results/emulator/clean_diag.json)
| Metric | test | test_ood |
|---|---|---|
| one-step RMSE | 9.1e-4 | 9.6e-4 |
| free-forecast RMSE from true x0 @ 0.5 / 1 / 2 TU | 0.024 / 0.051 / 0.145 | 0.055 / 0.054 / 0.180 |
| valid prediction time (0.4 × clim. std) | 5.1 TU [4.3, 6.0] | 5.0 TU [4.2, 6.0] |
| leading Lyapunov exponent (M1: 0.904 ± 0.012) | 0.927 [0.888, 0.938] | 0.902 [0.875, 0.925] |
| TL fidelity over the window: cos / rel. error | 0.9998 / 0.050 | 0.9999 / 0.029 |
| **G1** strong/full at x₀ᵇ: cos / rel. error | **0.999 / 0.069** | 0.9999 / 0.022 |
| **G1** strong/full at x₀ᵗ (near minimum): cos / rel. error | **0.997 / 0.145** (86 % cases cos > 0.9) | 0.9996 / 0.177 (90 %) |
| **G1** weak/full at (x₀ᵇ, η = 0): cos / rel. error | 0.999 / 0.065 | 0.9998 / 0.030 |
| **G1** strong/x-only at x₀ᵇ: cos / rel. error | 0.999 / 0.066 | 0.9999 / 0.025 |

Observations: the clean emulator is accurate (valid prediction time ≈ 5 TU from the true state, ≫ the 2.2 TU of the
E1 analyses) and slightly over-chaotic (λ₁ +2.5 %). Its 4D-Var gradients are nearly parallel to the exact ones far
from the minimum (cos ≈ 0.999), but near the minimum, where the true gradient is small, the emulator error dominates
(relative error 15–18 %, 10–14 % of cases with cos < 0.9). Emulator-based 4D-Var is therefore expected to converge
to a slightly displaced minimum — the gap a meta learner would have to close.

### Candidate deliberate degradations (teacher model vs M1 truth, test ICs)
| Teacher | λ₁ | RMSE from true x0 @ 0.5 / 1 TU | VPT [TU] |
|---|---|---|---|
| A: ρ = 26 | 0.852 | 1.45 / 3.05 | 1.17 |
| B: xz_scale = 0.9 | 0.890 | 1.84 / 4.25 | 0.94 |
| B′: xz_scale = 0.95 | 0.911 | 0.88 / 2.15 | 1.62 |
| C: ρ = 26 & xz_scale = 0.9 | 0.857 | 1.01 / 2.14 | 1.68 |
All remain chaotic. Within the 1-TU window, B's model error exceeds the observation error (σ_o = 1.41) — a stress
test where strong constraint must fail; B′ is comparable to the observation error. Choice pending with the pilot.

### E3 with the clean emulator vs E1 (test / test_ood, 200 cases each; tmp/results/e3_clean/compare_e1.md)
Learning rates re-selected on val with the pre-registered rule (3-point grids around the E1 values): identical to E1
(Adagrad 0.1; GD 10⁻³ strong, 3·10⁻⁴ weak). E3 analyses are judged with the true model (P1 = J_M1 of the analysis;
forecast RMSE with M1 from the analysis).
| split | constraint / obs | method | P1: J_true E3 (J E1) | IC RMSE E3 (E1) | fc RMSE @1 TU, M1 forecast: E3 (E1) | emulator forecast E3 | paired gap E3−E1 @1 TU | evals to 1 % (E3) |
|---|---|---|---|---|---|---|---|---|
| test | strong / full | Adagrad | 17.0 (16.5) | 0.438 (0.429) | 0.690 (0.542) | 0.654 | 0.035 | 79 |
| test | strong / full | GD | 17.2 (17.0) | 0.452 (0.429) | 0.835 (0.704) | 0.755 | 0.016 | 234 |
| test | weak / full | Adagrad | 16.7 (16.4) | 0.431 (0.428) | 0.685 (0.558) | 0.596 | 0.026 | 138 |
| test | strong / x-only | Adagrad | 5.52 (5.43) | 0.610 (0.608) | 1.34 (1.34) | 1.33 | 0.019 | 41 |
| test_ood | strong / full | Adagrad | 17.3 (17.1) | 0.499 (0.494) | 1.02 (0.975) | 0.880 | 0.0005 | 153 |
| test_ood | weak / full | Adagrad | 17.2 (17.0) | 0.512 (0.518) | 1.09 (1.11) | 1.11 | 0.013 | 198 |
(all 16 configurations: 0 diverged; full table in compare_e1.md; figure tmp/figs/m3_e3_clean_forecast.{png,pdf})

Observations (paper-ready):
1. A well-trained emulator makes emulator-based 4D-Var almost as good as 4D-Var with the exact adjoint: the analysis
   is slightly displaced (P1 cost +1–3 %, IC error +0–5 %), consistent with the G1 degradation near the minimum, and
   the iteration cost is unchanged (79 vs 70 evaluations to 1 % for Adagrad, strong/full).
2. The case-paired forecast gap E3 − E1 is small (median 0.0005–0.035 at 1 TU) while the medians differ more
   (0.69 vs 0.54): the emulator penalty is concentrated in a minority of cases (heavy right tail).
3. Consequence for the North Star: with the clean emulator the gap-closure ratio (E3 − E4)/(E3 − E1) has a near-zero
   denominator and would be dominated by noise. Testing "the meta learner re-learns missing dynamics" requires the
   deliberately degraded emulator.

### Degraded emulators (pilot choice: B′ xz_scale = 0.95 main, B xz_scale = 0.9 stress)
Both trained like the clean emulator (10 000 steps, ≈ 12 min; best val rollout RMSE vs their own teacher 0.030 / 0.034).
Diagnostics vs the TRUE model M1 (test / test_ood medians [IQR]; tmp/results/emulator/<name>_diag.json):
| Metric | xz095 test | xz095 test_ood | xz090 test | xz090 test_ood |
|---|---|---|---|---|
| free-forecast RMSE @1 TU from true x0 | 2.12 | 1.15 | 4.19 | 2.43 |
| VPT [TU] | 1.63 | 2.50 | 0.93 | 1.72 |
| λ₁ | 0.908 | 0.890 | 0.910 | 0.881 |
| TL cos / rel. err (window) | 0.945 / 0.44 | 0.988 / 0.20 | 0.783 / 0.87 | 0.949 / 0.39 |
| **G1** cos at x₀ᵇ (strong/full) | 0.980 [0.82, 0.995] | 0.9995 | **0.909 [−0.36, 0.98]** | 0.998 |
| **G1** rel. err at x₀ᵇ | 0.68 | 0.12 | 1.16 | 0.24 |
| **G1** cos at x₀ᵗ (near minimum) | **0.24 [−0.88, 0.94]** | 0.92 | **−0.13 [−0.88, 0.86]** | 0.71 |
| one-obs-interval model error std (x, y, z) | 0.38, 0.88, 0.46 | | 0.77, 1.80, 0.96 | |
(clean emulator: one-obs-interval error std 0.004 / 0.010 / 0.010)

lr re-selected on val (full obs) with the pre-registered rule; strong-constraint grids extended at both edges to
make the choice interior: Adagrad 0.3 (strong) / 0.1 (weak); GD 3·10⁻⁴ (strong and weak), identical for both
degraded emulators.

### E3 with degraded emulators vs E1 (Adagrad, full obs, test; tmp/results/e3_<name>/compare_e1.md)
| | E1 (M1) | E3 clean | E3 xz095 | E3 xz090 |
|---|---|---|---|---|
| P1: J_true of analysis (χ² ref 16.5) | 16.5 | 17.0 | **31.9** | **72.2** |
| IC RMSE (background 0.895) | 0.429 | 0.438 | 0.771 | 1.17 |
| fc RMSE @1 TU, M1 forecast from analysis (background 3.83) | 0.542 | 0.690 | **3.10** | **5.02** |
| fc RMSE @1 TU, emulator forecast from its analysis | — | 0.654 | 1.36 | 2.47 |
| VPT [TU], M1 forecast (background 0.61) | 2.27 | 2.15 | 0.57 | 0.42 |
| paired gap E3 − E1 @1 TU (gap-closure denominator) | — | 0.035 | **2.09** | **3.77** |
| evals to 1 % of own J* | 70 | 79 | 30 | 46 |
Weak constraint, xz095: fc 3.46, gap 2.10; test_ood xz095 strong: fc 2.20 vs E1 0.975, gap 0.66.

Observations (paper-ready):
1. Missing dynamics destroys gradient fidelity long before forecast skill: with 5 % of the −xz coupling removed, the
   emulator still forecasts for 1.6 TU, but near the minimum its 4D-Var gradient is essentially uncorrelated with the
   true one (median cos 0.24; xz090: −0.13, with > 25 % of cases pointing uphill already at the background).
2. Emulator-based 4D-Var then converges quickly (30–46 evaluations) to a minimum of the WRONG cost: the analysis fits
   the emulator dynamics (emulator forecast 1.36) but not the truth (M1 forecast 3.10, valid prediction time no better
   than the background). P1 doubles (31.9) and quadruples (72.2) relative to the χ² reference.
3. This gives a large, well-defined gap-closure denominator (2.09 / 3.77 at 1 TU on test), so the North Star test
   "meta learner re-learns missing dynamics" is now measurable (E4 vs E3 vs E1).
4. The degradation is region-dependent: the missing term −xz in dy/dt scales with x, which is small near the z-axis,
   so test_ood (lobe-transition region) is LESS affected by this model error (G1 cos 0.9995, gap 0.66) even though it
   is dynamically harder. To be stated as a confound when interpreting OOD results for E3/E4.
5. Weak constraint does not help with the degraded emulators because Q is mis-specified: σ_q = 0.1 is 4–9× (xz095)
   and 8–18× (xz090) smaller than the actual one-interval model error (but 10× larger than the clean emulator's).
   Decision for the pilot: keep Q fixed (as approved) or re-estimate Q per emulator from train/val data.

---
## Consolidated report — 4D-Var initial-condition estimation (E1, E3) [paper-ready, 2026-11-11]
Reproduce: `python src/report_ic.py` → tmp/report/table{1,2,3}_*.{md,csv}, tmp/report/fig{1,2,3}_*.{png,pdf}.
This section supersedes the per-run E1/E3 numbers above where they differ (single cost definition, bootstrap CIs).

### Evaluation protocol
Each split (test: 200 in-distribution cases; test_ood: 200 cases from the held-out lobe-transition region
r_xy < 3) is a set of independent twin experiments sharing the same truth, observations and background across all
configurations, so every comparison is paired by case. The analysis x₀ᵃ is judged against the truth by (i) the
initial-condition RMSE, (ii) the RMSE of the forecast obtained by integrating x₀ᵃ with the true model M1 over 5 time
units (TU) beyond the 1-TU window, (iii) the valid prediction time (VPT), the first lead at which that forecast RMSE
exceeds 0.4 × the climatological standard deviation (5.62), and (iv) the 4D-Var cost of x₀ᵃ evaluated with M1 (P1).
Cost is the number of gradient evaluations (one forward + one adjoint integration) needed to reach
J ≤ 1.01 J*, where J* is the best final cost reached by either classical optimizer on the same case with the same
dynamics. Summaries are medians over cases with 95 % percentile-bootstrap confidence intervals (2000 resamples);
E3–E1 differences are paired medians with bootstrap CIs and two-sided Wilcoxon signed-rank tests. Emulator-based
analyses are always evaluated with M1, so that differences reflect the quality of the estimated initial condition.

### Table 1 — Classical 4D-Var with the exact model (E1), median [95 % CI], 200 cases per split
| split | constraint / obs | optimizer (lr) | IC RMSE | fc RMSE @1 TU | VPT [TU] | J (m/2) | evals to 1 % of J* (reach) |
|---|---|---|---|---|---|---|---|
| test | background (no DA) | — | 0.895 [0.836, 0.949] | 3.83 [3.34, 4.56] | 0.61 [0.50, 0.90] | — | — |
| test | strong / full | Adagrad (0.1) | 0.429 [0.385, 0.483] | 0.54 [0.43, 0.78] | 2.27 [2.06, 2.68] | 16.5 (16.5) | **70** [56, 88] (92 %) |
| test | strong / full | GD (10⁻³) | 0.429 [0.382, 0.522] | 0.70 [0.53, 0.85] | 2.13 [1.94, 2.32] | 17.0 | 261 [224, 283] (92 %) |
| test | weak / full | Adagrad (0.1) | 0.428 [0.379, 0.492] | 0.56 [0.42, 0.70] | 2.25 [2.07, 2.64] | 16.4 | **138** [95, 171] (99 %) |
| test | weak / full | GD (3·10⁻⁴) | 0.525 [0.478, 0.556] | 0.76 [0.56, 0.94] | 2.17 [2.01, 2.37] | 17.0 | 617 [526, 682] (54 %) |
| test | strong / x-only | Adagrad (0.1) | 0.608 [0.567, 0.661] | 1.34 [1.11, 1.66] | 1.58 [1.37, 1.82] | 5.43 (5.5) | **40** [30, 51] (98 %) |
| test | strong / x-only | GD (10⁻³) | 0.611 [0.568, 0.680] | 1.36 [1.03, 1.57] | 1.46 [1.33, 1.78] | 5.47 | 549 [481, 580] (90 %) |
| test_ood | background (no DA) | — | 0.853 [0.814, 0.904] | 7.66 [5.06, 10.28] | 0.74 [0.66, 0.81] | — | — |
| test_ood | strong / full | Adagrad (0.1) | 0.494 [0.445, 0.569] | 0.98 [0.68, 1.37] | 2.24 [2.00, 2.39] | 17.2 | **125** [101, 184] (84 %) |
| test_ood | strong / full | GD (10⁻³) | 0.517 [0.444, 0.605] | 1.94 [1.15, 2.98] | 1.72 [1.57, 2.06] | 18.3 | 219 [186, 278] (82 %) |
| test_ood | weak / full | Adagrad (0.1) | 0.518 [0.452, 0.568] | 1.11 [0.77, 1.75] | 2.18 [1.85, 2.37] | 17.0 | **190** [148, 234] (94 %) |
| test_ood | weak / full | GD (3·10⁻⁴) | 0.559 [0.489, 0.624] | 1.48 [1.02, 2.40] | 1.79 [1.66, 2.17] | 17.7 | 454 [318, 563] (64 %) |
All 24 E1 runs: 0 / 200 diverged. Full table incl. x-only and weak/x-only rows: tmp/report/table1_e1_classical.md.

### Table 2 — 4D-Var through the emulator (E3) vs E1, Adagrad, paired by case
| emulator | split | constraint / obs | IC RMSE E3 (E1) | Δ fc @1 TU, E3 − E1 [95 % CI] | p (Wilcoxon) | E3 worse | VPT E3 (E1) | P1: J_M1 E3 / E1 |
|---|---|---|---|---|---|---|---|---|
| clean | test | strong / full | 0.438 (0.429) | 0.035 [0.010, 0.065] | 3·10⁻³ | 59 % | 2.15 (2.27) | 1.00 |
| clean | test | weak / full | 0.431 (0.428) | 0.026 [0.004, 0.050] | 1·10⁻² | 58 % | 2.12 (2.25) | 1.00 |
| clean | test_ood | strong / full | 0.499 (0.494) | 0.000 [−0.008, 0.011] | 0.50 | 50 % | 2.31 (2.24) | 1.00 |
| **xz095** | test | strong / full | 0.771 (0.429) | **2.09** [1.73, 2.79] | 1·10⁻²⁴ | 89 % | 0.57 (2.27) | 1.75 |
| xz095 | test | weak / full | 0.722 (0.428) | 2.10 [1.85, 2.78] | 1·10⁻²³ | 87 % | 0.56 (2.25) | 1.73 |
| xz095 | test | strong / x-only | 0.792 (0.608) | 1.56 [1.08, 1.82] | 3·10⁻¹³ | 74 % | 0.68 (1.58) | 1.31 |
| xz095 | test_ood | strong / full | 0.679 (0.494) | 0.66 [0.45, 0.94] | 3·10⁻¹² | 76 % | 1.40 (2.24) | 1.17 |
| xz095 | test_ood | weak / full | 0.637 (0.518) | 0.76 [0.59, 1.13] | 2·10⁻⁹ | 73 % | 1.34 (2.18) | 1.19 |
| **xz090** | test | strong / full | 1.172 (0.429) | **3.77** [3.32, 4.40] | 7·10⁻²⁸ | 94 % | 0.42 (2.27) | 4.30 |
| xz090 | test | weak / full | 1.114 (0.428) | 3.85 [3.17, 4.50] | 3·10⁻²⁶ | 92 % | 0.41 (2.25) | 3.98 |
| xz090 | test_ood | strong / full | 1.107 (0.494) | 2.09 [1.42, 2.80] | 9·10⁻¹⁹ | 86 % | 0.81 (2.24) | 1.79 |
All 40 E3 runs: 0 / 200 diverged. Full table (20 rows): tmp/report/table2_e3_vs_e1_paired.md.

### Table 3 — Emulator fidelity (test; test_ood in brackets)
| emulator (teacher xz_scale) | free-fc VPT from x₀ᵗ [TU] | λ₁ (M1 0.904) | TL cos (window) | G1 cos @ x₀ᵇ | G1 cos @ x₀ᵗ (cases < 0) | model error per obs interval (x, y, z) |
|---|---|---|---|---|---|---|
| clean (1.00) | 5.14 (4.98) | 0.927 | 1.000 | 0.999 (1.000) | 0.997 (6 %) | 0.004, 0.010, 0.010 |
| xz095 (0.95) | 1.62 (2.50) | 0.908 | 0.945 | 0.980 (0.999) | **0.24** (47 %) [OOD 0.92] | 0.38, 0.88, 0.46 |
| xz090 (0.90) | 0.93 (1.71) | 0.910 | 0.783 | 0.908 (0.998) | **−0.13** (50 %) [OOD 0.71] | 0.77, 1.80, 0.96 |
Weak-constraint Q is kept at σ_q = 0.1 (pilot decision) — 4–18× below the degraded emulators' model error.

### Figures (tmp/report/)
- **Fig. 1** `fig1_ic_estimation_overview` — Strong-constraint 4D-Var with full observations and Adagrad, 200
  cases per split (rows: test, test_ood). (a, d) Distribution of the initial-condition RMSE for the background, E1
  and E3 with the three emulators. (b, e) Median RMSE of the forecast integrated with the true model from each
  analysis, versus lead time beyond the window (bands: interquartile range for E1 and E3-xz095; dashed: VPT
  threshold). (c, f) Distribution of the valid prediction time. Boxes: interquartile range; whiskers: 1.5 × IQR.
- **Fig. 2** `fig2_e1_optimizer_cost` — Cost of classical 4D-Var with the exact model (E1). (a, b) Median
  relative excess cost J/J* − 1 versus gradient evaluations (band: IQR) for Adagrad and gradient descent, strong and
  weak constraint, test split. (c) Gradient evaluations to reach 1 % of J* (median, 95 % bootstrap CI) for all
  eight configurations (T = test, OOD = test_ood; st/we = strong/weak; full/x = observed variables).
- **Fig. 3** `fig3_e3_gradient_fidelity` — Gradient fidelity of the emulators (G1, strong constraint, full
  observations). (a, b) Cosine between the 4D-Var gradient computed through the emulator and the exact gradient,
  at the background and at the true initial condition (test). (c) Median relative gradient error at the true
  initial condition versus the median paired forecast penalty E3 − E1 at 1 TU (95 % bootstrap CIs; filled = test,
  open = test_ood).

### Results (paper-ready text)
**Classical baseline.** With the exact model, strong-constraint 4D-Var halves the initial-condition error of the
background (0.43 vs 0.90, test) and extends the valid prediction time from 0.6 to 2.3 TU; the median final cost
matches the χ² expectation m/2 in every configuration, and no case diverges (Table 1, Fig. 1). Adagrad reaches
1 % of the attainable cost in 70 gradient evaluations against 261 for the best fixed-step gradient descent, and
the advantage grows for the weak-constraint and partially observed problems (up to 14×), where gradient descent
fails to reach the target in 46–80 % of the cases (Fig. 2). The held-out lobe-transition region is harder for the
optimizer (125 vs 70 evaluations) and for the forecast (RMSE at 1 TU 0.98 vs 0.54), while the initial-condition
error increases only moderately (0.49 vs 0.43).

**4D-Var through an emulator.** A well-trained emulator (clean) is nearly a drop-in replacement for the exact model:
the paired forecast penalty is 0.035 [0.010, 0.065] at 1 TU on test and indistinguishable from zero on test_ood
(p = 0.50), the cost of the analysis evaluated with the true model is unchanged (ratio 1.00), and the optimizer
needs the same number of iterations (Table 2). When 5 % of the −xz coupling is missing from the emulator's training
dynamics (xz095), the emulator still forecasts for 1.6 TU from the true state, but the analysis degrades sharply:
the initial-condition error rises from 0.43 to 0.77, the forecast penalty is 2.09 [1.73, 2.79] at 1 TU (89 % of
cases worse, p ≈ 10⁻²⁴), and the valid prediction time collapses to the level of the background (0.57 TU). The
stress emulator (xz090) roughly doubles these effects (penalty 3.77, P1 cost ×4.3). The penalty is systematic in
strong and weak constraint alike: with σ_q fixed at 0.1, the weak-constraint control cannot absorb a model error
4–18 times larger.

**Mechanism.** The degraded emulators' gradients are accurate far from the minimum (cosine 0.98 for xz095 at the
background) but lose their direction close to it (median cosine 0.24; 47 % of cases point uphill at the true
initial condition), so emulator-based 4D-Var converges quickly to the minimum of a different cost (Table 3,
Fig. 3a–b). Across emulators and splits, the median analysis penalty increases monotonically with the median
relative gradient error at the true initial condition (Fig. 3c). At the level of individual cases, gradient error
is only a weak predictor of the penalty (Spearman ρ ≤ 0.38 for the degraded emulators, ≤ 0.23 for the clean one),
indicating that the penalty reflects the displaced minimum of the emulator cost rather than local gradient noise.

**Out-of-distribution region.** The missing term −xz in dy/dt scales with x, which is small near the z-axis. The
held-out region is therefore less affected by this particular model error (xz095 gap 0.66 vs 2.09; G1 cosine 0.92
vs 0.24 at the true initial condition) even though it is dynamically harder. OOD comparisons of E3/E4 must be read
with this confound in mind.

### Implications for Milestones 4–5
- Cost targets for the meta optimizers (≤ 1/10 of Adagrad, claim): ≈ 7 (strong/full), 14 (weak/full), 4
  (strong/x-only) evaluations on test; 13 (strong/full) and 19 (weak/full) on test_ood.
- Gap-closure denominators (median paired gap at 1 TU, strong/full): xz095 2.09 (test), 0.66 (test_ood);
  xz090 3.77 (test), 2.09 (test_ood). The clean emulator (0.035) is a control, not a gap-closure case.
- Limitations: one dataset seed and one emulator seed per configuration; learning rates tuned on 50 validation
  cases; forecast RMSE is averaged over the three variables; σ_q deliberately not re-estimated.
