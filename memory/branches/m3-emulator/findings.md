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
