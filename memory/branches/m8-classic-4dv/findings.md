# Findings — branch m8-classic-4dv

## Background- and model-error covariances of the 4D-Var experiments (2026-11-26)

### 1. Assimilation problem
Each estimation model m ∈ {F1, F2, E1, E2} (resolvent M_m over one step Δt = 0.01; F = physical L63 with misspecified
parameters, E = MLP emulator trained on F trajectories; the truth L1 never enters the assimilation) solves, for every
case, an initial-condition (IC) estimation problem over a window of L = 160 steps (1.6 TU) with observations every
K = 20 steps, i.e. at t_i = iK, i = 0, …, N (N = 8, 9 observation times):

    y_i = H x_true(t_i) + ε_i,      ε_i ~ N(0, R),   R = σ_o² I,   σ_o² = 4,
    H = I_3 (full) or H = [1 0 0] (x only).

Strong constraint (control x_0 ∈ ℝ³):

    J_s(x_0) = ½ (x_0 − x_b)ᵀ B̃_m⁻¹ (x_0 − x_b) + ½ Σ_{i=0}^{N} (H x_i − y_i)ᵀ R⁻¹ (H x_i − y_i),
    x_{k+1} = M_m(x_k).

Weak constraint (control (x_0, η_1, …, η_N) ∈ ℝ^{3(N+1)} = ℝ²⁷): a model-error increment η_i is added to the state at
each observation time t_i (i ≥ 1) after the K-step propagation, x_{t_i} = M_m^K(x_{t_{i−1}}) + η_i, and

    J_w(x_0, η) = J_s-terms on that trajectory + ½ Σ_{i=1}^{N} η_iᵀ Q_m⁻¹ η_i.

The background term uses an inflated covariance B̃_m = λ B_m with λ = 4 (deliberately weakened background so that
the model-dependent observation term shapes the cost geometry). First guess x_b: the 50-step (0.5 TU) forecast of
model m valid at t_0 − 5 steps and used as the state at t_0 (forecast error of the model plus a timing error).

### 2. Background-error covariance B_m (static, truth-free)
B_m is one fixed 3×3 matrix per estimation model, identical for every case and window (no flow dependence). It is
estimated from the innovations at t_0 on the training set (n = 1000 cases), which requires only observations and
first guesses:

    d = y_0 − x_b,        E[d dᵀ] = B + R   (background and observation errors mutually independent),
    B_m = (1/n) Σ_c d_c d_cᵀ − R.

The second moment (not the centred covariance) is used, so a systematic first-guess bias is counted as background
error (same convention as Q_m). Positive definiteness is enforced by flooring the eigenvalues at 1 % of tr(B_m)/3;
the floor was not active for any model. Full-state observations at t_0 are used for the estimation also when the
experiment assimilates x only (B_m is a property of the model's first guess, estimated once offline).

| model | B_m (rows x, y, z) | √diag (x, y, z) | corr (xy, xz, yz) | eigenvalues | mean d (x, y, z) |
|---|---|---|---|---|---|
| F1 | [3.22 4.09 −0.26; 4.09 9.98 −0.96; −0.26 −0.96 9.49] | 1.79, 3.16, 3.08 | 0.72, −0.05, −0.10 | 1.29, 9.15, 12.25 | −0.12, −0.19, −0.12 |
| F2 | [19.90 28.92 −1.60; 28.92 49.65 −1.31; −1.60 −1.31 51.54] | 4.46, 7.05, 7.18 | 0.92, −0.05, −0.03 | 2.24, 51.31, 67.54 | 0.06, 0.14, −3.91 |
| E1 | [3.26 3.84 −0.11; 3.84 9.76 −0.66; −0.11 −0.66 11.07] | 1.80, 3.12, 3.33 | 0.68, −0.02, −0.06 | 1.47, 10.62, 11.99 | −0.04, −0.10, −0.28 |
| E2 | [19.66 27.67 −2.16; 27.67 47.58 −1.02; −2.16 −1.02 54.12] | 4.43, 6.90, 7.36 | 0.90, −0.07, −0.02 | 2.59, 53.78, 64.99 | 0.12, 0.18, −3.97 |

Consistency check (uses the truth; diagnostic only, never used in the estimation): χ² = eᵀ B_m⁻¹ e with
e = x_b − x_true(t_0), expected mean 3 for a calibrated B.

| model | χ² train | χ² val | χ² test | χ² test_ood |
|---|---|---|---|---|
| F1 | 3.16 | 3.21 | 3.43 | 1.33 |
| F2 | 3.05 | 3.09 | 2.97 | 1.29 |
| E1 | 3.16 | 4.16 | 3.77 | 1.48 |
| E2 | 3.04 | 3.13 | 3.18 | 1.43 |

Properties: (i) B_m is ≈ 6× larger in variance for the strongly misspecified models (F2, E2) than for the mild ones
(F1, E1), and the emulators inherit the B of their teacher; (ii) x–y errors are strongly correlated (0.7 for F1/E1,
0.9 for F2/E2), z is nearly uncorrelated with x and y; (iii) for F2/E2 the z first guess is biased (mean innovation
≈ −3.9), which the second-moment estimate absorbs into the z variance; (iv) B_m is calibrated in distribution
(χ² 3.0–3.8) and conservative on the held-out region (test_ood, χ² 1.3–1.5).

Replaced specification (for the record): B_h = α B_ens + (1 − α) s² I with B_ens the case-specific covariance of 10
time-lagged forecasts started from true past states and (α, s²) fitted by maximum likelihood of the t_0 innovations.
It was flow-dependent, used the truth history through the ensemble initial states, B_ens was nearly rank one
(≈ 95 % of its variance in one direction, χ² ≈ 7–10), and (α, s²) were unstable (grid-edge solutions; near-tied
likelihood optima for F2/E2 that changed with the observation-noise realisation).

### 3. Model-error covariance Q_m (per-model, diagonal)
Q_m is the covariance of the error accumulated by model m over one observation interval (K = 20 steps) when started
from the true state, estimated on the training set over all intervals of the window (n = 1000 cases × 8 intervals =
8000 samples):

    e_{c,i} = M_m^K( x_true(t_{i−1}) ) − x_true(t_i),
    Q_m = diag( E[e_x²], E[e_y²], E[e_z²] )        (second moment, bias included).

Q_m therefore matches the actual per-interval model-error magnitude of each model and each variable (an oracle
specification, standard in twin experiments; it uses the truth, whereas B_m does not). Off-diagonal terms are
neglected.

| model | Q_m diag (x, y, z) | √Q_m (x, y, z) | mean e (x, y, z) | std e (x, y, z) |
|---|---|---|---|---|
| F1 | 0.083, 0.182, 0.244 | 0.29, 0.43, 0.49 | −0.00, −0.01, 0.34 | 0.29, 0.43, 0.36 |
| F2 | 3.325, 6.996, 10.399 | 1.82, 2.65, 3.23 | −0.03, −0.03, 2.34 | 1.82, 2.65, 2.22 |
| E1 | 0.140, 0.346, 0.405 | 0.37, 0.59, 0.64 | −0.00, −0.01, 0.33 | 0.37, 0.59, 0.55 |
| E2 | 3.279, 7.195, 9.760 | 1.81, 2.68, 3.12 | −0.04, −0.09, 2.23 | 1.81, 2.68, 2.19 |

Properties: (i) the model error of the strongly misspecified models (F2, E2) is ≈ 6× larger in standard deviation
than that of the mild ones (F1, E1), and each emulator's error is close to its teacher's; (ii) the z component carries
a systematic drift (+2.2 to +2.3 per interval for F2/E2, +0.3 for F1/E1) that the second-moment Q includes; (iii)
relative to the observation noise (σ_o = 2), the per-interval model error is ≈ 0.15–0.3 σ_o for F1/E1 and
≈ 0.9–1.6 σ_o for F2/E2.

Replaced specification (for the record): Q = σ_q² I with σ_q = 0.1 for every model. On the 1.6-TU / K = 20 set-up
it was 4–30× smaller than the actual model error, the analysed increments stayed at |η| ≈ 0.01 (≈ 100× below the
model error) and the weak-constraint analysis coincided with the strong-constraint one.

### 4. Implementation and reproducibility
- B_m: src/estimate_b.py → tmp/data/static_B_seed42.json; loaded by `load_static_background`
  (src/dameta/data/background.py); runner flag `background_cov: static` (src/configs/fourdvar_v2.yaml), inflation
  `cost.b_infl: 4`.
- Q_m: src/estimate_q.py → tmp/data/model_q_seed42.json; `CostConfig.q_var`, `FourDVarCost.jq`
  (src/dameta/da/cost.py); runner flag `cost.q: model`.
- Data: D1 v3 (tmp/data; window 160, obs every 20, σ_o² = 4, first-guess time shift 5 steps), seed 42, train split
  for both estimations; result files record `background_cov`, `B_info`, `b_infl`, `q_var`.
