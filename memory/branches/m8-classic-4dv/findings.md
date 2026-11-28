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

---

## M8-R — Isolated impacts of model misspecification and model approximation on strong- and weak-constraint 4D-Var (experiment definition, 2026-11-27)

**Status.** The experiment is defined, implemented and checked for feasibility on a 10-case validation smoke run:
120 / 120 runs completed, with one diverged case (F1, weak, x-only, GD, truth L1). **The 200-case test has not been
run, so no result is reported in this section.** The M8 configuration above (each model with its own first guess and
B, oracle Q) is kept as the control experiment.

### 1. Objective and design

The aim is to measure, separately, how two kinds of dynamical-model error change (a) the geometry of the 4D-Var cost
function and (b) the quality of its solution, under strong and weak constraint.

- **Misspecification (L1 → F1, F2).** The model is physically consistent but has wrong parameters. Its trajectory is
  biased, but its tangent-linear model and adjoint are exact for the model actually used, so the error mainly moves
  the minimum of J.
- **Approximation (F1 → E1, F2 → E2).** A learned emulator replaces its physical teacher. It can follow the teacher's
  trajectory closely while having a different Jacobian, so it can also distort the gradient and the curvature of J.

**Model rectangle** (src/configs/models.yaml):

| model | definition | parameters / architecture |
|---|---|---|
| L1 | Lorenz-63, RK4, Δt = 0.01 | θ_L = (σ, ρ, β) = (10, 28, 8/3) |
| F1 | L63 with θ = θ_L ⊙ (1 + ε₁ d), d = (1, 1, 1)/√3 | ε₁ = 0.03 → θ = (10.173, 28.485, 2.713) |
| F2 | same, ε₂ = 0.18 | θ = (11.039, 30.910, 2.944) |
| E1 / E2 | MLP tendency + RK4, trained on F1 / F2 trajectories | 10 training initial conditions, 1 hidden layer × 8 (SiLU) |

Model error over one observation interval, relative to σ_o (m7 screen): L1→F1 0.155, L1→F2 0.958, F1→E1 0.142,
F2→E2 0.126. The mild physical step (L1→F1) and the emulator steps are therefore of comparable size. This allows a
direct comparison of a physics error and an emulator error of the same magnitude.

**Twin experiments.** In every comparison, the imperfect model and a perfect-model reference assimilate the same
observations of the same truth, start from the same first guess and use the same B. Only the dynamics inside the cost
differ. The degradation is the paired per-case difference Δ = metric(model) − metric(reference).

| stage | truth = observations | shared first guess / B | strong | weak (Q = q_m I) | question |
|---|---|---|---|---|---|
| A1 misspecification | L1 | L1 lagged forecast | **L1 (ref.)**, F1, F2 | F1, F2 | cost of wrong physics |
| A2 approximation, isolated | F1 | F1 lagged forecast | **F1 (ref.)**, E1 | E1 | cost of the emulator alone |
| A2 approximation, isolated | F2 | F2 lagged forecast | **F2 (ref.)**, E2 | E2 | same, with a strongly misspecified teacher |
| B approximation, realistic | L1 | L1 lagged forecast | E1, E2 (vs F1, F2, L1) | E1, E2 | emulator cost on real data |

- **Stage A2.** The teacher is the truth, so the emulator error is isolated, and the data lie in the emulator's
  training distribution.
- **Stage B (real-life setting).** The emulator error adds to the teacher's misspecification, and the data (L1
  trajectories) lie mildly outside the emulator's training distribution. Comparing the approximation increment
  E_k − F_k in stage B (L1 observations) with the same increment in stage A2 (F_k observations) measures how the two
  errors interact.
- **Reference.** The perfect-model reference runs with strong constraint only: with a perfect model Q = 0, and the
  weak problem reduces to the strong one. Weak-constraint runs of the imperfect models are compared with this strong
  reference.

### 2. Assimilation problem

Window L = 160 steps (1.6 TU). Observations every K = 20 steps (t_i = iK, i = 0, …, 8, so nine observation times).
Forecast of 500 steps (5 TU) beyond the window.

    y_i = H x_true(t_i) + ε_i,   ε_i ~ N(0, R),   R = σ_o² I,   σ_o² = 4,   H = I_3 (full) or [1 0 0] (x only).

Strong constraint (control x_0 ∈ ℝ³), with x_{k+1} = M_m(x_k):

    J_s(x_0) = ½ (x_0 − x_b)ᵀ (λB)⁻¹ (x_0 − x_b) + ½ Σ_{i=0}^{8} (H x_{t_i} − y_i)ᵀ R⁻¹ (H x_{t_i} − y_i).

Weak constraint (control (x_0, η_1, …, η_8) ∈ ℝ²⁷), with x_{t_i} = M_m^K(x_{t_{i−1}}) + η_i:

    J_w = the J_s terms on that trajectory + ½ Σ_{i=1}^{8} η_iᵀ Q_m⁻¹ η_i.

λ = 4 (b_infl): the background term is deliberately weakened, so that the model-dependent observation term shapes the
landscape.

### 3. First guess x_b (shared by all models of a truth)

x_b is the truth model's time-lagged forecast at lead 0.5 TU (50 steps). It starts from the true state, is valid at
t_0 − 5 steps, and is used as the state at t_0. Because the forecasting model is the truth model itself,

    x_b = x_true(t_0 − 5 Δt)      (verified: max |x_b − x_true(t_0 − 5)| = 0 on every split).

The first-guess error is therefore a pure timing error, independent of the estimation model. Median
|x_b − x_true(t_0)| on the test split: 2.77 (truth L1), 2.92 (F1), 3.26 (F2). All models of a truth (reference, F and
E) start from this x_b.

### 4. Background-error covariance B (static, truth-free, shared)

B is one fixed 3×3 matrix per truth, the same for every case and every estimation model. It is estimated from the
innovations at t_0 on the training set (n = 1000 cases):

    d = y_0 − x_b,   E[d dᵀ] = B + R   ⇒   B = (1/n) Σ_c d_c d_cᵀ − R.

- **Second moment:** a first-guess bias counts as background error.
- **Positive definiteness:** eigenvalues are floored at 1 % of tr B / 3; the floor was not active.
- **Inputs:** only observations and first guesses are used.

Calibration check (diagnostic only, uses the truth): χ² = eᵀ B⁻¹ e, with e = x_b − x_true(t_0); the expected mean is 3.

| truth (x_b) | B (rows x, y, z) | √diag (x, y, z) | corr (xy, xz, yz) | eigenvalues | mean d | χ² train / val / test / test_ood |
|---|---|---|---|---|---|---|
| L1 | [4.55 5.40 −0.38; 5.40 14.09 −1.49; −0.38 −1.49 14.79] | 2.13, 3.75, 3.85 | 0.67, −0.05, −0.10 | 2.11, 13.92, 17.40 | −0.14, −0.26, 0.71 | 3.10 / 3.09 / 3.30 / 0.96 |
| F1 | [5.65 6.38 0.05; 6.38 13.15 0.45; 0.05 0.45 16.11] | 2.38, 3.63, 4.01 | 0.74, 0.01, 0.03 | 2.00, 15.92, 16.99 | −0.04, 0.07, 0.50 | 3.16 / 3.56 / 3.71 / 0.94 |
| F2 | [7.42 8.33 −0.53; 8.33 18.46 1.11; −0.53 1.11 24.84] | 2.72, 4.30, 4.98 | 0.71, −0.04, 0.05 | 2.90, 22.70, 25.13 | −0.03, 0.15, 1.09 | 3.14 / 2.98 / 3.39 / 0.85 |

B is calibrated in distribution (χ² 3.0–3.7) and conservative in the held-out region (χ² ≈ 0.9). The x and y errors
are correlated (≈ 0.7), as expected for a timing error along the attractor. F2's larger attractor (ρ = 30.9) gives a
larger B. The cost uses λB = 4B.

### 5. Model-error covariance Q (truth-free, χ²-tuned scalar per model)

Q_m = q_m I, with q_m chosen per (truth, model) from the consistency of the weak-constraint minimum. For a consistent
linear-Gaussian system, E[2 J_min] = p, the number of scalar observations (full: p = 9 × 3 = 27).

**Procedure.** The tuning uses 50 training cases, full observations, weak-constraint L-BFGS, and the shared x_b and
λB above. The q grid spans 10^[−3, 2] in steps of 0.25 decade, extended to 10³ for the L1-truth models. For each q:

    r(q) = median over cases of 2 J_min(q) / p,     q_m = argmin_q |log r(q)|.

**Rules, fixed before the choice:**
- The **median** is used, not the mean. On the L1-truth problems, a minority of cases end in local minima with
  2 J_min / p ≫ 1, which made the mean non-monotone in q (spread 2–4 × the mean).
- **Plateau rule:** if |r(q_min) − 1| ≤ 5 % already at the smallest q, the χ² criterion cannot detect the model error.
  q_m is then the largest q with r(q) ≥ 0.95, i.e. the most model error the data tolerate.

The same q_m is used for x-only observations. No truth enters the choice.

| truth | model | q_m | √q_m | median r(q_m) | mean ± std r(q_m) | r at q = 10⁻³ | rule |
|---|---|---|---|---|---|---|---|
| L1 | F1 | 0.1 | 0.32 | 0.97 | 1.65 ± 3.46 | 1.00 | plateau |
| L1 | F2 | 3.16 | 1.78 | 1.06 | 4.15 ± 6.13 | 3.53 | crossing |
| L1 | E1 | 0.056 | 0.24 | 1.00 | 1.71 ± 3.50 | 1.07 | crossing |
| L1 | E2 | 3.16 | 1.78 | 1.06 | 3.84 ± 6.58 | 3.27 | crossing |
| F1 | E1 | 0.18 | 0.42 | 1.02 | 1.38 ± 2.70 | 1.07 | crossing |
| F2 | E2 | 0.032 | 0.18 | 0.95 | 1.04 ± 0.54 | 0.99 | plateau |

Properties of the tuning:
1. **Strong misspecification is clearly detected.** Without Q, F2 and E2 on L1 data are inconsistent (median
   2 J_min / p ≈ 3.3–3.5 at q → 0). They get q ≈ 3.2, i.e. √q ≈ 1.8 ≈ 0.9 σ_o, the order of their one-interval model
   error.
2. **Mild errors are barely visible.** For F1 and E1 on L1, E1 on F1, and E2 on F2, the median ratio is already within
   7 % of 1 with no Q. At this observation noise the χ² test hardly sees the model error, and q_m only marks how much
   model error the data tolerate.
3. **Local minima dominate a minority of cases.** The mean exceeds the median throughout the L1-truth problems; those
   cases are governed by local minima, not by Gaussian misfit.

The oracle diagonal Q of the M8 control (one-interval errors started from the truth) is replaced.

### 6. Optimizers

GD, Adagrad, Adam, and L-BFGS (strong-Wolfe line search, lr 1). The learning rates are those selected for the M8
control (tmp/results/m8/selected_lr.json, per model × scheme × observation pattern × optimizer), with no new tuning.
The references reuse a model's rates: L1 uses those of F1, and each F_k uses its own. max_iter 1000, tolerance 1e-6.

### 7. Diagnostics

**Geometry.** The exact Hessian of J with respect to the full control is computed by double backward, at every
iterate and at the analysis (src/dameta/eval/landscape.py). From it:
- λ_min(H), λ_max(H), the number of negative eigenvalues, and κ_H = λ_max / λ_min (only when H is positive definite);
- the share of iterates with a non-positive-definite Hessian, i.e. the non-convex regions crossed by the optimizer;
- cos(x_true(0) − x_0^k, −∇_{x_0} J): whether the descent direction points toward the truth;
- J on the coordinate planes x–y, x–z and y–z through each model's L-BFGS analysis, for one shared case (weak: η fixed
  at the analysed η_a).

**Solution.**
- |x_a(0) − x_true(0)| (sol_deg);
- mean |x_a(t) − x_true(t)| over the window plus the 5-TU forecast, run with the estimation model (traj_deg);
- forecast RMSE at 1 TU;
- gradient evaluations to convergence, and the number of diverged cases.

**Degradation** (paired per case, mean ± std over cases):
- misspecification: F_k − L1 (stage A1);
- approximation, isolated: E_k − F_k on F_k data (A2);
- approximation, realistic: E_k − F_k on L1 data, and the total E_k − L1 (B);
- interaction: [E_k − F_k]_B − [E_k − F_k]_A2. Stages A2 and B have different truths, so this is a difference of
  distributions, not a paired difference.

### 8. Run matrix and reproducibility

There are 120 runs, each × {full, x-only} × {GD, Adagrad, Adam, L-BFGS}:
- truth L1: L1 strong; F1, F2, E1 and E2 strong and weak;
- truth F1: F1 strong; E1 strong and weak;
- truth F2: F2 strong; E2 strong and weak.

The test split has 200 cases (smoke: 10 validation cases).

- **Data:** src/gen_data_v2.py, with `truth_model=f1|f2` (out_dir tmp/data_tF1 | tmp/data_tF2), and
  `backgrounds_only=true background.registry_models=[l1]` on tmp/data. Commands are in
  src/scripts/lists/m8r_gendata.txt; seed 42.
- **B:** src/estimate_b.py `--models l1 | f1 | f2` → <data>/static_B_seed42.json.
- **Q:** src/scripts/m8_lists.py `qtune` and `qtune_extra` → src/select_q.py → <data>/q_chi2_seed42.json and
  tmp/results/m8r/qtune.md.
- **Runs:** m8_lists `rect_smoke` / `rect`, i.e. run_4dvar_v2 with `data_dir`, `background=<truth model>`,
  `background_cov=static`, `cost.q=null cost.sigma_q=√q_m` and `landscape_iter=true` →
  tmp/results/m8r/<smoke|test>/<truth>/<model>/.
- **Report and figures:** src/report_rect_4dv.py and src/fig_rect_4dv.py → tmp/results/m8r/rect_<sub>.{md,csv},
  rect_<sub>_delta.csv, and tmp/figs/m8r_{land,geom}_*.

## Model rectangle — quantitative model quality and the bad emulator (2026-11-28)

### 1. Purpose
The rectangle varies two kinds of model error: physical misspecification (L1 → F1 → F2: wrong L63 parameters) and
emulator approximation (F → E: an MLP surrogate of F). To compare both axes on one scale, every model's error is
measured against a reference model in units of the observation-error standard deviation σ_o. Two numbers are used: a
local one (one observation interval) and a window one (the whole assimilation window).

### 2. Definitions
Notation: M = the model being assessed, M_ref = its reference, M^n = n steps of M (RK4, Δt = 0.01). Reference:
- the teacher (F1 or F2) for an emulator: approximation error alone;
- L1 for any model: total error (approximation + misspecification). For F1 / F2 the two coincide.
Assimilation setting of M8: window 1.6 TU (160 steps), observations every n_o = 20 steps at t_k = k n_o,
k = 0..K−1 with K = 9, σ_o² = 4, full observations H = I.

**Local (one-interval) error m.** From true states x on the truth trajectories,
    δ(x) = M^{n_o}(x) − M_ref^{n_o}(x),
    m(M; M_ref) = (1/σ_o) · √( (1/3) Σ_{v∈{x,y,z}} Var_x[δ_v(x)] ).
- Samples x: the true states at the first K−1 = 8 observation times of each window, pooled over the 200 test cases
  (1600 samples).
- Var is the pooled variance, so a constant bias is not counted. This is the same statistic as the pre-registered ε
  screen of F1 / F2 and the m_h screen of the emulators (branch m7), on the current data.
- Meaning: the size of the model error that enters the weak-constraint 4D-Var each observation interval (its
  natural Q scale), relative to the observation noise.

**Window error w.** Both models run freely from the true initial state x0 of case i:
    e_k = M^{t_k}(x0) − M_ref^{t_k}(x0),   k = 0..K−1,
    w_i(M; M_ref) = (1/σ_o) · √( (1/(3K)) Σ_k |e_k|² ),
reported as mean ± std over the 200 test cases. The sum includes t_0, where e_0 = 0.
- Link to the 4D-Var cost: with y_k = x_true(t_k) + ε_k, ε_k ~ N(0, σ_o² I), and M_ref = the truth model,
      J_o(x0) = ½ Σ_k |y_k − M^{t_k}(x0)|² / σ_o²   ⇒   E[2 J_o(x0_true)] = p (1 + w²),   p = 3K = 27.
  So w² is the excess normalised observation misfit that the model produces from the true state. Strong-constraint
  4D-Var must absorb it by moving x0 away from the truth. w < 1: the model error over the window is below the
  observation noise. w > 1: it is detectable from the observations of one window.
- Unlike m, w includes the chaotic growth of the error over the window, and bias.

Both are computed by src/model_quality.py (test split, D1 v2, seed 42).

### 3. Results for the rectangle (test, 200 cases)

| model | definition | setting | ref | m vs ref | w vs ref | m vs L1 | w vs L1 | share w_L1 > 1 |
|---|---|---|---|---|---|---|---|---|
| L1 | L63, θ_L = (10, 28, 8/3) | — | — | 0 | 0 | 0 | 0 | 0 |
| F1 | θ_L ⊙ (1 + ε d), ε = 0.03, d = (1,1,1)/√3 | (10.1732, 28.4850, 2.7129), +1.73 % each | L1 | 0.180 | 1.33 ± 1.48 | 0.180 | 1.33 ± 1.48 | 0.40 |
| F2 | same, ε = 0.18 | (11.0392, 30.9098, 2.9438), +10.39 % each | L1 | 1.125 | 4.22 ± 1.65 | 1.125 | 4.22 ± 1.65 | 0.99 |
| E1 (good) | MLP tendency + RK4, teacher F1 | 10 train ICs, MLP 1 × 8, 10 000 steps | F1 | 0.170 | 0.97 ± 0.88 | 0.250 | 1.73 ± 1.30 | 0.70 |
| E2 (good) | same, teacher F2 | same | F2 | 0.167 | 0.86 ± 1.01 | 1.123 | 4.34 ± 1.56 | 1.00 |
| E1b (bad) | same, teacher F1 | 1 train IC, MLP 1 × 6, 5 000 steps | F1 | 0.442 | 2.45 ± 1.41 | 0.480 | 2.36 ± 1.28 | 0.86 |
| E2b (bad) | same, teacher F2 | same | F2 | 0.695 | 3.58 ± 1.74 | 1.302 | 4.23 ± 1.51 | 1.00 |

All seven models are stable over 20 TU (50 test ICs, |x| < 100). On the current data F1 / F2 have m = 0.180 / 1.125;
the screen values 0.155 / 0.958 (m7) were on the earlier data (window 1.1 TU, obs every 10 steps, σ_o² = 2).
Source: tmp/results/models/quality_test_rect.{json,md}, rectangle_table.md (job 84592).

### 4. Bad emulator: definition and selection
- Target (pilot): about 3 × the good emulator's error vs its teacher, i.e. m ≈ 0.45 (good 0.17), or w ≈ 3
  (good ≈ 0.9).
- Candidates: the good recipe (multi-step rollout loss, 1 → 10-step curriculum, Adam 1e-3 → 1e-5 cosine, best
  validation checkpoint) degraded by 1 training IC, one hidden layer, width 2–6 and 1 000–10 000 training steps;
  16 new emulators (8 per teacher) plus the earlier d1w6d1, d3w4d1 and d3w6d1.
- Screen (tmp/results/models/quality_test_badscreen.md), m / w vs teacher:

| setting | E1: m | E1: w | E2: m | E2: w | note |
|---|---|---|---|---|---|
| d1w6d1 (10k steps, previous bad) | 0.330 | 1.88 | 0.358 | 1.84 | |
| d1w6d1s5000 (5k steps) | 0.442 | 2.45 | 0.695 | 3.58 | **picked** |
| d1w5d1 | 0.393 | 2.51 | 0.609 | 3.67 | |
| d1w6d1s3000 | 0.601 | 2.75 | 0.844 | 3.76 | |
| d1w6d1s2000 | 0.950 | 3.15 | 1.141 | 3.80 | |
| d1w3d1, d1w4d1 | 1.6–2.7 | 5.0–6.1 | | | too degraded |
| d1w2d1, d1w6d1s1000 (F1), d3w4d1 (F1) | | | | | unstable over 20 TU |

- Pick (pilot, after the screen and the diagnostic figures tmp/figs/m8emu_*_d1w6d1s5000.png): d1w6d1s5000 for both
  teachers (one setting for F1 and F2, as for the good emulators). It meets the m target
  on F1 (0.44); on F2 the same setting gives m = 0.70.
- Diagnostics vs teacher (fig_emulator_ics): one-interval |error| / σ_o median (95 %) E1b 0.56 (1.40), E2b 0.72 (2.22)
  vs good 0.22 (0.57) / 0.23 (0.56); median |error| at 1.6 TU from the same test IC E1b 10.2, E2b 17.3 vs good 3.7 /
  3.0.
- Files: tmp/emulators/e{1,2}_d1w6d1s5000.pt; src/configs/models.yaml e1b / e2b.

### 5. Consequences for M8-R
- Q re-tuned for E1b / E2b with the same χ² rule (92 runs, select_q): truth L1 0.316 / 3.16 (unchanged), truth F1
  E1b 1.0 (was 0.562), truth F2 E2b 0.562 (was 0.1); no edge pick, no diverged case.
- Smoke rerun (64 runs, 10 val cases): 0 diverged; tmp/results/m8r/rect_smoke.{md,csv}; README rows updated.
- Previous bad-emulator (d1w6d1) results archived in tmp/results/m8r_d1w6d1/, tmp/figs/m8r_d1w6d1/,
  <data>/q_chi2_seed42_d1w6d1.json.
