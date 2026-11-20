# Findings — branch m6-ice-table (4D-Var ICE table)

# Milestone 6 — Learned optimizers and an adjoint-free learned gradient for 4D-Var initial-condition estimation under model error: OSSE training, real-world (OSE) transfer [paper-ready, 2026-11-20]

Sources: tmp/results/ice/ice_table_{osse,ose}_{test,test_ood}.{md,csv} (src/report_ice.py, 2026-11-20 15:09; the
older tmp/results/ice/ice_table_{test,test_ood}.* of 09:51 are superseded). Every number below is copied from these
tables; ratios are computed from the table means.

## Summary
- **Design.** The same 4D-Var experiment is run twice and only the observations change. In the **OSSE** (Table 1)
  the observations come from our knowledge model F1, and every meta learner is trained there. In the **OSE**
  (Table 2) the observations come from the real system E1, and the OSSE-trained learners are applied **frozen**.
- **Classical references.** All four classical optimizers (GD, Adam, Adagrad, strong-Wolfe L-BFGS) converge to the
  same analysis (IC RMSE 0.61–0.65 in-distribution). They differ only in cost. Strong-Wolfe L-BFGS is the cheapest by
  an order of magnitude: 5.5 gradients to reach 1 % of J\* in strong constraint and 10.5 in weak constraint (OSSE test).
- **Learned optimizers (cells 1, 2).** They keep L-BFGS quality (IC within ±0.007, 1-TU forecast within ±1 %). They
  reach 1 % of J\* **×1.10–1.28 faster** than L-BFGS in distribution. The gain shrinks to ×1.03–1.20 out of
  distribution. Cell 1 reaches the target in 0.97–0.99 of cases, cell 2 in 0.99–1.00.
- **Learned optimization (cell 3, K = 20 unroll).** It does **not** improve on L-BFGS. Only 0.62–0.92 of cases reach
  the target. Forecasts with the estimation model are 2–9 % worse.
- **Adjoint-free learned gradient (cell 4).** Learned tangent-linear operators are combined by the exact adjoint
  recursion and drive the unchanged Wolfe L-BFGS. This gives **the same analysis as L-BFGS with the exact adjoint**
  (IC within ±0.002 in distribution, forecast within ±1 %) with **no adjoint run**. The cost is
  **×2.4–3.8 lower** in gradient equivalents, and it holds on test_ood (×2.5–3.8) and in the OSE (×2.4–3.7).
  It needs about the same number of model evaluations as L-BFGS (5.6–10.9 forward runs vs 5.5–10.6 gradients), so
  the saving is exactly the cost of the adjoint.
- **Transfer to the real world.** The speed-ups measured in the OSSE are **kept unchanged** when the frozen learners
  assimilate real-system observations. Learners trained on OSE data themselves are no better (difference ≤ 0.3
  gradients). **Training on model-generated observations is enough.**
- **Model error (OSE).** 4D-Var with the false model brings that model's trajectory **37 % closer** to the real truth
  (mean error over lead 0–1 TU: 3.90 → 2.44, test). Cells 1, 2 and 4 preserve this to within ±0.02; cell 3 loses
  0.06–0.11.

## 1. Methods

**Truth, models and observations.** We use D1 v2 (Milestone 5). The real system E1 is Lorenz-63 (σ 10, ρ 28,
β 8/3; RK4, Δt 0.01). Estimation always uses a false model:
- **F1**, a knowledge numerical model (L63 with ρ = 27.5);
- **F2**, an MLP emulator of F1 with an autodiff adjoint.

E1 is never used as a 4D-Var model, because its physics is unknown in practice. The assimilation window is 1.1 TU
with 12 observation times every 0.1 TU. Only the first variable is observed (pattern x_only, H = [1 0 0]), with
noise variance σ_o² = 2. The first guess is a lagged-ensemble member of the false model, and B is the hybrid
covariance fitted on validation innovations (Milestone 5). Weak-constraint runs also estimate model-error forcings
η_i with σ_q = 0.1.

**OSSE and OSE.** The two settings share the true initial states, observation times, first guesses and B. They
differ only in the observations:
- **OSSE** (src/make_osse_obs.py): the truth is the F1 trajectory from the true initial state, observed with fresh
  noise. With F1 as estimation model this is a perfect-model twin, and with F2 it is an emulator twin. All meta
  learners are trained on the OSSE train split (1000 ICs) and select their best epoch on the OSSE val split
  (200 ICs).
- **OSE**: the observations are of E1, so model error is present. The OSSE-trained learners are applied frozen, with
  no retraining. Supplementary rows give learners trained on E1 observations ("OSE-trained").

**Scoring splits.** test (200 ICs) and test_ood (200 ICs in the held-out lobe-transition region). Meta rows pool
3 training seeds (600 runs per row).

**Classical references.** Each case runs c^{k+1} = c^k − f(c^k), with c = x₀ in strong constraint and [x₀, η] in
weak constraint. A case stops at the first iteration where |J^k − J^{k−1}| / J^{k−1} < 10⁻⁶, or at 1000
iterations; non-finite J is counted as diverged. The optimizers are:
- **L-BFGS**: history 10, lr 1, strong-Wolfe line search (c₁ 10⁻⁴, c₂ 0.9, ≤ 25 trials). The first step is −g with
  t = min(1, 1/‖g‖₁).
- **GD and Adagrad**: learning rates selected in Milestone 5 for full observations. GD uses 10⁻³ (3·10⁻⁴ for F2
  weak); Adagrad uses 0.3 in strong constraint and 0.1 in weak constraint.
- **Adam**: lr 0.1.

The learning rates were not re-tuned for x_only. L-BFGS is the reference and the teacher.

**Meta learners.** All run with the same stopping rule, a 150-iteration budget and the exact adjoint unless stated.
- **Cell 1, supervised learned optimizer (Δ_exact).** Input: x^k, g^k, the L-BFGS direction −H g^k of its own
  memory, k, and the 10 latest (s, y) pairs. Output: the update Δ̂ relative to the two-loop direction. Training
  target: the L-BFGS teacher displacement over two iterations, learned online with a log-relative-error loss. The
  first iteration is one teacher L-BFGS step, which is counted in the cost. At inference a backtracking safeguard
  halves any move that raises J (no extra model run), and there is no teacher.
- **Cell 2, unsupervised learned optimizer (Δ).** Same network and inputs as cell 1. It is trained on its own
  trajectories by the first-order decrease of J (l_perf, weight 2.5; cases that have already converged are masked
  out), plus a small alignment to the teacher step (weight 0.1) and an L2 penalty. 10 epochs; the best epoch
  minimises the 90 % quantile of J(x¹⁰) on the validation split.
- **Cell 3, learned optimization (Meta-A unroll).** Each move is the L-BFGS direction of its own moves plus a learned
  correction from (x, g, k). It is trained through a K = 20 unroll with loss mean_m J(x^m)/J(x⁰) and the backtracking
  safeguard. The budget is fixed at 20 gradients.
- **Cell 4, adjoint-free learned gradient.** One forward run of the false model gives J, the states x_{t_i} at the
  observation times and the residuals r_i = Hᵀ R⁻¹ (H x_{t_i} − y_i).
  - A network outputs one 3 × 3 tangent-linear operator per observation interval, A_θ(x_{t_i}) = I + MLP(x_{t_i})
    (3 layers × 128). Its only input is the standardised state.
  - The operators enter the exact adjoint recursion λ_L = r_L, λ_i = r_i + A_θᵀ λ_{i+1}. This gives ĝ_o = λ₀ for x₀
    and λ_i for η_i in weak constraint. ∇J_b and ∇J_q are added analytically.
  - Supervision: MSE between A_θ and the one-interval Jacobian of the ρ 27.5 emulator, computed by autodiff at the
    same states. The training controls are x₀ = x₀_b + s L z (L Lᵀ = B, s ∈ {0.5, 1, 2}), 64 per training IC. The
    best epoch minimises the median relative gradient error on 200 validation ICs.
  - (J, ĝ) drive the unchanged strong-Wolfe L-BFGS.
  - Cost is counted in forward runs, line-search trials included. It is converted to gradient equivalents by the
    measured gradient/forward wall-clock ratio of the same model: 3.66 (F1 strong), 3.80 (F1 weak), 2.45 (F2 strong),
    2.53 (F2 weak).

**Metrics.** All are given as mean ± std over cases (and seeds for meta rows).
- **IC RMSE**: x₀ error against the table's truth.
- **fc1_model**: 1-TU forecast RMSE from the analysis, run with the estimation model (the operational forecast).
- **fc1_true**: the same forecast run with the table's true model. In the OSE this is a diagnostic only.
- **traj_model**: estimation-model trajectory error, averaged over lead 0–1 TU.
- **Cost**: gradient evaluations (forward + adjoint, line-search trials included) until J ≤ 1.01 J\*, where J\* is
  the best final J over the four classical methods for the same case, model and constraint.
- **reached**: fraction of runs that hit this target.
- **G1**: cosine between the consumed gradient and the true-model gradient.

Wall-clock times were measured on shared nodes under different loads and are **not comparable** across rows. The
cost unit is therefore the gradient evaluation.

## 2. Results

### Table A — Cost to 1 % of J\* (gradient evaluations; cell 4 in gradient equivalents) and fraction reached

Entries: mean cost (fraction reached), with ×speed-up against L-BFGS on the right.

| setting | split | model / constraint | L-BFGS | cell 1 | cell 2 | cell 3 | cell 4 | ×c1 / ×c2 / ×c4 |
|---|---|---|---|---|---|---|---|---|
| OSSE | test | F1 strong | 5.5 (1.00) | 4.6 (0.99) | 4.9 (1.00) | 4.8 (0.92) | 1.6 (0.99) | 1.20 / 1.12 / 3.4 |
| OSSE | test | F2 strong | 5.5 (1.00) | 4.6 (0.99) | 5.0 (1.00) | 4.8 (0.82) | 2.3 (1.00) | 1.20 / 1.10 / 2.4 |
| OSSE | test | F1 weak | 10.6 (1.00) | 8.3 (0.97) | 8.6 (0.99) | 9.5 (0.81) | 2.8 (1.00) | 1.28 / 1.23 / 3.8 |
| OSSE | test | F2 weak | 10.5 (1.00) | 8.7 (0.98) | 8.5 (0.99) | 9.5 (0.80) | 4.3 (1.00) | 1.21 / 1.24 / 2.4 |
| OSSE | test_ood | F1 strong | 7.1 (1.00) | 6.6 (0.98) | 6.4 (0.98) | 6.5 (0.84) | 2.3 (0.96) | 1.08 / 1.11 / 3.1 |
| OSSE | test_ood | F2 strong | 7.3 (1.00) | 6.1 (0.97) | 6.7 (0.99) | 7.8 (0.74) | 2.8 (0.97) | 1.20 / 1.09 / 2.6 |
| OSSE | test_ood | F1 weak | 9.6 (0.99) | 9.3 (0.96) | 9.1 (0.97) | 11.7 (0.62) | 2.5 (0.97) | 1.03 / 1.05 / 3.8 |
| OSSE | test_ood | F2 weak | 9.7 (0.98) | 9.4 (0.96) | 9.3 (0.96) | 11.7 (0.67) | 3.9 (0.99) | 1.03 / 1.04 / 2.5 |
| OSE (frozen) | test | F1 strong | 5.6 (1.00) | 4.8 (0.99) | 5.2 (1.00) | 4.6 (0.90) | 1.5 (1.00) | 1.17 / 1.08 / 3.7 |
| OSE (frozen) | test | F2 strong | 5.5 (1.00) | 4.8 (0.99) | 5.3 (1.00) | 4.5 (0.84) | 2.3 (1.00) | 1.15 / 1.04 / 2.4 |
| OSE (frozen) | test | F1 weak | 10.3 (1.00) | 8.2 (0.97) | 8.4 (0.99) | 9.3 (0.82) | 2.8 (1.00) | 1.26 / 1.23 / 3.7 |
| OSE (frozen) | test | F2 weak | 10.4 (1.00) | 8.3 (0.97) | 8.2 (0.99) | 9.6 (0.83) | 4.2 (1.00) | 1.25 / 1.27 / 2.5 |
| OSE (frozen) | test_ood | F1 strong | 7.2 (0.99) | 6.5 (0.98) | 6.6 (0.99) | 6.9 (0.86) | 2.1 (0.97) | 1.11 / 1.09 / 3.4 |
| OSE (frozen) | test_ood | F2 strong | 7.3 (1.00) | 6.4 (0.99) | 6.6 (0.99) | 7.9 (0.74) | 3.0 (0.98) | 1.14 / 1.11 / 2.4 |
| OSE (frozen) | test_ood | F1 weak | 9.9 (0.99) | 9.5 (0.97) | 9.1 (0.98) | 11.8 (0.67) | 2.6 (0.99) | 1.04 / 1.09 / 3.8 |
| OSE (frozen) | test_ood | F2 weak | 10.2 (0.99) | 9.2 (0.97) | 9.7 (0.97) | 11.8 (0.71) | 4.0 (0.99) | 1.11 / 1.05 / 2.6 |

Cell 4 measured in raw forward runs (gradient equivalents × ratio), OSSE test: 5.9 / 5.6 / 10.6 / 10.9, against
L-BFGS 5.5 / 5.5 / 10.6 / 10.5 gradients.

The first-order classical methods are far behind (OSSE test, cost to 1 % of J\*):
- GD: 265–532 gradients, reached 0.54–0.83;
- Adam: 30.5–56.2 gradients, reached 0.99;
- Adagrad: 42.9–117.1 gradients, reached 0.94–0.99.

### Table B — Analysis quality, OSSE test (F1 strong; the other configurations show the same pattern)

| method | IC RMSE | fc1_model | traj_model | gradients total |
|---|---|---|---|---|
| first guess | 0.794 ± 0.579 | 6.955 ± 6.194 | 5.005 ± 3.967 | — |
| GD (lr 10⁻³) | 0.616 ± 0.457 | 2.863 ± 4.429 | 2.198 ± 2.990 | 831.5 ± 194.5 |
| Adam (lr 0.1) | 0.617 ± 0.473 | 2.845 ± 4.366 | 2.098 ± 2.896 | 117.0 ± 111.4 |
| Adagrad (lr 0.3) | 0.613 ± 0.468 | 2.834 ± 4.351 | 2.097 ± 2.897 | 168.5 ± 256.9 |
| L-BFGS (strong Wolfe) | 0.610 ± 0.470 | 2.838 ± 4.359 | 2.100 ± 2.904 | 10.6 ± 4.7 |
| cell 1 | 0.611 ± 0.471 | 2.838 ± 4.358 | 2.100 ± 2.905 | 11.8 ± 11.1 |
| cell 2 | 0.611 ± 0.471 | 2.838 ± 4.359 | 2.099 ± 2.904 | 10.8 ± 4.4 |
| cell 3 | 0.623 ± 0.472 | 2.919 ± 4.408 | 2.123 ± 2.917 | 20.0 (fixed) |
| cell 4 | 0.609 ± 0.471 | 2.822 ± 4.338 | 2.092 ± 2.889 | 5.5 ± 3.0 (grad. eq.) |

### Table C — Real world (OSE, frozen OSSE learners), test, F1: does 4D-Var bring our model closer to the truth?

| constraint | method | IC RMSE | traj_model | fc1_model | fc1_true (diagnostic) |
|---|---|---|---|---|---|
| — | first guess | 0.794 ± 0.579 | 3.902 ± 3.608 | 5.060 ± 5.231 | — |
| strong | L-BFGS | 0.642 ± 0.463 | 2.442 ± 2.585 | 3.636 ± 5.014 | 5.567 ± 6.589 |
| strong | cell 1 | 0.648 ± 0.481 | 2.425 ± 2.561 | 3.631 ± 5.014 | 5.588 ± 6.613 |
| strong | cell 2 | 0.642 ± 0.463 | 2.442 ± 2.585 | 3.636 ± 5.013 | 5.566 ± 6.589 |
| strong | cell 3 | 0.654 ± 0.467 | 2.506 ± 2.608 | 3.791 ± 5.118 | 5.418 ± 6.513 |
| strong | cell 4 | 0.639 ± 0.462 | 2.441 ± 2.589 | 3.627 ± 5.000 | 5.562 ± 6.591 |
| weak | L-BFGS | 0.641 ± 0.461 | 2.451 ± 2.737 | 3.617 ± 4.854 | 5.587 ± 6.643 |
| weak | cell 1 | 0.644 ± 0.464 | 2.453 ± 2.736 | 3.618 ± 4.851 | 5.583 ± 6.645 |
| weak | cell 2 | 0.643 ± 0.467 | 2.450 ± 2.735 | 3.617 ± 4.852 | 5.583 ± 6.639 |
| weak | cell 3 | 0.647 ± 0.485 | 2.564 ± 2.832 | 3.872 ± 5.228 | 5.664 ± 6.753 |
| weak | cell 4 | 0.637 ± 0.457 | 2.464 ± 2.801 | 3.670 ± 4.985 | 5.599 ± 6.657 |

### Table D — Meta training cost (wall-clock per seed, s, OSSE; not part of the inference cost)

| cell | F1 strong | F2 strong | F1 weak | F2 weak |
|---|---|---|---|---|
| 1 supervised Δ | 100 ± 38 | 127 ± 41 | 283 ± 128 | 282 ± 117 |
| 2 unsupervised Δ | 1719 ± 282 | 5006 ± 906 | 11244 ± 760 | 14046 ± 2346 |
| 3 unroll K 20 | 1214 ± 147 | 2012 ± 252 | 1240 ± 134 | 2046 ± 233 |
| 4 learned gradient | 708 ± 99 | 1098 ± 132 | 782 ± 117 | 756 ± 119 |

Seed spread (std over seeds of the mean IC RMSE): ≤ 0.007 for cells 1, 2 and 4 in distribution (≤ 0.014 on
test_ood); 0.003–0.012 for cell 3 (0.036 on test_ood, F2 strong).

## 3. Results (paper-ready text)

**Classical 4D-Var.** With only the first variable observed, the four classical optimizers reach the same analysis.
In the OSSE the IC RMSE drops from 0.794 at the first guess to 0.610–0.617 for Adam, Adagrad and L-BFGS, and to
0.616–0.651 for GD. The 1-TU forecast error is about halved (6.96 → 2.84). The methods differ only in cost
(Table B). Strong-Wolfe L-BFGS reaches 1 % of J\* in 5.5 gradients in strong constraint and 10.5–10.6 in weak
constraint, in every case. Adam needs ×6, Adagrad ×8–11 and GD ×50. GD fails to reach the target in 17–46 % of the
cases within 1000 iterations, because its learning rate was not re-tuned for x_only observations. The weak-constraint
problem costs twice as much, because the control grows from 3 to 36 variables (x₀ plus 11 forcings). L-BFGS is
therefore the reference and the teacher for the learned methods.

**Learned optimizers (cells 1 and 2).** Both learned Δ optimizers reproduce the L-BFGS analysis: IC RMSE within
±0.007, 1-TU forecast within ±1 %, seed spread ≤ 0.007 (Tables A, B). In distribution they reach 1 % of J\* ×1.10–1.28
faster, and the gain is largest in weak constraint (8.3–8.7 vs 10.5–10.6 gradients). The gain is a faster early
descent, not a cheaper convergence. The total number of gradients until the stopping rule is equal to L-BFGS for
cell 2 (10.8 vs 10.6 strong, 19.8–20.7 vs 21.1–21.5 weak) and slightly larger with a heavier tail for cell 1
(11.8 ± 11.1 vs 10.6 ± 4.7). The two cells trade off differently:
- **Cell 1 (supervised)** trains in 2–5 minutes but misses the target in 1–3 % of the runs.
- **Cell 2 (unsupervised)** reaches it in 99–100 % but trains 17–50× longer (0.5–3.9 h per seed), because the Wolfe
  teacher line search runs on the learner's own states.

Out of distribution (test_ood) the speed-up shrinks to ×1.03–1.20, smallest in weak constraint (×1.03–1.05), and
the reached fraction drops to 0.96–0.99.

**Learned optimization (cell 3).** Training an L-BFGS-based learned optimizer through a 20-step unroll does not
pay. Its cost to the target looks similar (4.8 vs 5.5 strong), but it is computed only over the 0.80–0.92 of runs
that reach the target, and the fixed budget is 20 gradients. The analysis is slightly worse: IC 0.623 vs 0.610 and
forecast 2.92 vs 2.84 (OSSE test, F1 strong). Out of distribution only 0.62–0.84 of runs reach the target. Cell 3
is the least reproducible learner (seed spread up to 0.036).

**Adjoint-free learned gradient (cell 4).** Replacing the adjoint by learned interval tangent-linear operators,
combined by the exact adjoint recursion, leaves the 4D-Var solution unchanged. In distribution the IC RMSE is within
±0.002 of L-BFGS and the forecast within ±0.6 %. Table B: 0.609 vs 0.610, forecast 2.822 vs 2.838. L-BFGS driven by
the learned gradient needs about as many model evaluations as with the exact gradient (5.6–10.9 forward runs vs
5.5–10.6 gradients). Each evaluation is a forward run only, which costs 1/3.7 (F1) or 1/2.5 (F2) of a
forward + adjoint pair. The cost to 1 % of J\* therefore falls ×3.4–3.8 for F1 and ×2.4 for F2, while 0.99–1.00 of
runs still reach the target. The gain survives out of distribution (×2.5–3.8, reached 0.96–0.99). One weak-constraint
F1 run diverged out of 600 on test_ood. This is the only divergence in the study. The speed-up is the ratio of
forward to adjoint cost, so it is model-dependent and should grow for models whose adjoint is expensive or missing.

**Transfer from OSSE training to real observations.** Learners trained only on observations generated by our
knowledge model keep their full speed-up when they assimilate observations of the real system, without retraining
(Table A, OSE rows). In-distribution speed-ups:
- cells 1 and 2: ×1.04–1.27 (×1.10–1.28 in the OSSE);
- cell 4: ×2.4–3.7 (×2.4–3.8 in the OSSE).

Learners trained on the real observations themselves bring nothing extra. Their cost to the target differs by
≤ 0.3 gradients from the frozen OSSE learners (e.g. cell 4: 1.5 vs 1.5; cell 1: 4.9 vs 4.8; F1 strong). The learned
components therefore encode properties of the optimisation problem and of the model dynamics, not of the observed
system, and **meta-training can be done entirely in a twin experiment**.

**Model error.** With real observations the analysis is about as accurate at t₀ as in the twin (IC RMSE 0.637–0.654
vs 0.609–0.623). The forecast from it with our model degrades by 28 % (fc1_model 3.64 vs 2.84), which is the cost of
model error. 4D-Var still brings our model's trajectory substantially closer to the real one. Over lead 0–1 TU its
error drops from 3.90 at the first guess to 2.44 (−37 %), and the 1-TU forecast from 5.06 to 3.64 (−28 %; Table C).
Cells 1, 2 and 4 preserve this improvement (traj_model within ±0.02 of L-BFGS). Cell 3 loses part of it: 2.51–2.56,
with a forecast 4–7 % worse.

Running the true model E1 from the analysis gives a worse forecast than running our own model (5.57 vs 3.64). The
analysis is a state that makes the false model fit the observations, so it compensates the model error and is not
the best initial condition for the real dynamics. This is why the IC RMSE alone is a poor score under model error,
and why traj_model is the relevant real-world metric.

Cell 3 is the only learner that moves the analysis away from the false-model optimum. It slightly improves the
true-model diagnostic in strong constraint (fc1_true 5.42 vs 5.57 for F1, 5.46 vs 5.69 for F2), as cell 5 did in
Milestone 5. It is worse with our own model and in weak constraint, so this is not a usable regularisation.

**Lobe-transition cases (test_ood).** In the held-out region 4D-Var increases the mean IC error: 0.441 at the first
guess vs 0.55–0.69 after assimilation, with std 1.0–2.3 against 0.47, which points to a few cases switching basin.
Yet it improves the trajectory in every setting: traj_model 6.06 → 2.22–2.83 in the OSSE and 6.13 → 2.61–3.03 in
the OSE; fc1_model 10.2 → 4.1–5.4. Near the lobe transition the first guess is close to the truth at t₀ but on the
wrong side of the separatrix. 4D-Var fixes the trajectory at the price of a larger t₀ error in a few cases. All
methods share this behaviour, so it is a property of the problem, not of the optimizer.

**Emulator gradients near the optimum (F2).** The emulator's gradient is aligned with the F1 gradient at the first
guess (G1 = 0.998). At the F2 analysis the cosine is low and scattered: 0.42 for GD, 0.21 for Adam, 0.00 for Adagrad
and −0.05 for L-BFGS (OSSE test, strong). The F2 gradient is near zero at its own minimum, and what remains of the
F1 gradient there is the model-error mismatch, so G1 is uninformative near convergence. Analysis quality is
unaffected: IC 0.610 with F1 and with F2, true-model forecast +0.6 %.

## 4. Limitations and open points
- Speed-ups are reported as differences of case means. **Paired tests (Wilcoxon, by case and seed) have not been
  run yet.** The case-to-case std (≈3 gradients strong, ≈6 weak) is large relative to the cell 1/2 gains of about
  1–2 gradients. 600 meta runs per row against 200 classical cases give a standard error of ≈0.2–0.4 gradients, so
  the cell 1/2 gains in distribution are likely significant, while their test_ood gains (×1.03–1.05 weak) are not
  established.
- Cell 4's gradient-equivalent cost depends on a single measured forward/gradient wall-clock ratio per model and
  constraint (tmp/results/grad_cost_ratio.json). The raw forward-run counts are given above for independence from
  that measurement.
- GD and Adagrad learning rates are those selected for full observations. Their costs are upper bounds for x_only.
- Cell 3 inputs (x, g, k) do not include the L-BFGS memory that cells 1 and 2 receive. Whether aligning its inputs
  would close the gap is open.
- Real-world checks (c) (observations through an operator not seen in training) and (d) (future observations of E1)
  are planned for Milestone 7 and not covered here.

---

# Development log (chronological)
