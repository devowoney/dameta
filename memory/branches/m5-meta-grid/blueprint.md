# M5 experiment blueprint — model-error-isolated BICE / BPE with a meta-learner grid (draft 2026-11-14)

Status: design agreed with the pilot on all structural points (2026-11-14); items marked [ASSUMPTION] are defaults
awaiting confirmation. Nothing below has been built or run yet.

## 1. Models
| role | model | used for |
|---|---|---|
| truth | L63 (σ 10, ρ 28, β 8/3), RK4, Δt 0.01 | truth trajectories, observations, evaluation only |
| false F1 | knowledge numerical model: L63 with ρ 27.5 (other parameters exact) | 4D-Var BICE and BPE |
| false F2 | emulator rho275 (MLP tendency + RK4, trained on ρ 27.5 trajectories; quality-checked) | 4D-Var BICE |
| reference E1 | truth model inside 4D-Var (perfect-model upper bound) | reference only |
All estimation runs on F1 or F2: the inverse problem carries model error and nothing else.

## 2. Data (regenerated; D1 v2)
- Window L = Lyapunov time of the false model, common to F1 and F2: 1/λ₁ = 1.11 TU (F1), ≈ 1.16 TU (F2)
  → **L = 1.1 TU = 110 steps** [ASSUMPTION: common value = F1 rounded to whole 0.1-TU obs intervals].
- Observations from the truth: every 10 steps (0.1 TU) → 12 obs times; y = H x_true + N(0, 2 I); patterns full /
  x_only. Forecast range 5 TU beyond the window.
- Splits by IC as before (train 1000 / val 200 / test 200 / test_ood 200; same held-out region rule); D1 v2 also
  stores the truth over the lag horizon before t0 (needed for the lagged ensemble).

## 3. First guess and B: time-lagged ensemble of the FALSE model
For each case and each false model F ∈ {F1, F2} (and the truth model for the E1 reference):
    member_j = M_F^(τ_j) ( x_true(t0 − τ_j) ),   τ_j = j × 0.1 TU, j = 1 … 10   [ASSUMPTION: N = 10, max lag 1 TU]
i.e. forecasts launched at earlier times, all valid at t0 (lagged-average forecasting). Their error is grown by the
false model's own dynamics and model error — the first guess is never the truth or truth + white noise.
    x0_b = ONE time-lagged member, M_F^(τ_b)(x_true(t0 − τ_b))   — a balanced state of the false model; NOT the
           ensemble mean (a smoothed mean is off-attractor / unbalanced: harmless for L63, harmful for QG or ocean
           models; pilot 2026-11-14)   [ASSUMPTION: τ_b = 0.5 TU, about half the window]
    B    = sample covariance of the lagged members (3 × 3, case-specific, flow-dependent)
    [ASSUMPTION: B regularised by a small diagonal floor, e.g. + 0.01·tr(B)/3·I, so it is invertible]
Weak constraint: Q unchanged (σ_q = 0.1).

## 4. Inverse problems
- **BICE** (F1 and F2): control x0 (strong) or (x0, η) (weak); J = J_b(B lagged) + J_o (+ J_q).
- **BPE** (F1 only — the emulator has no physical parameters): joint control [x0, σ, ρ, β] (strong constraint),
  J = J_b(x0; B lagged) + J_p + J_o, parameter prior = F1's own parameters (10, 27.5, 8/3)
  [ASSUMPTION: prior std 10 % of each value; weak-constraint BPE not planned].

## 5. Optimisers
Classical (lr by the pre-registered val rule per configuration): GD, Adagrad.
Adjoint-free baselines (for the cell-8 claim): finite-difference gradient on F1 (d + 1 forward runs per gradient),
hybrid (J from F1, gradient from the F2 emulator adjoint).
Meta grid (learner input always the running state x^k, [∇J(x^k)], emb(k); never x0_b):
| cell | kind | training | input | inference cost |
|---|---|---|---|---|
| 1 | learned optimizer | unsupervised (Δ inner loss) | x, ∇J, k | 1 gradient / iteration |
| 2 | learned optimizer | supervised (5-step teacher lookahead) | x, ∇J, k | 1 gradient / iteration |
| 3 | learned optimizer | unsupervised (greedy) | x, k [+ obs] | 1 forward / iteration |
| 4 | learned optimizer | supervised (running gradient) | x, k [+ obs] | 1 forward / iteration |
| 5 | learned optimization | unsupervised J(x^K), K 20 (sweep 5-50) | x, ∇J, k | K gradients |
| 7 | learned optimization | unsupervised J(x^K) | x, k [+ obs] | 0 model calls |
| 8 | learned gradient (adjoint-free) | supervised on emulator ∇J | x, innovations d^k, k | 1 forward / iteration |

## 6. Evaluation (vs truth; 3 training seeds; val for tuning only; test and test_ood)
IC RMSE, forecast RMSE beyond the window (false model forecast = operational; truth-model forecast = IC quality),
cost to 1 % / 5 % of J*, gradient / forward evaluations, G2, P1, P3, basin switches; BPE: parameter error vs
(10, 28, 8/3) and forecast with the estimated parameters; gap closure vs E1 for F1 and F2.

## 7. Build order
1. D1 v2 generator (window 110, 12 obs, pre-t0 truth) + lagged-ensemble backgrounds per model (x0_b, B).
2. Cost with case-specific B; BPE control [x0, σ, ρ, β] with parameter prior; F1 as a step function.
3. Tests (TL/adjoint with B, BPE gradient FD check, lagged ensemble reproducibility).
4. Classical baselines: lr sweeps (val) + finals for E1 / F1 / F2 (BICE) and F1 (BPE); adjoint-free baselines.
5. Cell 8; then the meta grid (cells 1-5, 7, 8) on F1 and F2, 3 seeds.