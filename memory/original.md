# DAMETA — Project Constitution (long-term memory)

**Title:** Meta-Learning Impact on 4D-Var Initial Condition Estimation
**Owner / Pilot:** Jungwon Lee (IMT Atlantique)
**Created:** 2026-11-09
**Spec source:** /Odyssey/private/j25lee/vibeCodingInit.md

## Operating principles (from spec)
Data-First · Surgical Changes · Simplicity First · Goal-Driven · explore → plan → code

## Phase status
| Phase | Status |
|---|---|
| A – Architect (M-O-R) | DONE 2026-11-09 (open: g_theta inputs, await reference code) |
| B – Blueprint (North Star, Integrations, Source of Truth, Payload, Rules) | IN PROGRESS |
| C – Copilot | not started |

## M-O-R (scientific layer) — TO BE FILLED WITH PILOT
### Motivation
- Background: toy model **Lorenz-63** as first testbed.
- Problem statement (pilot, revised 2026-11-09): **a 4D-Var analysis is costly because of its iterative
  optimization process** (many cost/gradient evaluations through model + adjoint per analysis).
- Scope clarification (pilot): this is NOT about building a meta-trained emulator. The meta module learns the
  **iterative optimization process** itself (learning-to-optimize), cost-efficiently and physically consistently.

### Experimental design (pilot, 2026-11-09)
Two models:
- **M1** — physical Lorenz-63 (also generates truth + observations).
- **M2** — ML-based, auto-differentiable emulator mimicking M1 dynamics (gradient via autodiff, no hand adjoint).

Initial-condition estimation in strong- and weak-constraint 4D-Var, 4 configurations:
| # | Dynamics | Optimizer |
|---|---|---|
| E1 | M1 | classical iterative 4D-Var |
| E2 | M1 | 4D-Var + meta (learned optimizer) |
| E3 | M2 (emulator) | classical iterative 4D-Var |
| E4 | M2 (emulator) | 4D-Var + meta |
Meta module = learned update rule x0^{k+1} = x0^k + g_theta(...). Two meta configurations to compare (pilot, 2026-11-09):
- **Meta-A — optimized correction operator**: g_theta unrolled for fixed K steps, trained end-to-end on the outcome.
- **Meta-B — explicit learned update rule (imitation)**: observe the chosen classical optimizer's behaviour and
  train g_theta to mimic it (supervised or unsupervised).
Resolved (pilot, 2026-11-09):
- Meta-A loss = J(x0^K) (4D-Var cost after K unrolled steps; obs + background only, no truth).
- Meta-B teacher = gradient descent (adaptive variants e.g. Adagrad candidate).
  - supervised: mimic teacher steps/iterates.
  - unsupervised: optimize theta through J inside the meta loop via chain rule (pilot to provide reference code).
- g_theta inputs: TBD with reference code.
- Architectures of Meta-A and Meta-B may differ (not forced identical).
- One meta learner trained per dynamics model (M1, M2).
- Pilot expectation: Meta-B generalizes better because it learns from observing the iterative optimization.

Obs operator, assimilation window, cost function, B/R/Q etc.: to be defined with pilot later.

### Objective
- Hypothesis: _TBD_
- Target question: _TBD_
### Result
- Decision metrics (pilot, 2026-11-09, in priority order):
  1. Forecast RMSE **beyond the assimilation window** from the estimated x0.
  2. Computational cost.
  3. Physical consistency + gradient fidelity — core set approved: **G2** cos(dx^k, -gradJ_M1(x^k)) & fraction of
     true-J-decreasing steps; **P1** true-model cost J_M1(x0^a) for every analysis; **P3** fraction of analysis
     increment in unstable subspace (leading backward Lyapunov vectors). (G1, P2, P4 = optional/secondary.)
  4. Generalization to unseen configurations (initial conditions).
- Expected result: _TBD_
- Scientific impact: paper publication (target).
- Potential development: next models = Lorenz-96 (forced) or multi-layer QG.

## Blueprint — TO BE FILLED (Phase B)
- North Star (pilot, 2026-11-09):
  - **Scientific claim (real North Star):** "meta learner can re-learn missing dynamics of emulator and Meta-B beats
    Meta-A on generalization with delta inner loss meta task".
  - **Quantitative gate:** on held-out L63 initial conditions, a meta optimizer reaches forecast RMSE (beyond window)
    within **X = 5%** of classical GD 4D-Var using **<= 1/N, N = 10x** model+gradient evaluations, with G2 > 0,
    P1 ~ classical, P3 >= classical; for M1 and M2, strong and weak constraint.
  - "Missing dynamics" = BOTH intrinsic emulator imperfection (finite data/capacity) AND deliberate controlled
    degradation (e.g. dropped term, perturbed sigma/rho/beta, coarser dt).
  - Re-learning measured by **gap-closure ratio** = (RMSE_E3 - RMSE_E4) / (RMSE_E3 - RMSE_E1)
    (0 = no gain over emulator 4D-Var, 1 = recovers true-model 4D-Var); supported by G2 vs true M1 gradient.
  - "Delta inner loss meta task": NOT per-step dJ, NOT changing inner-loss tasks -> defined by pilot reference code (TBD).
- Integrations (pilot, 2026-11-09): **SLURM GPU** (compute), **TensorBoard** (tracking), **local git only** (no remote).
- Source of Truth: _TBD_
- Delivery Payload: _TBD_
- Behavioral Rules (partial, pilot 2026-11-09):
  - **Git: ONLY the user commits/pushes. Agent runs git only when explicitly asked.**
  - Paper target: every development step ships a **delivery-ready markdown update** (usually findings.md).
  - (remaining rules: Phase B Q5)

## Data schema — TO BE DEFINED (Data-First)
- Input: _TBD_
- Output: _TBD_
