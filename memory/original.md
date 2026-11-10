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
| A – Architect (M-O-R) | DONE 2026-11-09 (g_theta inputs resolved from bice reference 2026-11-10) |
| B – Blueprint (North Star, Integrations, Source of Truth, Payload, Rules) | DONE 2026-11-10 (plan v0 + DA config v0 approved) |
| C – Copilot | IN PROGRESS — M1 done, M2 done (merged), next M3 |

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
- Meta-B protocol (2026-11-10): train across train ICs, then FROZEN inference on test ICs (main); online per-IC
  adaptation ("one task") = secondary variant.
- Meta-B supervised (2026-11-10) = ALIGN phase alone; S input = current IC x0^k + iteration embedding emb(k)
  (no gradient), target = teacher update f(x0^k), MSE. [assumption: "original IC" = current iterate, not x0^b]
- Code reuse authorized: port bice delta-chain, strong-4DV IC optimization, Hydra and SLURM logic into src/
  (bice repo itself untouched).
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
    within **X = 5%** of classical 4D-Var using **<= 1/N, N = 10x** model+gradient evaluations, with G2 > 0,
    P1 ~ classical, P3 >= classical; for M1 and M2, strong and weak constraint.
  - **Cost-gate reference (pilot, 2026-11-10):** report vs BOTH classical GD and Adagrad; the claim is anchored on
    Adagrad (strongest classical baseline). Cost = gradient evaluations to reach J <= 1.01 J* (J* = best final J per
    case). E1 targets for meta (<= 1/10 Adagrad): ~7 evals strong/full test, ~14 weak/full test, ~13 strong/full
    test_ood (GD-referenced targets ~26 / 62 / 22 reported alongside).
  - "Missing dynamics" = BOTH intrinsic emulator imperfection (finite data/capacity) AND deliberate controlled
    degradation (e.g. dropped term, perturbed sigma/rho/beta, coarser dt).
  - Re-learning measured by **gap-closure ratio** = (RMSE_E3 - RMSE_E4) / (RMSE_E3 - RMSE_E1)
    (0 = no gain over emulator 4D-Var, 1 = recovers true-model 4D-Var); supported by G2 vs true M1 gradient.
  - "Delta inner loss meta task" = bice Δ inner-loss mode (teacher-step stretch chain + first-order ΔJ meta loss);
    see findings.md "Reference implementation". This IS the Meta-B mechanism.
- Integrations (pilot, 2026-11-09): **SLURM GPU** (compute), **TensorBoard** (tracking), **local git only** (no remote).
- Source of Truth (pilot, 2026-11-09): data **generated by M1 (L63) from fixed seeds**; generator script in src/ +
  Hydra config is the canonical source (exact regeneration). Stored as NetCDF/Zarr (xarray) in tmp/data/ until pilot
  authorizes run/. Per file: truth trajectory, observations, obs mask/operator, background x0^b, metadata
  (sigma, rho, beta, dt, seed, window). Train/val/test split **by initial condition** (test ICs held out).
  Emulator trained on the **same trajectories** (deliberate degradation applied where required).
- Delivery Payload (pilot confirmed, 2026-11-09):
  - memory/findings.md: per-development, paper-ready entry (methods paragraph, results table E1-E4 x Meta-A/B x
    strong/weak with metrics, figure links, verification note).
  - Figures: PNG + PDF/SVG in tmp/figs/ (forecast RMSE vs lead, J & RMSE vs #grad calls, G2 per iter,
    gap-closure ratio, P3 bars, phase-space trajectories).
  - Tables: CSV + markdown (LaTeX-convertible). TensorBoard logs: tmp/tb/. Hydra configs + checkpoints per run.
  - Everything stays in tmp/ until pilot authorizes move to run/.
  - Leakage guard: held-out test ICs excluded from BOTH emulator training and meta training.
- Behavioral Rules (partial, pilot 2026-11-09):
  - **Git: ONLY the user commits/pushes. Agent runs git only when explicitly asked.**
  - Paper target: every development step ships a **delivery-ready markdown update** (usually findings.md).
  - All work in tmp/; move to run/ only with pilot authorization. Update progress.md every step. No CLAUDE.md.
  - Code: 132-char line limit, commented, new features behind feature flags.
  - SLURM: agent writes job scripts, submits ONLY with pilot OK (state resources + est. time).
  - Correctness first: float64 for L63; TL/adjoint dot-product test (M1) and finite-difference vs autodiff check
    (M2) before any comparison experiment.
  - Reproducibility: fixed seeds, saved Hydra configs, environment version logged per run.
  - Scientific integrity: report negative/contrary results as-is; never tune on test ICs; report multiple seeds with
    spread, never best-only.
  - Stack: PyTorch (+ Lightning where useful) + Hydra, conda env `oceanai`.
  - Tone: concise, technical, English.
  - Never without asking: install packages, delete files, modify pilot's reference code.

## Data schema — DRAFT v0 (2026-11-09, awaiting pilot sign-off; DA hyper-params are symbols, values TBD)
Symbols: N_ic (# initial conditions per split), L (window steps), F (forecast steps beyond window),
d=3 (L63 state), p (obs per time), K (optimizer iterations), dt.

### D1 dataset  `tmp/data/l63_<split>_seed<S>.nc`   (xarray, float64)
| var | dims | meaning |
|---|---|---|
| x_true | (ic, t, var) t in [0, L+F] | truth trajectory from M1 |
| y_obs | (ic, t_obs, obs) | observations y = H(x_true) + eps, eps ~ N(0,R) |
| obs_mask | (t, var) bool | which variables observed at which times (defines H) |
| x0_b | (ic, var) | background IC = x_true[t=0] + eps_b, eps_b ~ N(0,B) |
| attrs | — | sigma, rho, beta, dt, L, F, seed, R, B, (Q), split, generator git-free version tag |
Splits by IC: train / val / test (in-distribution) + **test_ood** (held-out lobe-transition region, 2026-11-10):
feature flag heldout_region; train/val/test ICs have r_xy = sqrt(x^2+y^2) >= 4.0, test_ood ICs r_xy < 3.0
(gap = 1 background std). Test/test_ood ICs never seen by emulator or meta training.
Secondary OOD axis (no new data): x-only obs pattern held out from training.

### D2 emulator  M2 — APPROVED 2026-11-10 (NN tendency + RK4)
NN tendency + fixed RK4 integrator: dx/dt = h_phi(x), MLP 3→64→64→3, smooth activations (SiLU/tanh; no ReLU → smooth
Jacobian for gradient fidelity), RK4 at the same dt as M1. Trained on D1-train x_true, multi-step rollout loss
(n_roll ~10). Degradation flag: e.g. h_phi = known-partial-L63 + NN trained with a term removed / wrong rho.

### D3 analysis output (one per method x constraint x ic)  `tmp/results/<exp>.nc`
| var | dims | meaning |
|---|---|---|
| x0_iter | (ic, k, var) | iterates x0^k, k=0..K (weak: + eta_iter (ic,k,t,var)) |
| J_iter | (ic, k) | cost along iterations (method's own dynamics) |
| J_true_iter | (ic, k) | cost re-evaluated with M1 (P1) |
| cos_step_grad | (ic, k) | G2: cos(dx^k, -grad J_M1(x^k)) |
| n_grad_evals, wall_s | (ic,) | cost metrics |
| rmse_fc | (ic, lead) | forecast RMSE beyond window from x0^a |
| p3_unstable_frac | (ic,) | P3 |
| attrs | — | method in {GD, Adagrad, MetaA, MetaB-sup, MetaB-unsup}, dynamics in {M1, M2}, constraint, seed, config hash |


## DA configuration v0 — APPROVED 2026-11-10
- L63 sigma=10, rho=28, beta=8/3; RK4 dt=0.01; float64.
- Window L=100 steps (1 TU); obs every 10 steps (11 obs times incl. t0); H = full state (default), x-only (pattern 2).
- R = sigma_o^2 I, sigma_o^2 = 2. B = sigma_b^2 I, sigma_b = 1; x0_b = truth + N(0,B).
- Strong: J = 1/2||x0-x0_b||^2_{B^-1} + 1/2 sum_t ||y_t - H M_{0->t}(x0)||^2_{R^-1}.
- Weak: control (x0, eta_1..eta_10), eta added at obs times; J_q = 1/2 sum ||eta_i||^2_{Q^-1}, Q = sigma_q^2 I, sigma_q=0.1.
- Forecast F = 500 steps beyond window. ICs: spin-up 1000 steps; train/val/test/test_ood = 1000/200/200/200.
- Classical budget: <= 1000 iters or rel dJ < 1e-6; GD/Adagrad lr tuned on val.
- Meta: K = 20 (bice default); meta budget <= 1/10 classical; Meta-A K-sweep {5,10,20,50}.

## Memory protocol for branches / worktrees (pilot, 2026-11-10)
- Each milestone/feature after M1 is developed in a git worktree under .claude/worktrees/<branch> on its own branch.
  The user alone commits, pushes and merges.
- Inside a worktree, do NOT edit the shared memory files (original.md, findings.md, progress.md, decisions.md,
  task_plan.md). Write branch-local notes instead in `memory/branches/<branch>/`:
    progress.md   — dated log of the branch
    findings.md   — paper-ready entries produced on the branch (same style as findings.md)
    decisions.md  — decision rows (same table columns as decisions.md)
- New files never conflict at merge. After the user merges a branch into main, the agent (on request, on main) folds
  the branch notes into the shared files and marks the branch folder "folded".
- The main checkout keeps the shared files as the single source of truth for state between milestones.
