# Findings

## Environment (2026-11-09)
- `environment.yaml` defines conda env `oceanai` (py3.12, torch, lightning, hydra, xarray/netcdf/zarr, CUDA 12.8).
- Local `.venv` (py3.12.13) is essentially empty — science stack not installed there.
- conda/mamba available at /Odyssey/private/j25lee/miniforge3.
- Current login node: no GPU visible (`nvidia-smi`: No devices found).

## Research notes
_(Phase B.2 — to be filled)_

## Meta-A vs Meta-B — open ambiguities (2026-11-09)
1. Meta-A training loss: J(x0^K) [obs-only, 4D-Var-consistent] vs ||x0^K - x0_true|| [truth-supervised] vs sum_k J(x0^k).
2. Meta-B "unsupervised" meaning undefined (supervised = regress teacher step dx^k).
3. Teacher optimizer for Meta-B (GD / L-BFGS / Gauss-Newton incremental) and K at inference (fixed vs until convergence).
4. Inputs of g_theta (grad J only? + x0^k, J, history/memory?) -> determines whether cost saving = fewer iterations only.
5. Fair comparison: same g_theta architecture for A and B, only training target differs?
6. Meta trained separately per dynamics (M1, M2) or transferred; weak-constraint control vector size.
- (2026-11-09) Resolved 1,2,3,5,6 with pilot (see original.md). Open: g_theta inputs (await reference code);
  watch-point: Meta-B unsupervised (J via chain rule) vs Meta-A (J(x0^K)) — difference to be pinned down from code
  (e.g. per-step vs end-of-unroll loss, GD-structured vs free correction operator).

## Proposed metrics for physical consistency & gradient fidelity (draft, 2026-11-09, awaiting pilot)
Gradient fidelity (reference = exact M1 adjoint gradient, available for L63):
- G1 emulator gradient: cos(gradJ_M2, gradJ_M1) and ||gradJ_M2 - gradJ_M1|| / ||gradJ_M1|| at same x0 (E3/E4).
- G2 meta step direction: cos(dx^k, -gradJ_M1(x^k)) per iteration; fraction of steps decreasing true J_M1.
Physical consistency:
- P1 true-model cost: re-evaluate every analysis (incl. emulator-based) with J_M1(x0^a).
- P2 chi-square check: E[2 J_min] ~ number of observations (Talagrand/Desroziers) — analysis statistically consistent with B, R(, Q).
- P3 increment in unstable subspace: fraction of (x0^a - x0^b) projected on leading backward Lyapunov vectors (AUS theory).
- P4 smoothness: iterate-path tortuosity = sum||dx^k|| / ||x0^K - x0^0||; weak-constraint: temporal smoothness of eta_k.
Forecast: RMSE vs lead time beyond window; valid prediction time (lead time until normalized error > threshold, in Lyapunov times).
Cost: # model+adjoint (or emulator fwd+bwd) evaluations, iterations to reach target J, wall-clock; meta/emulator training cost reported separately (amortization).
Generalization: held-out initial conditions on the attractor (train/val/test split by IC).
