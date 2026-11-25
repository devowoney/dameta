# Decisions — branch m8-classic-4dv

| Date | Decision | Rationale |
|---|---|---|
| 2026-11-25 | Classical 4D-Var with F1, F2, E1, E2 as estimation models; L1 never inside 4D-Var (truth for metrics only) | Pilot: L1 unknown in the experiments |
| 2026-11-25 | Grid: strong / weak × full / x-only (noisy) × GD, Adagrad, Adam, L-BFGS (strong Wolfe, lr 1); window 0.6 TU (60 steps), forecast 5 TU; each model's own lagged first guess + hybrid B; test split 200 cases | Pilot spec (≥ 200 samples); D1 v2 test has 200 ICs |
| 2026-11-25 | E1 / E2 adjoint = torch autograd (reverse mode) through MLP + RK4 | Pilot: "ML backprop to compute the adjoint" |
| 2026-11-25 | traj_deg = 1/K Σ_k ‖x_a^k − x_true^k‖₂ over k = 0..560 (window + 5 TU), x_a^k run by the estimation model (with η if weak) | Pilot choice (AskUserQuestion) |
| 2026-11-25 | κ_H = λ_max/λ_min only when λ_min > 0, else NaN; λ_min, λ_max, # negative eigenvalues, non-PD fraction reported separately | Pilot choice |
| 2026-11-25 | Hessian = exact (double backward) of the estimation-model J w.r.t. the full control (d = 3 strong, 21 weak), every iterate + at the analysis; cos_truth uses the x0 block of the gradient | Spec "each iteration"; weak η has no true counterpart |
| 2026-11-25 | lr re-tuned on val (200 cases) per model × scheme × obs × optimizer with the pre-registered select_lr rule (≤ 1 % diverged → lowest median J → fewer evals), grid extended at an edge | Pilot choice; window changed from 1.1 to 0.6 TU |
| 2026-11-25 | Worktree .claude/worktrees/m8-classic-4dv, branch m8-classic-4dv from 3b9a0e1; tmp/data (60-step D1 v2, bg_*, hybrid_B) and tmp/emulators/{e1,e2}_d10w8d1.pt copied from m7 | Pilot request |
