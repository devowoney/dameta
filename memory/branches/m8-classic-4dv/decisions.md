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
| 2026-11-25 | Redefined 4D-Var experiment (D1 v3 data): first guess = each model's 0.5-TU lagged forecast valid at **t0 − 5 steps** used as the t0 state (model error + timing error); the whole lagged ensemble (B) shifted by the same 5 steps (`background.time_shift`, history 105) | Pilot: "uncertainty = model propagation + temporal misrepresenting"; option "shift whole ensemble" |
| 2026-11-25 | σ_o² = 4 (was 2); ε1 / ε2 and E1 / E2 unchanged | Pilot: larger observation noise |
| 2026-11-25 | Harder problem: obs every 20 steps (4 times in the 0.6-TU window, was 10) and a weaker background term J_b with B = b_infl × B_hybrid (`cost.b_infl`); b_infl picked after a val smoke (1, 4, 16; L-BFGS) | Pilot options "sparser obs in time" + "weaker background term" |
| 2026-11-25 | Old data / results archived: tmp/data/w60_o10_s2/, tmp/results/m8_o10s2/ | Keep the previous M8 run |
| 2026-11-25 | Hybrid B s² grid extended to 1e3 (0.05-decade step kept): with the timing error all four models hit s² = 10; retuned α 0.95 and s² 63 / 12.6 / 31.6 / 11.2 (F1 / F2 / E1 / E2), χ² test 3.2 / 2.8 / 2.8 / 2.9 | Grid-edge rule |
| 2026-11-25 | b_infl = 4 (config default) | Pilot choice after the val smoke (tmp/results/m8/binfl.md) |
| 2026-11-25 | Feasibility first: every configuration is smoke-tested on 10 val cases before any 200-case run | Pilot: "don't run 200 samples for all; once every configuration is confirmed, run the real simulation" |
| 2026-11-25 | Longer windows tried: 100 (1.0 TU, 6 obs) and 160 (1.6 TU, 9 obs) steps, obs every 20 (tmp/data_w100, tmp/data_w160; same ICs / first guess); ONE hybrid B for all windows (tmp/data's): the background does not depend on the window, and F2 / E2 have two near-tied NLL optima (α 0.95 s² 12.6 vs α 1 s² 0.001, ΔNLL 0.003) that flip with the obs-noise draw (own tuning kept as hybrid_B_seed42_own.json) | Pilot: "try longer window too"; isolate the window effect |
