# Branch m3-emulator — decisions (same columns as memory/decisions.md; fold on main after merge)

| Date | Decision | Rationale |
|---|---|---|
| 2026-11-10 | Deliberate degradation implemented in the emulator's TRAINING TEACHER (biased sigma/rho/beta, xz_scale on the -x*z term of dy/dt), trajectories from the same D1 train ICs; network and D1 untouched | Controlled, reproducible model error; keeps the pilot rule "same trajectories" (same ICs) |
| 2026-11-10 | E3 analyses judged with the true model: J_true_final (P1) and rmse_fc_m1 (forecast with M1 from the analysis); rmse_fc = forecast with the emulator | Separate IC quality from emulator forecast error; P1 defined in North Star |
| 2026-11-10 | Emulator training: rollout loss, curriculum n_roll 1 -> 10, Adam + cosine decay, best-val checkpoint (val = D1 val ICs) | Standard for stable multi-step emulators; test splits never used |
| 2026-11-10 | E3-clean lr = E1 lr (Adagrad 0.1; GD 1e-3 strong / 3e-4 weak), re-selected by the same rule on a 3-point grid (all interior) | Pre-registered rule; emulator does not shift the optimum step |
| 2026-11-10 | Degradation: main = B' (teacher xz_scale = 0.95, emulator 'xz095'); stress test = B (xz_scale = 0.9, emulator 'xz090'); clean emulator = control | Pilot decision; B' model error ~ obs error, B exceeds it |
| 2026-11-10 | Degraded-emulator lr check on full obs only (Adagrad {0.03,0.1,0.3}; GD strong {3e-4,1e-3,3e-3}, weak {1e-4,3e-4,1e-3}); x-only reuses the full-obs choice | x-only lr matched full-obs lr exactly in E1 and E3-clean; CPU budget |
| 2026-11-10 | Degraded-emulator lr: Adagrad 0.3 strong / 0.1 weak; GD 3e-4 (strong grid edges extended to 1.0 and 1e-4 to confirm interior) | Pre-registered rule |
| 2026-11-11 | Keep Q fixed (sigma_q = 0.1, approved DA config v0) for all emulators; no per-emulator Q re-estimation | Pilot decision; mis-specified Q is part of the degraded-emulator test |
| 2026-11-11 | Report statistics: median + 95 % bootstrap CI (2000, seed 0); E3 vs E1 paired by case (Wilcoxon); cost = evals to 1.01 J*, J* = best final J over GD/Adagrad with the same dynamics | One pre-fixed definition for all tables; supersedes the 'own J*' cost used earlier for E3 |
