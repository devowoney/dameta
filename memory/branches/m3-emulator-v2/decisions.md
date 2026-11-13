# Branch m3-emulator (revision v2, after the M3 fold) — decisions (same columns as memory/decisions.md)

| Date | Decision | Rationale |
|---|---|---|
| 2026-11-13 | Replace the removed-physics degradation (xz_scale 0.95 / 0.90) by a biased model PARAMETER; candidates screened vs M1 on VAL ICs (src/screen_teachers.py): rho 27.5 MAIN candidate, sigma 10.5 milder alternative; pilot to confirm before E3/E4 reruns | Pilot: xz095 changes the dynamics too much (one-obs-interval model error 0.36-0.84, comparable to sigma_o 1.41; E3-E1 gap 2.09). rho 27.5: 0.12-0.28 (1/3), sigma 10.5: 0.05-0.12 |
| 2026-11-13 | Emulator architecture and training unchanged (MLP tendency + RK4, multi-step rollout loss with 1 -> 10 step curriculum, 10 000 steps, best val rollout checkpoint) | Already a rollout model; keeps comparability with the clean / xz emulators |
| 2026-11-13 | Emulator quality gate = 3 checks on test ICs vs M1 and vs own teacher (src/check_emulator.py): 1 same IC, 2 perturbed pair (1e-3), 3 100-member ensembles (a: ICs at different times, 20 TU climatology; b: sigma0 0.1 perturbed spread growth) | Pilot request 2026-11-13 |
