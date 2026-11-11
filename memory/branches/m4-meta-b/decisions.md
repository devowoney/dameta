# Branch m4-meta-b — decisions (same columns as memory/decisions.md; fold on main after merge)

| Date | Decision | Rationale |
|---|---|---|
| 2026-11-11 | Reuse M2 `da/reference_step.py` (already the bice ReferenceStep port) as the Meta-B teacher; no second port | Same object = E1/E3 classical optimizer and teacher, so s = 0 reproduces E1 exactly (tested) |
| 2026-11-11 | S_theta = MLP (2 x 64 SiLU) on [x_std, g_std, emb(k)] -> s in R^d (d = 3 strong, 33 weak); bice UNet dropped | L63 has no spatial dims; mechanism (stretch, bound, chain) unchanged |
| 2026-11-11 | output_scale = 5 (|s| <= 5, per-component stretch in [0, 6]); K = 20; num_meta_steps 10, ALIGN 6 / TRANS 2 / PERF 2; w_align 1, w_perf_delta 2.5, meta_lr 1e-3, clip 1, Adam no weight decay | bice delta-mode defaults (output_scale ~5 recommended by bice for this mode) [ASSUMPTION: pilot to confirm before E2-B sweeps] |
| 2026-11-11 | Gradient input g_std = g / RMS(g) per case and per control block (x0, eta); no mean subtraction | bice divides by per-channel std over space; L63 has 3 components, and per-case scaling keeps cases independent in frozen inference (tested) |
| 2026-11-11 | State input x_std = (c - mean) / std: x0 block from D1 train truth climatology, eta block 0 / sigma_q | Analogue of bice normalizer.mean/std; train split only (no leakage) |
| 2026-11-11 | Iteration embedding on log1p(k) instead of raw k (flag `log_k`) | Outer loop runs up to 1000 iterations; raw k saturates the embedding MLP |
| 2026-11-11 | S head zero-initialised (flag `zero_init`): s = 0 at start, Meta-B starts as the teacher | Exact imitation baseline at theta_0; ALIGN pulls s to 0 in bice anyway |
| 2026-11-11 | Batched cases: l_perf normalised per case by <g^{k-1}, f^{k-1}> then averaged; l_align averaged over cases | Each IC is an independent 4D-Var problem; per-case normalisation keeps cases on the same scale |
| 2026-11-11 | l_reg = lambda (lambda sum theta^2 / n) kept as in bice (flag `reg_square_lambda`, default True) | Faithful port; findings flagged lambda^2 as possibly unintended -> pilot to confirm |
| 2026-11-11 | No reporting forward call per outer iteration (bice does one); J(c^{k+1}) comes from the next gradient evaluation | Same cost accounting and stopping rule as run_fourdvar (n_grad_evals comparable) |
| 2026-11-11 | Main protocol implemented as `train_meta_b`: episodes = batch of train ICs x episode_iters outer iterations with online delta updates (theta + Adam state persist), then frozen inference (`run_meta_b(train=False)`) | Pilot protocol 2026-11-10 (train across train ICs, frozen on test) |
| 2026-11-11 | Supervised variant (S(x0^k, emb(k)) -> f, MSE, no gradient input) NOT implemented yet | Open question for pilot: without g or y, S cannot locate a case-specific minimum (see progress) |
