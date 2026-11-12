# Branch m4-meta-b — findings (paper-ready; fold into memory/findings.md after merge)

## Milestone 4 — E2-B: Meta-B (Δ inner-loss delta chain) on the physical model, strong constraint, full obs (2026-11-12)

### Methods (paper-ready draft)
Meta-B is the L63 port of the Δ inner-loss learned optimizer. At outer iteration k one gradient evaluation (forward +
adjoint run of M1) gives J(c^k) and ∇J(c^k); the classical optimizer (teacher: Adagrad, lr 0.1, the E1 setting)
proposes the step f(c^k). A network S_θ — an MLP (2 × 64, SiLU) fed the standardized running state, the running
gradient divided by its per-case RMS, and an embedding of log(1 + k) — outputs a per-component stretch
s ∈ [−10, 10] (10·tanh bound). It is applied K = 20 times without any model call:
c_{m+1} = c_m − (f/K) ⊙ max(1 + s_θ(c_m, ∇J, k), 0); with s = 0 the chain is exactly one teacher step (verified to
1e−13), and S can rescale but never reverse the teacher step. θ is meta-trained online on the 1000 training ICs
(episodes of 100 cases × 100 outer iterations, 3 epochs, 10 Adam steps per outer iteration, meta lr 1e−2) with
2 ALIGN steps (l_align = mean s²), 2 TRANS steps and 6 PERF steps (l_perf = −⟨∇J(c^k), c^{k−1} − c_K(θ)⟩ /
⟨∇J(c^{k−1}), f(c^{k−1})⟩, weight 2.5; its θ-gradient equals dJ(c_K(θ))/dθ exactly, verified to 3e−16), plus
l_reg = 10^{−4} mean θ². The epoch with the lowest median J after 10 frozen iterations on validation ICs is kept.
At test time θ is frozen; the stopping rule and cost accounting are those of the classical runs (cost = gradient
evaluations; the chain costs 20 small-MLP evaluations per iteration, negligible against a model + adjoint run).
Hyper-parameters were selected on validation ICs only (4 sweeps, 54 runs, pre-registered rule: reach the 1 % target
at least as often as Adagrad, forecast RMSE at 1 TU within 5 % of Adagrad, then fewest evaluations to the target;
3 training seeds). Cost-to-target: per case, J* = best final J of the classical E1 runs (GD, Adagrad); evaluations
until J ≤ 1.01 J*, median over cases that reach it.

### Results (3 training seeds, mean ± std over seeds; medians over 200 ICs per split)
| split | method | evals to 1 % target | reached 1 % | evals to 5 % | rmse x0 | fc RMSE 1 TU | fc RMSE 2 TU | J_final |
|---|---|---|---|---|---|---|---|---|
| val (tuning) | E1 Adagrad | 108 | 0.925 | 34 | 0.458 | 0.656 | 1.81 | 16.5 |
| val (tuning) | **Meta-B** | **11.7 ± 0.6** | 0.970 | 4 | ~0.45 | 0.665 ± 0.011 | ~1.8 | 16.5 |
| test | E1 Adagrad | 70 | 0.915 | 30 | 0.429 | 0.542 | 1.56 | 16.48 |
| test | E1 GD | 261 | 0.915 | 76 | 0.429 | 0.704 | 1.95 | 16.99 |
| test | **Meta-B** | **9.5 ± 0.5** | 0.967 | 4 | 0.419 | 0.595 ± 0.000 | 1.75 | 16.51 |
| test_ood | E1 Adagrad | 125 | 0.835 | 37 | 0.494 | 0.975 | 1.57 | 17.15 |
| test_ood | E1 GD | 219 | 0.820 | 38 | 0.517 | 1.939 | 2.63 | 18.27 |
| test_ood | **Meta-B** | **29.5 ± 0.5** | 0.898 | 8 | 0.460 | 1.141 ± 0.027 | 2.01 | 17.52 |
Records: tmp/results/e2b/*_{test,test_ood}.nc; tables tmp/results/e2b/sweep_summary_{test,test_ood}.md; learners
tmp/meta/e2b_sweep4/*_C1_wap0_s{0,1,2}.pt; val sweeps tmp/results/e2b_sweep{,2,3,4}.

### Findings
- **Cost:** Meta-B reaches the 1 % cost target with 7.4× fewer gradient evaluations than Adagrad on held-out test
  ICs (9.5 vs 70; 27× fewer than GD) and 4.2× fewer on out-of-distribution ICs (29.5 vs 125), and reaches it in more
  cases (0.97 vs 0.915 test; 0.90 vs 0.835 test_ood). Seed spread is small (± 0.5 eval).
- **Gate NOT met on test:** the North Star gate (≤ 1/10 of Adagrad evaluations ≈ 7 on test, AND forecast RMSE within
  5 % of classical) fails on both counts: 9.5 evals, and forecast RMSE at 1 TU is +9.7 % on test (0.595 vs 0.542)
  and +17 % on test_ood (1.14 vs 0.975), although val showed parity (+1.4 %).
- **Analysis accuracy vs forecast:** the initial-condition error is slightly LOWER than Adagrad (0.419 vs 0.429 test;
  0.460 vs 0.494 test_ood) and the final cost is equal on test (16.51 vs 16.48), yet the forecast beyond the window is
  worse. Hypothesis (not yet tested): the stretched steps and earlier stopping leave a different error in the
  unstable directions, which dominate forecast growth — to be checked with P3 (projection of the analysis
  increment on the leading backward Lyapunov vectors).
- **Generalization:** the speed-up shrinks out of distribution (7.4× → 4.2×) and the skill penalty grows
  (+9.7 % → +17 %), consistent with S having learned a state-dependent preconditioner of the training region.
- **Tuning insight (val):** bice's default curriculum (6 of 10 meta-steps ALIGN) keeps S at ~1.4× the teacher step
  (2× speed-up). Reducing ALIGN to 2 gives stretches of 5–8× (9× speed-up). With no ALIGN or a looser bound, S
  saturates at the bound and skill degrades; a residual l_align in PERF does not prevent this, best-epoch selection
  on validation does. Cause: l_perf is linear in the step length, so overshoot is only penalised after the gradient
  flips.
- **Supervised variant** (state + emb(k) → running gradient): smoke only (diverges, cos(ĝ, ∇J) ≈ −0.5); not evaluated.

### Verification
tests/test_m4.py 19/19 (s = 0 chain = teacher step; frozen s = 0 = classical Adagrad; exact l_perf chain rule;
stretch bound / no sign flip; case independence; supervised loop with oracle gradient = classical); M2 8/8, M3 5/5;
run_4dvar refactor bit-identical to saved E1 record; sweep reproducibility (a repeated config matched to all digits).
