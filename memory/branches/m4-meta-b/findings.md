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

### P3 / G2 and the origin of the forecast-skill gap (2026-11-12)
Method: backward Lyapunov vectors by QR iteration of the exact discrete TL model along the 1000-step spin-up preceding
each test IC (reproduced from the seeded generator, bit-identical to x_true(t0); exponents ≈ (0.9, 0, −14.5) in-dist,
(0.74, −0.06, −14.3) on test_ood; invariance |cos(M b1(t), b1(t+1))| = 1) and on along the truth window.
src/report_p3g2.py -> tmp/results/e2b/p3g2_{test,test_ood}.md (+ paired Wilcoxon tests, E2-B seed mean vs Adagrad).

| split | method | P3 (b1) | P3 (b1+b2) | IC error | unstable IC error | end-of-window error | its unstable part | G2 cos(step, −∇J) | G2 first 10 | steps decreasing J (mean) | cases with ≥ 1 increasing step |
|---|---|---|---|---|---|---|---|---|---|---|---|
| test | E1 Adagrad | 0.252 | 0.890 | 0.743 | 0.308 | 0.438 | 0.330 | 0.911 | 0.888 | 0.9998 | 0.03 |
| test | E1 GD | 0.269 | 0.921 | 0.743 | 0.341 | 0.466 | 0.357 | 1 | 1 | — | — |
| test | E2-B | 0.270 | 0.899 | 0.726 | 0.301 | 0.454 | 0.350 | 0.903 | 0.887 | 0.970 | 0.49 |
| test_ood | E1 Adagrad | 0.443 | 0.954 | 0.855 | 0.194 | 0.574 | 0.368 | 0.906 | 0.870 | 0.997 | 0.11 |
| test_ood | E1 GD | 0.309 | 0.939 | 0.895 | 0.223 | 0.784 | 0.573 | 1 | 1 | — | — |
| test_ood | E2-B | 0.357 | 0.931 | 0.797 | 0.209 | 0.657 | 0.460 | 0.935 | 0.846 | 0.958 | 0.63 |
(medians over cases; E2-B = mean of 3 seed medians, seed sd ≤ 0.01 except end-of-window unstable error ood 0.012)

- **The skill gap is a basin-switching tail, not a systematic degradation.** On test, 193/200 cases end in the same
  minimum as Adagrad and their forecast is unchanged (median per-case relative difference +0.15 %; median fc@1TU
  0.567 vs 0.584). In 6 cases Meta-B lands in a worse local minimum (median fc@1TU 3.5 vs 0.19) and in 1 a better one;
  these few cases shift the median fc@1TU by +9.7 %. test_ood: 16–17 worse vs 5–7 better (same cases in all seeds).
  The 1-TU window makes the L63 4D-Var cost multimodal; the 5–8× stretched early steps of Meta-B sometimes jump
  across a ridge that Adagrad does not cross.
- **Unstable-direction hypothesis rejected as the main cause:** P3 on test equals Adagrad (0.270 vs 0.252, p = 0.77)
  and the unstable end-of-window error differs by +0.0005 per case (p = 0.02, negligible). On test_ood P3 is lower
  than Adagrad (0.357 vs 0.443, p = 2e-4), so the North Star condition P3 ≥ classical holds on test but not on
  test_ood.
- **G2:** step direction is as gradient-aligned as Adagrad (cos 0.90 vs 0.91 test; 0.94 vs 0.91 ood), but the
  stretched steps overshoot: 3.0 % of Meta-B steps increase the true cost (Adagrad 0.02 %), in 49 % of test cases at
  least once (Adagrad 3 %). Consistent with l_perf being linear in step length (overshoot penalised only afterwards).
- Implication: the cost/skill trade-off of Meta-B is governed by a few large early steps; a basin-safe step control
  (e.g. bounding the first-iteration stretch, or rejecting a chain move when the next J rises) is the natural fix.
