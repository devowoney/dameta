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

## Milestone 4 — E2-B / E4-B across all configurations (locked C1, 3 seeds, 2026-11-13)
Protocol: locked C1 config (no per-configuration retuning); teacher = Adagrad with the E1/E3-selected lr (0.3 for
emulator strong runs, else 0.1); one learner per dynamics x constraint x obs pattern, meta-trained on train ICs with
that dynamics (emulator J only for E4-B), best epoch on val, frozen on test / test_ood. Cost-to-target and skill vs
classical Adagrad ON THE SAME DYNAMICS (E1 for E2-B, E3 for E4-B); forecasts always with the true model M1.
Gap closure = median(fc_E3 − fc_E4) / median(fc_E3 − fc_E1), fc RMSE at 1 TU, per-case pairing.
Source: src/report_meta_b.py -> tmp/results/meta_b_summary_{test,test_ood}.{md,csv}.

### Test (200 held-out ICs; Meta-B mean ± sd over 3 seeds)
| dynamics | constraint / obs | evals classical → Meta-B | speed-up | reached cls / meta | fc@1TU classical → Meta-B | Δ fc | basin worse / better | E3−E1 gap | closure |
|---|---|---|---|---|---|---|---|---|---|
| M1 (E2-B) | strong / full | 70 → 9.5 ± 0.5 | 7.4× | 0.92 / 0.97 | 0.542 → 0.595 | +9.6 % | 6 / 1 | — | — |
| M1 (E2-B) | strong / x_only | 40 → 6.7 ± 1.2 | 6.0× | 0.97 / 0.98 | 1.34 → 1.36 | +1.2 % | 2 / 2 | — | — |
| M1 (E2-B) | weak / full | 138 → 21.7 ± 1.5 | 6.4× | 0.99 / 0.92 | 0.558 → 0.692 | +24 % | 17 / 2 | — | — |
| M1 (E2-B) | weak / x_only | 50 → 8.3 ± 0.6 | 6.0× | 0.98 / 0.94 | 1.35 → 1.54 | +14 % | 10 / 0 | — | — |
| xz095 (E4-B) | strong / full | 29.5 → 6.0 ± 5.3 | 4.9×* | 1.00 / 0.57 | 3.10 → 3.56 | +15 % | 79 / 0 | 2.09 | −0.05 ± 0.04 |
| xz095 (E4-B) | strong / x_only | 14 → 3.3 ± 2.9 | 4.2×* | 1.00 / 0.65 | 3.47 → 3.63 | +4.7 % | 64 / 0 | 1.56 | −0.04 ± 0.04 |
| xz095 (E4-B) | weak / full | 155 → 20.2 ± 4.3 | 7.7× | 0.96 / 0.80 | 3.46 → 3.40 | −1.8 % | 10 / 2 | 2.10 | +0.005 ± 0.007 |
| xz095 (E4-B) | weak / x_only | 76 → 7.0 ± 0.0 | 10.9× | 0.96 / 0.96 | 3.45 → 3.57 | +3.2 % | 7 / 3 | 1.45 | −0.001 ± 0.000 |
| xz090 (E4-B) | strong / full | 46 → 11.0 ± 1.0 | 4.2× | 0.99 / 0.94 | 5.02 → 5.13 | +2.2 % | 9 / 1 | 3.77 | −0.000 ± 0.000 |
| xz090 (E4-B) | weak / full | 250 → 17.8 ± 0.8 | 14.0× | 0.94 / 0.94 | 5.29 → 5.19 | −1.9 % | 7 / 6 | 3.85 | −0.001 ± 0.000 |
| clean (E4-B, control) | strong / full | 73 → 9.5 ± 0.5 | 7.7× | 0.93 / 0.94 | 0.69 → 0.78 | +12 % | 10 / 0 | 0.035 | (gap ≈ 0) |
\* median over the cases that reach the target; only 57–65 % do, so the speed-up is optimistic there.
test_ood (same order): speed-ups 4.2 / 5.1 / 5.6 / 6.1 (E2-B) and 2.3 / 2.8 / 5.7 / 6.6 / 2.2 / 8.3 / 4.2 (E4-B);
Δ fc +12 % to +66 % (E2-B) and +26 % to +130 % (E4-B); no configuration passes the gate.

### Findings
- **Cost:** Meta-B reduces gradient evaluations 4–14× on held-out ICs in every configuration (2–8× out of
  distribution). The formal gate (≥ 10× and Δ fc ≤ 5 % vs classical on the same dynamics) is met only by
  xz095 weak/x_only (10.9×, +3.2 %) and xz090 weak/full (14×, −1.9 %), on test only.
- **No re-learning of missing dynamics:** the gap-closure ratio is ≈ 0 wherever the emulator gap is large
  (xz090: −0.0003 / −0.001; xz095 weak: ±0.005), and negative where basin switching dominates (xz095 strong ≈ −0.05;
  test_ood down to −1.1). This is structural: Meta-B only rescales the teacher's steps on the EMULATOR cost, so it
  converges to the same emulator-4D-Var minimum (true-model cost ratio P1 E4/E1 ≈ E3/E1, e.g. 4.3 vs 4.3 xz090
  strong); nothing in its inputs or its training loss carries information about the true dynamics. The North Star
  claim "the meta learner can re-learn missing dynamics" is NOT supported for Meta-B (delta).
- **Universal failure mode = basin switching:** the forecast penalty tracks the number of cases ending in a worse
  local minimum (E2-B weak/full 17, xz095 strong 79 vs ≤ 10 elsewhere). It is worst when the teacher step is already
  large (Adagrad lr 0.3 for emulator strong runs: × up to 11 stretch), where 35–43 % of cases never reach the 1 %
  target — the locked config, tuned with lr 0.1, does not transfer to a 3× larger teacher step.
- **Configurations without basin switching keep skill:** E2-B strong/x_only (+1.2 %), E4-B weak and xz090 strong
  (−1.9 % to +3.2 %) — there Meta-B is a pure accelerator of the classical solution.
- Out of distribution, speed-ups shrink and basin switches grow in every configuration (generalization limit of a
  state-dependent preconditioner learned on the training region).

## Meta-B supervised variant at full scale (2026-11-13)
Setup: G_θ(running state, emb(k)) → running gradient (MLP 3 × 128, SiLU), trained by MSE on all teacher pairs of the
1000 training ICs (classical Adagrad run, same lr/stopping: 567 186 pairs M1, 314 553 pairs xz095), best epoch on
val-IC pairs, 3 seeds; frozen inference with the same Adagrad teacher fed ĝ (no gradient evaluation; one forward
run per iteration for J). Configs: strong/full on M1 (E2-B) and xz095 (E4-B). src/report_meta_b.py --variant
supervised -> tmp/results/meta_b_sup_summary_{test,test_ood}.md.

| split | config | reached 1 % (cls / sup) | IC error (cls / sup; background 0.87) | fc@1TU (cls / sup) | cos(ĝ, ∇J) | worse basin |
|---|---|---|---|---|---|---|
| test | M1 strong/full | 0.92 / 0.03 | 0.43 / 3.55 | 0.54 / 7.09 | −0.55 | 196 / 200 |
| test | xz095 strong/full | 1.00 / 0.11 | 0.77 / 7.86 | 3.10 / 9.03 | −0.53 | 192 / 200 |
| test_ood | M1 strong/full | 0.83 / 0.13 | 0.49 / 3.49 | 0.98 / 10.2 | −0.39 | 187 / 200 |
| test_ood | xz095 strong/full | 0.99 / 0.15 | 0.68 / 10.0 | 2.20 / 10.8 | −0.58 | 192 / 200 |

- **Negative result, by construction:** training MSE stays at ≈ 1.0 in units of the gradient std (= predicting the
  mean; 0.96 after 50 epochs by memorisation) and validation MSE rises (1.24 → 1.28); validation cos(ĝ, g) ≈ 0.
  The 4D-Var gradient depends on each case's observations and background, which the model does not see, so the
  target is not a function of its inputs (state, iteration) — the identifiability concern raised at design time.
- At inference the predicted gradient is anti-aligned with the true one (cos ≈ −0.5): the iterate drifts away from
  the data, analysis errors grow to 4–10× the background error, and almost no case reaches the cost target.
- Contrast with the delta variant: the same imitation idea works when S receives the running gradient (which carries
  the observation information) and only rescales the teacher step. Any supervised variant needs case information
  (gradient, or observations / background) among its inputs.

## E3 and E4-B with the new main emulator rho275 (L63 with ρ = 27.5; 2026-11-14)
Emulator: trained and quality-checked on branch m3-emulator (memory/branches/m3-emulator-v2/findings.md). E3 lr by
the pre-registered rule on val (Adagrad 0.3 strong / 0.1 weak, GD 3·10⁻⁴; edges extended, interior). E4-B = locked C1
Meta-B, 3 seeds, teacher = selected Adagrad lr. Sources: tmp/results/e3_rho275/compare_e1.md (src/report_e3.py),
tmp/results/meta_b_summary_{test,test_ood}.md (src/report_meta_b.py).

### E3 (classical Adagrad) with rho275 vs E1 and vs the old xz095 emulator (test, medians)
| config | P1 J_M1: E3 rho275 / xz095 (E1) | IC RMSE: rho275 / xz095 (E1) | fc@1TU (M1 forecast): rho275 / xz095 (E1) | paired gap E3−E1 @1TU: rho275 / xz095 | evals to 1 % |
|---|---|---|---|---|---|
| strong / full | 21.8 / 31.9 (16.5) | 0.534 / 0.771 (0.429) | 1.95 / 3.10 (0.542) | **1.16 / 2.09** | 23 |
| strong / x_only | 6.50 (5.43) | 0.644 (0.608) | 2.20 (1.34) | 0.48 / 1.56 | 8 |
| weak / full | 21.5 (16.4) | 0.529 (0.428) | 2.07 (0.558) | 1.15 / 2.10 | 140 |
| weak / x_only | 6.47 (5.36) | 0.657 (0.601) | 2.17 (1.35) | 0.48 / 1.45 | 47 |
The parameter-biased emulator roughly halves the emulator penalty (gap 1.16 vs 2.09 strong/full; 0.48 vs 1.5 x_only)
while keeping a clear, measurable gap for the gap-closure analysis.

### E4-B (Meta-B) with rho275 (test; test_ood in brackets)
| config | evals E3 → Meta-B | speed-up | reached E3 / Meta-B | Δ fc vs E3 | basin worse / better | gap closure |
|---|---|---|---|---|---|---|
| strong / full | 23 → 11.2 ± 2.9 | 2.1× [1.4×] | 0.99 / 0.82 | +12 % [+99 %] | 32 / 1 | −0.003 [−0.16] |
| strong / x_only | 8 → 5.3 ± 0.6 | 1.5× [1.9×] | 0.99 / 0.97 | +1.2 % [+7 %] | 4 / 0 | −0.000 [−0.000] |
| weak / full | 139 → 18.8 ± 2.0 | 7.4× [5.4×] | 0.96 / 0.91 | +3.1 % [+54 %] | 16 / 3 | −0.003 [−0.005] |
| weak / x_only | 46 → 6.7 ± 0.6 | 6.9× [5.6×] | 0.98 / 0.96 | +9.4 % [+10 %] | 9 / 0 | −0.002 [−0.001] |
- Conclusions of the xz runs hold with the milder, parameter-biased emulator: Meta-B accelerates emulator 4D-Var
  (up to 7× weak; only 1.5–2× strong, where the classical teacher already uses lr 0.3 and needs 8–23 evaluations)
  but does not close the emulator gap (closure −0.003 to 0; P1 E4/E1 ≈ E3/E1, 1.06–1.27 vs 1.06–1.16).
- Basin switching remains the failure mode with the lr-0.3 teacher (strong/full: 32 worse cases, 82 % reach target),
  less severe than with xz095 (79 worse, 57 %). No configuration passes the gate.
