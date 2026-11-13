# Progress — branch m4-meta-b (Milestone 4: Meta-B port, E2-B, E4-B)

- 2026-11-11: Worktree created from main 1df3c83 (after m3-emulator merge) at .claude/worktrees/m4-meta-b.
- 2026-11-11: Inputs copied from the m3 worktree into tmp/: data (D1), emulators (clean/xz095/xz090), report,
  results/{e1, e3_clean, e3_xz095, e3_xz090, emulator} (lr sweeps not copied; selected lrs are in decisions).
- 2026-11-11: Ported bice delta chain (branch meta-learner-one-task, read via git show; bice untouched):
  src/dameta/meta/networks.py (StretchNet, IterationEmbedding), src/dameta/meta/meta_b.py (MetaB: chain,
  online delta meta-update ALIGN/TRANS/PERF, advance, checkpoint; run_meta_b = outer loop with run_fourdvar's
  stopping rule and record + step_ratio; train_meta_b = episodes over train ICs). ReferenceStep reused from M2.
  tests/test_m4.py 12/12 pass (s=0 chain == teacher step, frozen s=0 == run_fourdvar Adagrad, l_perf theta-grad
  == dJ(c_K)/dtheta / denom to 3e-16, stretch bound / no sign flip, case independence, training smoke, checkpoint).
  M2 tests 8/8 still pass. Smoke (32 train cases, 2 x 20 iters, Adagrad): step_ratio ~1.29, L_perf_applied ~-1.1.
  Next: Hydra runner run_meta_b.py + config, then meta-training (strong M1) and E2-B val check.
  Open for pilot: bice hyper-parameter defaults; lambda^2 in l_reg; supervised variant design.
- 2026-11-11: Pilot: hyper-parameters adjustable; lambda^2 is a bug (fixed: l_reg = lambda mean(theta^2));
  supervised variant = G(running state, emb(k)) -> running gradient. Implemented meta/meta_b_sup.py (teacher pairs
  via run_fourdvar on_step hook, GradNet fit with best-val epoch, frozen loop fed g_hat, n_fwd_evals cost),
  GradNet + per-sample emb(k) in networks.py, run_4dvar.analysis_record (refactor bit-identical vs saved E1),
  src/run_meta_b.py + configs/meta_b.yaml (variant delta|supervised, every hyper-parameter exposed, checkpoint
  to tmp/meta/<group>, records to tmp/results/<group>). Tests: M4 19/19, M2 8/8, M3 5/5.
  Runner smoke (tmp/results/smoke, tmp/meta/smoke; NOT results): delta strong 32 train ICs x 2 short episodes ->
  0/16 diverged, 10/16 hit the 300-iter cap (episodes only cover k < 20 -> episode_iters must cover the teacher's
  convergence length); supervised weak 3 epochs -> J blows up (median J_final 2.6e4), as expected without obs/grad.
  Next: delta meta-training strong/full M1 on all train ICs + val sweep (episode_iters, w_perf_delta, output_scale).
- 2026-11-11: E2-B delta val sweep launched (12 configs, 4 single-thread workers, src/scripts/sweep_e2b_delta.sh;
  logs tmp/logs/e2b_sweep_*.log, records tmp/results/e2b_sweep, learners tmp/meta/e2b_sweep). Report: src/report_e2b.py.
- 2026-11-12: E2-B delta val sweep done (12/12, 0 diverged; tmp/results/e2b_sweep/sweep_summary.{csv,md}).
  All 12 eligible. Evals to 1% target (median): 53-71 vs E1 Adagrad 108 (GD 252); reached 0.965-0.985 vs 0.925;
  skill unchanged (fc@1TU 0.649-0.666 vs 0.656; J_final 16.4 vs 16.5). Selected (pre-registered rule):
  ep300_wp2.5_os5 (53 evals, reached 0.97). Gate (<= 10.8 evals) NOT met (~5x off): speed-up ~2x only.
  w_perf_delta 1 weaker (68-71); output_scale 2 vs 5 no effect (median stretch ~1.3-1.4, bound never active);
  ep300 slightly better than ep100 (53-58 vs 59-62) at 3x train cost (~19 min vs ~6.5 min). Single training
  seed: differences between top configs likely within seed noise. report_e2b.py uses report_e1.to_markdown
  (tabulate not installed).
- 2026-11-12: Pilot: close the gate gap. Sweep v2 stage A launched (12 configs, src/scripts/sweep_e2b_delta_v2.sh,
  tmp/results/e2b_sweep2, logs tmp/logs/e2b_sweep2_*). report_e2b.py now lists all swept params + seed aggregation.
- 2026-11-12: Sweep v2 stage A done (12/12, 0 diverged; tmp/results/e2b_sweep2/sweep_summary.*). Reproducibility:
  a6/1e-3/os5 == sweep-1 ep100_wp2.5_os5 row exactly (62 evals). ALIGN is the bottleneck: a6 45-65 evals
  (stretch ~1.4), a2 11.5-19.5 (stretch 4-6, fc@1TU 0.645 = E1 skill), a0 unstable (stretch pinned at 1+os;
  os5 stalls: reached 0.24, fc 0.86; os10 fc 0.712 > 1.05 x E1 -> ineligible). Best single run a2/mlr1e-2/os10:
  11.5 evals to 1% target (gate 10.8), reached 0.97, fc 0.646. Stage B launched (18 runs: C1 a2/1e-2/os10,
  C2 a2/1e-3/os10, C3 a2/1e-2/os20 x epochs {1,3} x seeds {0,1,2}; tmp/results/e2b_sweep3).
- 2026-11-12: Stage B done (18/18, 0 diverged; tmp/results/e2b_sweep3/sweep_summary.md + sweep_seed_summary.csv).
  Seed means (3 seeds, val, evals to 1% target / reached / fc@1TU; E1 Adagrad 108 / 0.925 / 0.656; gate 10.8):
    C1 os10 mlr1e-2 ep3: 11.3 +- 0.6 / 0.967 / 0.675 +- 0.035  -> SELECTED by rule (gate missed by 0.5 eval)
    C1 os10 mlr1e-2 ep1: 13.3 +- 2.8 / 0.970 / 0.655 +- 0.014  (most robust skill)
    C3 os20 ep1 11.7 (fc 0.690, ineligible); C2 mlr1e-3 and C3 ep3 ineligible (seed collapses).
  Failure mode: with longer training / larger bound S drifts to s = +output_scale everywhere (step_ratio pinned at
  1 + os; C1-ep3-s1 11.0, C3-ep3-s1/s2 21.0 -> identical trajectories, seed-independent) and skill degrades
  (fc 0.71-0.79). Cause: l_perf is linear in the step, so it rewards longer steps until the gradient flips;
  overshoot is only penalised after it happens, and with ALIGN 2/10 nothing pulls s back.
  Speed-up vs Adagrad now ~9.5x (11.3 vs 108), from ~2x in sweep 1.
- 2026-11-12: Pilot accepted anti-saturation tuning. Implemented w_align_perf (MetaBConfig) and train_meta_b
  val_score / best-epoch restore; runner keys delta.{w_align_perf, select_epoch, select_k, select_n_val}.
  Tests 19/19; selection smoke OK. Sweep v4 launched (12 runs; tmp/results/e2b_sweep4).
- 2026-11-12: Sweep v4 done (12/12, 0 diverged; tmp/results/e2b_sweep4/sweep_summary.md, sweep_seed_summary.csv).
  Seed means (evals to 1% / reached / fc@1TU):  wap0 + selection 11.7 +- 0.6 / 0.970 / 0.665 +- 0.011 (SELECTED);
  wap0.05 12.3 +- 2.1 / 0.960 / 0.670 +- 0.037; wap0.1 13.2 +- 1.0 / 0.960 / 0.673 +- 0.047; wap0.2 16.0 +- 5.6 /
  0.918 (ineligible). Residual l_align does NOT prevent saturation (wap0.05-s2 ratio 10.4 fc 0.71; wap0.1-s1 ratio
  11.0 fc 0.73) and slows convergence -> rejected. Best-epoch selection alone fixes it: wap0 best epochs (2, 1, 1),
  all seeds stretch ~7.2, fc 0.652-0.671 (stage B C1-ep3 fc std 0.035 -> 0.011; s0 identical to stage B since
  epoch 2 = last). Gate (10.8) still missed by ~1 eval: ~9.3x vs Adagrad (per seed 11 / 12 / 12).
  Caveat: epoch score uses the first 100 val ICs, also part of the val evaluation (mild optimism; test is clean).
- 2026-11-12: Pilot locked C1 (sweep v4 wap0) and asked for test. Frozen inference of the 3 v4 seed learners on
  test / test_ood (src/scripts/test_e2b_delta.sh; tmp/results/e2b; report_e2b.py --split). Test: 9.5 +- 0.5 evals
  to 1% vs Adagrad 70 (7.4x), reached 0.967 vs 0.915, fc@1TU 0.595 vs 0.542 (+9.7%) -> gate NOT met (needs ~7 evals
  and <= +5%). test_ood: 29.5 vs 125 (4.2x), reached 0.898 vs 0.835, fc@1TU 1.14 vs 0.975 (+17%). rmse_x0 slightly
  better than Adagrad on both. Paper-ready entry: memory/branches/m4-meta-b/findings.md.
- 2026-11-12: Pilot asked for P3 + G2. New src/dameta/eval/lyapunov.py (spin-up replay, bit-identical; QR BLVs) and
  src/report_p3g2.py -> tmp/results/e2b/p3g2_*.md. Result: forecast gap = basin switching in a few cases (test: 6
  worse / 1 better of 200; ood 16-17 / 5-7); same-basin cases have identical skill. P3 = Adagrad on test, lower on ood;
  G2 cos ~ Adagrad; 3 % of Meta-B steps increase J (overshoot) vs 0.02 %. Findings appended to branch findings.md.
- 2026-11-12: Extension launched: 30 jobs (10 configs x 3 seeds; src/scripts/meta_b_job.sh, xargs -P 4, ~4.5 h);
  records tmp/results/{e2b, e4b_xz095, e4b_xz090, e4b_clean}, learners tmp/meta/<group>, logs tmp/logs/<group>_*.
  Smokes OK (weak/x_only M1; emulator xz095).
- 2026-11-13: Extension interrupted by session end after 18/30 jobs (E2-B 3 configs; xz095 strong/full, strong/x_only,
  weak/full). Found 3 bad records: xz095 weak/x_only s0 val (interrupted write) and xz095 weak/full s2 test/test_ood
  (NetCDF HDF write error at 18:46/18:54 — /Odyssey shared FS at 100 %, 256 GB free, down from 727 GB on 11-12).
  meta_b_job.sh now resumes from a loadable checkpoint (rebuilds val/test/test_ood without retraining). Relaunched
  13 jobs (xz095 weak/full s2 rebuild, xz095 weak/x_only, xz090 strong/weak full, clean strong/full).
- 2026-11-13: Extension complete (30/30 jobs + 3 rebuilt records; 96 records verified readable; log "Logging error"
  tracebacks = log-flush failures during the FS incident, runs completed). report_meta_b.py now skips unreadable
  records with a warning. Results: speed-up 4-14x test (2-8x ood); gate met only xz095 weak/x_only and xz090 weak/full
  (test); gap closure ~0 everywhere (Meta-B converges to the emulator minimum: structural); basin switching drives the
  skill penalty, worst with the lr-0.3 teacher (xz095 strong: 79/200 worse basins, 57 % reach target).
  Findings section appended to branch findings.md.
- 2026-11-13: Pilot: supervised variant at full scale. meta_b_job.sh gained a variant argument (7th, default delta;
  delta naming unchanged); report_meta_b.py --variant supervised. 6 jobs (M1 + xz095 strong/full x 3 seeds), 18
  records OK. Result: fails by construction — train MSE ~1.0 (= mean), val cos ~0; at inference cos(g_hat, grad)
  ~ -0.5, reached 3-15 %, IC error 3.5-10 (background 0.87). Findings appended.
