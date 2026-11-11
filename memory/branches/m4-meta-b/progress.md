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
