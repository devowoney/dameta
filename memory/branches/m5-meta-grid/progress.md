# Progress — branch m5-meta-grid (Milestone 5: meta-learner grid)

- 2026-11-14: Worktree created from c0a1db0 (= main after the M4 merge). Inputs copied (1.3 GB). Design recorded in
  decisions.md (6-cell grid). Implementation order: cell 5 (Meta-A) and 7 (shared unroll code), then 2, 3; 4 rerun
  with obs_input.
- 2026-11-14: Built the remaining grid cells.
  * cells 5 / 7 (learned optimization): src/dameta/meta/meta_a.py (LearnedOptimization: K-step unroll, first order
    default, second_order flag, obs_input flag, fit with best val epoch, run_meta_a fixed budget) + FreeUpdateNet.
  * cell 2 (lookahead): src/dameta/meta/meta_b_lookahead.py (teacher-trajectory pairs, m = 5, relative MSE; Meta-B
    chain + run_meta_b inference reused).
  * cell 3 (greedy): src/dameta/meta/meta_greedy.py (online episodes, J(x^{k+1})/J(x^k), stopping rule, n_fwd_evals).
  * cell 4: obs_input flag (GradNet d_obs, pairs carry obs, inference passes obs).
  * run_meta_b.py variants delta | lookahead | greedy | supervised | unroll; configs/meta_b.yaml blocks.
  Tests: test_m5.py 18/18 (zero-init/no-move, call counts, 2nd-order theta-grad == FD 3e-9, case independence, obs
  flag, training smokes, lookahead pairs == classical trajectory, checkpoint); M4 19/19 and M2 8/8 still pass.
  Observation (smoke): cell 3 greedy ratios ~1.0 even on training episodes — with input (x, k) a one-step move
  cannot be case-specific (identifiability); cell 7 smoke still lowers J(x^K)/J(x^0) to 0.94 (common drift).
