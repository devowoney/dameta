# Progress — branch m3-emulator, revision v2 (new parameter-biased emulator)

- 2026-11-13: Pilot: xz emulators remove too much physics; use a biased model parameter instead; emulator must work
  as a rollout model; add 3 quality checks. Worked in worktree .claude/worktrees/m3-emulator (old m3 notes are
  folded/closed -> this folder). Teacher screen on val (tmp/results/emulator/teacher_screen.md): rho 27.5 and
  sigma 10.5 chosen. Trained rho275 (best val rollout rmse@50 0.027, 637 s) and sig105 (0.029, 681 s) with the
  M3 recipe (10k steps). Quality checks (tmp/results/emulator/emulator_qc.md, *_qc.json; figures
  tmp/figs/m3_emuqc_<name>_{1_same_ic,2_pair,3_ensemble}) for rho275, sig105, clean, xz095.
  Next (pilot): choose rho275 vs sig105, then rerun E3 (lr re-selection on val) and E4-B with the new emulator.
