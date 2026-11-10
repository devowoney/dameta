# Branch m2-classical-4dvar — decisions (same columns as memory/decisions.md; fold on main after merge)

| Date | Decision | Rationale |
|---|---|---|
| 2026-11-10 | Weak-constraint control = (x0, eta_1..eta_10), eta added to the state at obs times t_i, i >= 1 | Approved DA config v0 |
| 2026-11-10 | Cost unit = 1 gradient evaluation (forward + adjoint run per case); per-case stopping on relative dJ < tol | Fair per-case iteration count in a batched run |
| 2026-11-10 | lr selection rule (fixed before results): min median final J on val; exclude lr with >1% diverged; tie-break median n_grad_evals | Truth-free like 4D-Var; pre-registered to avoid tuning bias |
| 2026-11-10 | lr sweep on first 50 val cases (CPU, 4 cores shared); chosen lr re-run on full val/test/test_ood | Login node limits; SLURM needs pilot OK |
| 2026-11-10 | Final lr: Adagrad 0.1 (all); GD 0.001 (strong), 3e-4 (weak) | Pre-registered rule on val sweep |
| 2026-11-10 | Cost metric = gradient evaluations to reach J <= (1+tol) J*, tol 1 % and 5 % (J* = best final J per case) | Curves plateau before the 1e-6 stopping rule |
| 2026-11-10 | North Star cost gate: report vs BOTH GD and Adagrad; the claim (<= 1/10 evals) is anchored on Adagrad, the strongest classical baseline (target ~7 evals to 1 % of J*, strong/full test) | Pilot decision; reviewers expect the strongest baseline. FOLD INTO original.md North Star on main after merge |
