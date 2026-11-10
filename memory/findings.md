# Findings

## Environment (2026-11-09)
- `environment.yaml` defines conda env `oceanai` (py3.12, torch, lightning, hydra, xarray/netcdf/zarr, CUDA 12.8).
- Local `.venv` (py3.12.13) is essentially empty — science stack not installed there.
- conda/mamba available at /Odyssey/private/j25lee/miniforge3.
- Current login node: no GPU visible (`nvidia-smi`: No devices found).

## Research notes
_See "Related work (B.2)" below._

## Meta-A vs Meta-B — open ambiguities (2026-11-09)
1. Meta-A training loss: J(x0^K) [obs-only, 4D-Var-consistent] vs ||x0^K - x0_true|| [truth-supervised] vs sum_k J(x0^k).
2. Meta-B "unsupervised" meaning undefined (supervised = regress teacher step dx^k).
3. Teacher optimizer for Meta-B (GD / L-BFGS / Gauss-Newton incremental) and K at inference (fixed vs until convergence).
4. Inputs of g_theta (grad J only? + x0^k, J, history/memory?) -> determines whether cost saving = fewer iterations only.
5. Fair comparison: same g_theta architecture for A and B, only training target differs?
6. Meta trained separately per dynamics (M1, M2) or transferred; weak-constraint control vector size.
- (2026-11-09) Resolved 1,2,3,5,6 with pilot (see original.md). Open: g_theta inputs (await reference code);
  watch-point: Meta-B unsupervised (J via chain rule) vs Meta-A (J(x0^K)) — difference to be pinned down from code
  (e.g. per-step vs end-of-unroll loss, GD-structured vs free correction operator).

## Proposed metrics for physical consistency & gradient fidelity (draft, 2026-11-09, awaiting pilot)
Gradient fidelity (reference = exact M1 adjoint gradient, available for L63):
- G1 emulator gradient: cos(gradJ_M2, gradJ_M1) and ||gradJ_M2 - gradJ_M1|| / ||gradJ_M1|| at same x0 (E3/E4).
- G2 meta step direction: cos(dx^k, -gradJ_M1(x^k)) per iteration; fraction of steps decreasing true J_M1.
Physical consistency:
- P1 true-model cost: re-evaluate every analysis (incl. emulator-based) with J_M1(x0^a).
- P2 chi-square check: E[2 J_min] ~ number of observations (Talagrand/Desroziers) — analysis statistically consistent with B, R(, Q).
- P3 increment in unstable subspace: fraction of (x0^a - x0^b) projected on leading backward Lyapunov vectors (AUS theory).
- P4 smoothness: iterate-path tortuosity = sum||dx^k|| / ||x0^K - x0^0||; weak-constraint: temporal smoothness of eta_k.
Forecast: RMSE vs lead time beyond window; valid prediction time (lead time until normalized error > threshold, in Lyapunov times).
Cost: # model+adjoint (or emulator fwd+bwd) evaluations, iterations to reach target J, wall-clock; meta/emulator training cost reported separately (amortization).
Generalization: held-out initial conditions on the attractor (train/val/test split by IC).

---
## Related work (Phase B.2, 2026-11-09) — paper-ready draft
Legend: [V] verified against arXiv/publisher during scan; [U] unverified — check before citing.

### Learned solvers for variational DA
Fablet et al. (arXiv:2007.12941 [V]; JAMES 2021, doi:10.1029/2021MS002572 [U]) introduced 4DVarNet: a trainable prior
with an LSTM gradient-based solver unrolled for fixed iterations, tested on L63/L96, outperforming fixed-step GD in
accuracy and iteration count. Notably, solvers trained to minimise the variational cost reconstruct worse than
truth-supervised ones. Extensions: SSH mapping (arXiv:2211.05904 [V]), multimodal SST-SSH (arXiv:2207.01372 [V]),
neural OI solvers (arXiv:2211.07209 [V]). Related: learned inverse observation operators to warm-start L-BFGS
(Frerix et al., ICML 2021, arXiv:2102.11192 [V]); learning 4D-Var inversion from observations (Filoche et al.,
arXiv:2211.09741 [V]); neural incremental DA (Blanke et al., arXiv:2406.15076 [V]; ~23 -> ~4 iterations, secondary
summary); ML preconditioner for CG inner loop (Trappler & Vidard, arXiv:2501.04369 [V]); meta-learned FNO inverse-Hessian
warm start for CG (arXiv:2509.22949 [V-id, authors U]).

### Learning to optimize (L2O)
Andrychowicz et al., NIPS 2016 (arXiv:1606.04474 [V]); VeLO (Metz et al., arXiv:2211.09760 [V]); short-horizon bias
in unrolled meta-optimization (Wu et al., ICLR 2018, arXiv:1803.02021 [V]) — fixed-K unrolls bias toward too-small
steps; algorithm unrolling review (Monga et al., IEEE SPM 2021, doi:10.1109/MSP.2020.3016905 [V]); LISTA (Gregor &
LeCun, ICML 2010 [U]); L2O primer/benchmark (Chen et al., JMLR 2022, arXiv:2103.12828 [V]) — L2O often fails OOD.
No canonical citation found yet for imitation/teacher-student L2O (open search item).

### Emulators, adjoints, model error
NN-emulator TL/adjoint usable in 4D-Var (Hatfield et al., JAMES 2021, doi:10.1029/2021MS002521 [V]). Adjoint-matching
surrogates beat forward-only surrogates in L63 4D-Var (Chennault et al., arXiv:2111.08626 [V]) — key gradient-fidelity
evidence. DA+ML emulation and model-error learning: Brajard et al. 2020 (doi:10.1016/j.jocs.2020.101171 [V]);
Bocquet et al. 2020 (doi:10.3934/fods.2020004 [V]); Farchi et al. 2021, strong-constraint 4D-Var model-error
correction on 2-layer QG (doi:10.1002/qj.4116 [V]); online / IFS follow-ups (arXiv:2107.11114, 2403.03702 [V]).

### Assimilation in the unstable subspace (P3)
Trevisan, D'Isidoro & Talagrand, QJRMS 2010 (doi:10.1002/qj.571 [V]) — 4DVar-AUS, optimal dim ~ N+ + 1;
Palatella, Carrassi & Trevisan, J. Phys. A 2013 (doi:10.1088/1751-8113/46/25/254020 [V]); Trevisan & Uboldi 2004 [U].

### L63 4D-Var setup & tooling
Classic L63 4D-Var: Miller, Ghil & Gauthiez 1994; Pires, Vautard & Talagrand 1996 (non-convexity grows with window) [U].
Typical values (dt=0.01, obs every 8-25 steps, sigma_obs ~ 1-sqrt(2), window 0.5-2 TU) [U] — take exact values from
DAPPER or Fablet 2020. Tools: DAPPER (JOSS 2024, doi:10.21105/joss.05150 [V]); TorchDA (arXiv:2409.00244 [V]);
TorchOpt (JMLR 2023, arXiv:2211.06934 [V]); higher (arXiv:1910.01727 [V]); learn2learn (arXiv:2008.12284);
Open-L2O (github.com/VITA-Group/Open-L2O).

### Positioning
- **Closest prior work:** Fablet et al. 4DVarNet — its fixed-iteration LSTM solver ~ our Meta-A, and it already compares
  cost- vs truth-supervised training. Differences: we keep the 4D-Var cost fixed (no learned prior); 2x2 physical-adjoint
  vs emulator-gradient design; imitation-based Meta-B; OOD on unseen ICs; G2 / P1 / P3 diagnostics.
- **Claimable novelty:** (1) controlled 2x2 separating optimizer learning from dynamics error; (2) imitation of a
  classical optimizer (Meta-B) as a generalization-oriented alternative to end-to-end unrolling; (3) learned optimizer
  compensating emulator missing dynamics, judged by true-model cost; (4) Lyapunov/AUS geometry diagnostics of learned
  steps — not found in L2O literature. No paper found that meta-trains a 4D-Var update rule and tests OOD on unseen ICs.
- **Risks to address in design:** Fablet's cost-trained < supervised finding partly anticipates Meta-B claim — define
  Meta-B vs supervised clearly; short-horizon bias (Wu 2018) confounds fixed-K Meta-A — vary K / report K-sweep.
- **Reusable:** DAPPER configs/baselines; TorchDA 4D-Var skeleton; TorchOpt or higher for meta-differentiation;
  4DVarNet solver design (repo URL U); Open-L2O baselines.

---
## Reference implementation: Δ inner-loss meta learner (bice, 2026-11-10)
Source (read-only): github.com/devowoney/bice, branch `meta-learner-one-task`, `src/gd_optimic/meta_learner/`
(`bilevel_optimizer.py::_delta_inner_step`, `delta_chain`, `one_task_update`; outer loop in `optimizer.py`).
Original context: ocean IC optimization through frozen GloNet emulator; J = J_obs only (J_b, J_q marked "Phase 1.b").

### Method (paper-ready draft)
Outer loop k: one forward+backward through the dynamics gives J(ψ^k), ∇J(ψ^k); a reference optimizer (teacher)
proposes f(ψ^k) (GD: η∇J; or Adagrad/Adam step with its own state).
Inner chain (no model calls): a network S_θ, fed the standardized state, ∇J/σ_∇ and an iteration embedding, outputs a
per-component stretch s, bounded by output_scale·tanh(·). It is applied K times with ∇J and f held fixed:
  ψ_{m+1} = ψ_m − (f/K) ⊙ max(1 + s_θ(ψ_m, ∇J(ψ^k), k), 0),   m = 0..K−1,   ψ^{k+1} = ψ_K.
With s ≡ 0 the chain reproduces exactly one teacher step; S can only rescale (never flip) the teacher step per
component, i.e. it acts as a learned, state-dependent diagonal preconditioner relative to the teacher.
Meta update (online, num_meta_steps Adam steps on θ per outer iteration, curriculum ALIGN → TRANS → PERF):
  l_align = mean(s²)                                       (stay at teacher step = imitation)
  l_perf  = −⟨∇J(ψ^k), ψ^{k−1} − ψ_K(θ)⟩ / ⟨∇J(ψ^{k−1}), f(ψ^{k−1})⟩   (Δ inner loss)
  l_meta  = w_align·l_align [ALIGN,TRANS] + w_perf_delta·l_perf [TRANS,PERF] + l_reg
l_perf is a first-order (chain-rule) estimate of the cost change from the previous move, reusing the gradient the
outer loop computes anyway at the new IC; its θ-gradient is exactly dJ(ψ^k)/dθ while θ is unchanged. ≈ −1 when
the move equals one teacher step, < −1 when the stretched move does better. Diagnostics: ρ = ⟨∇J^k,step⟩/⟨∇J^{k−1},step⟩
(~1 too short, ~0 well sized, <0 overshoot), |step|/|f|, cos(step, ∇J).
Cost per outer iteration: 1 forward+backward (gradient) + 1 forward (report cost) + num_meta_steps·K evaluations of S.
Savings must come from needing fewer outer iterations than the teacher.
Modes: online training ("one task", per IC), and frozen inference from a loaded checkpoint (same chain, no meta-update).

### Mapping to dameta
- **This mechanism = Meta-B.** Teacher = f (GD/Adagrad); supervised imitation ≈ l_align (s → 0 ⇔ teacher step);
  unsupervised chain-rule term ≈ l_perf (Δ inner loss). Resolves the earlier Meta-A/Meta-B watch-point:
  Meta-A = free update rule unrolled K steps **with a dynamics call per step**, trained end-to-end on J(x0^K);
  Meta-B = teacher-relative stretch, K chained moves **without** dynamics calls, Δ-J first-order meta loss.
- L63 adaptation: the UNet (2-D spatial) does not apply to d=3 → MLP S_θ on [x̃, ∇J/σ_∇, emb(k)] → s ∈ R^3 (strong)
  or R^{3(L+1)} (weak; control includes η). J must gain J_b (and J_q for weak), absent in bice.
- Observations on reference code (not modified): l_reg = λ·(λ·Σθ²/n), i.e. λ applied twice (λ²) — confirm intent.
  Per-channel σ_∇ standardization collapses to per-variable scaling for L63 (3 "channels", no spatial dims).

### bice components to port (2026-11-10, authorized)
- `ic_optimizer.ReferenceStep`: teacher step f(ψ^k) (GD: η·g; torch optimizer via private shadow copy, state advanced
  once per outer gradient). Port as-is (drop ocean mask).
- `optimizer.ICOptimizer.optimize`: outer loop forward → J → autograd.grad → step; train vs frozen-inference branches
  for the meta learner. J_obs only, ocean-specific pooling/masks/output handler → port a lean version, add J_b, J_q.
- `run_optimization_slurm.py`: Hydra `@hydra.main`, `project_root` resolver, run dirs under <project>/.tmp,
  `setup_slurm_environment`, `setup_reproducibility` (seeds + cudnn deterministic).
- `sbatch_parallel.sh`: Odyssey_GPU partition, account odyssey, qos low, `mamba activate oceanai`, env-var overrides.
Environment: conda env `oceanai` exists (/Odyssey/private/j25lee/miniforge3/envs/oceanai). Partitions: Odyssey_GPU
(a100/l40s/h100/h200), Brain_GPU, Codes_GPU, Cosyde_GPU.

---
## Milestone 1 — M1 Lorenz-63 model, D1 dataset, gradient verification (2026-11-10)

### Methods (paper-ready draft)
The physical model M1 is the Lorenz-63 system (σ = 10, ρ = 28, β = 8/3) discretized with a classical fourth-order
Runge–Kutta scheme (Δt = 0.01) in double precision. Tangent-linear and adjoint models are obtained by automatic
differentiation (PyTorch `torch.func.jvp`/`vjp`) through the discrete scheme, so they are the exact derivatives of the
discrete model. Twin-experiment data (D1) are generated from M1: initial conditions are drawn around the attractor
centre and spun up for 1000 steps; each case comprises a truth trajectory over a 1-time-unit assimilation window
(L = 100 steps) followed by a 5-time-unit forecast range (F = 500 steps), observations of the full state every
10 steps (11 times, including t₀) perturbed by N(0, 2I) noise, and a background x₀ᵇ = x₀ᵗ + N(0, I). An x-only
observation pattern is defined through a mask on the same observations. The three splits (1000 / 200 / 200
initial conditions) use statistically independent random streams derived from one master seed (NumPy
`SeedSequence.spawn`), and the test cases are never used for training.

### Verification (src/tests/test_m1.py — 11/11 checks pass, 33 s on CPU)
| Check | Result | Criterion |
|---|---|---|
| Tendency Jacobian, autodiff vs analytic | 0 (max abs) | < 1e-12 |
| TL Taylor test over L=100, ratio at α=1e-6 | 1 + 1.5e-6 (first-order convergence, 1.0156 → 1.0000015 for α=1e-2 → 1e-6) | |1−r| < 1e-4 |
| Adjoint dot-product test | 1.1e-15 (relative) | < 1e-12 |
| 4D-Var gradient (J_b + J_o), autodiff vs central FD | 1.8e-9 (relative) | < 1e-6 |
| Leading Lyapunov exponent, 16 traj × 50 TU | 0.890 | 0.85–0.96 |
| — long run, 16 traj × 300 TU (one-off) | **0.904 ± 0.012** | ref. 0.906 |
| Reproducibility, same seed | bit-identical (file on disk == regenerated) | exact |
| Obs-error / background-error variance | 2.042 / 1.068 | 2 / 1 ± 0.1 |
| Truth bounded on attractor | z ∈ [2.56, 46.85], all finite | 0 < z < 50 |

### Artifacts (tmp/, not yet authorized for run/)
- Data: tmp/data/l63_{train,val,test}_seed42.nc (x_true (ic, 601, 3); y_obs (ic, 11, 3); obs_mask (2, 11, 3); x0_b)
- Figures: tmp/figs/m1_d1_example.{png,pdf} (one test case), tmp/figs/m1_ic_split.{png,pdf} (train/test ICs on attractor)

### Observation for the generalization claim
Minimum distance between a test IC and its nearest train IC is 0.033 — far below the background-error std (1).
Test cases are distinct trajectories, but on the low-dimensional attractor they are densely surrounded by training
cases, so "unseen IC" is an **interpolation** test. A harder out-of-distribution split (e.g. region of the attractor
held out, different ρ, or x-only obs pattern unseen in training) may be needed to make the Meta-B vs Meta-A
generalization claim convincing. → decision for pilot.

### Held-out region split (harder generalization test, 2026-11-10)
To turn "unseen initial conditions" into an extrapolation test, the lobe-transition region of the attractor
(states near the z-axis, r_xy = √(x² + y²) < 3, ≈ 14 % of the invariant measure) is withheld from training and
used only in a separate split `test_ood`. In-distribution splits (train / val / test) draw initial conditions from
r_xy ≥ 4 only, so a gap of one background-error standard deviation separates the two populations. The region is
dynamically the most sensitive part of Lorenz-63 (regime switching between lobes), so it is harder as well as unseen.
A secondary out-of-distribution axis needs no new data: the x-only observation pattern withheld from training.

| Split | n | r_xy of ICs | nearest train IC: min / median |
|---|---|---|---|
| test (in-dist) | 200 | [4.03, 26.11] | 0.033 / 0.551 |
| test_ood | 200 | [0.00, 3.00] | **1.158 / 2.973** |

Verification: src/tests/test_m1.py now 13/13 (adds region membership and OOD separation ≥ 1). Data regenerated
(tmp/data/l63_{train,val,test,test_ood}_seed42.nc); figure tmp/figs/m1_ic_split.{png,pdf} shows the region.

---
## Milestone 2 — classical strong/weak-constraint 4D-Var (E1) (branch m2-classical-4dvar, merged)

### Conditioning of the L63 4D-Var problem (Hessian at the background, 20 val cases, 2026-11-10)
| Setting | median λ_max | max λ_max | median λ_min | min λ_min | median cond. |
|---|---|---|---|---|---|
| strong, full obs | 48.2 | 1.69e4 | 1.11 | −7.68e3 | 46.3 |
| strong, x-only | 6.0 | 515 | 1.13 | −5.93e3 | 3.96 |
| weak, full obs | 274 | 7.47e4 | −0.004 | −3.42e4 | 263 |
| weak, x-only | 125 | 2.35e3 | 1.11 | −2.64e4 | 103 |

Curvature is heavy-tailed across cases (λ_max spans > 2 orders of magnitude) and some cases are non-convex at the
background (negative eigenvalues). A single fixed gradient-descent step size cannot be both stable for the stiffest
cases (lr < 2/λ_max ≈ 1e-4) and efficient for typical ones (≈ 2/48 ≈ 0.04). The weak-constraint problem is further
ill-conditioned by the Q⁻¹ = 100 block on η. This is the case-dependent step-size problem that a learned,
per-component stretch of the teacher step (Meta-B) is designed to address.

### Learning-rate selection (val, first 50 cases; rule pre-registered: min median final J, ≤1 % diverged)
| Setting | Adagrad lr (median J, median evals) | GD lr (median J, median evals) |
|---|---|---|
| strong / full | **0.1** (15.83, 622) | **0.001** (16.87, 1000 = budget) |
| strong / x-only | **0.1** (5.46, 324) | **0.001** (5.39, 1000) |
| weak / full | **0.1** (16.12, 644) | **0.0003** (17.08, 1000) |
| weak / x-only | **0.1** (5.23, 340) | **0.0003** (5.37, 1000) |
Grids: strong lr ∈ {1e-3, 3e-3, 1e-2, 2e-2, 4e-2, 0.1, 0.3, 1} (+ {1e-4, 3e-4} for GD); weak lr ∈ {1e-4, 3e-4, 1e-3,
3e-3, 7e-3, 3e-2, 0.1, 0.3}. All selected values are interior to
their grid (strong-GD grid extended down to 1e-4 to confirm). Full tables: `python src/select_lr.py`.

Observations (paper-ready):
- Adagrad is robust: no divergence at any tested lr, a single lr (0.1) is optimal in all four settings.
- Fixed-step GD is fragile in the non-convex, heavy-tailed L63 problem: larger steps do not diverge at first but stop
  early at higher cost (strong/full, lr 0.01: "converged" in 204 evals at J = 18.1, forecast RMSE at 1 TU ×1.8 worse),
  and from lr ≥ 0.03 (weak) 46–98 % of cases diverge. The best GD step is small and exhausts the 1000-iteration budget.
- The χ² consistency check holds at the optimum: median J_min ≈ m/2 (16.5 for 33 full-state obs; 5.5 for 11 x-only obs).

### Methods (paper-ready draft)
The strong-constraint cost is J(x₀) = ½‖x₀ − x₀ᵇ‖²_{B⁻¹} + ½ Σᵢ ‖H M_{0→tᵢ}(x₀) − yᵢ‖²_{R⁻¹} over the 11 observation
times of the 1-TU window; the weak-constraint control adds model-error increments ηᵢ applied to the state at the
observation times t₁…t₁₀, penalized by ½ Σᵢ ‖ηᵢ‖²_{Q⁻¹} (Q = 0.01 I). Gradients are exact (reverse-mode
differentiation through the RK4 model); one gradient evaluation = one forward plus one adjoint integration and is the
unit of computational cost. Two classical optimizers are used, both as E1 baselines and as Meta-B teachers: fixed-step
gradient descent and Adagrad (ReferenceStep ported from bice). All cases of a split are minimized in parallel, but
each case stops individually when its relative cost decrease falls below 10⁻⁶, or after 1000 iterations. Step sizes
were selected on validation data with a rule fixed before the sweep (lowest median final J, ≤ 1 % divergence).

### Results — E1 (physical model M1), 200 cases per split (tmp/results/e1/summary.md)
| split | constraint / obs | method (lr) | median J (χ² ref m/2) | median IC RMSE (bg) | fc RMSE @1 TU (bg) | VPT [TU] (bg) | evals to ≤1 % of J* |
|---|---|---|---|---|---|---|---|
| test | strong / full | Adagrad (0.1) | 16.5 (16.5) | 0.429 (0.895) | 0.542 (3.83) | 2.27 (0.61) | **70** |
| test | strong / full | GD (0.001) | 17.0 | 0.429 | 0.704 | 2.13 | 261 |
| test | weak / full | Adagrad (0.1) | 16.4 | 0.428 | 0.558 | 2.25 | **138** |
| test | weak / full | GD (3e-4) | 17.0 | 0.525 | 0.756 | 2.17 | 617 (54 % reach) |
| test | strong / x-only | Adagrad (0.1) | 5.43 (5.5) | 0.608 | 1.34 | 1.58 | **40** |
| test | strong / x-only | GD (0.001) | 5.47 | 0.611 | 1.36 | 1.46 | 549 |
| test_ood | strong / full | Adagrad (0.1) | 17.1 | 0.494 (0.853) | 0.975 (7.66) | 2.24 (0.74) | **125** |
| test_ood | strong / full | GD (0.001) | 18.3 | 0.517 | 1.94 | 1.72 | 219 |
| test_ood | weak / full | Adagrad (0.1) | 17.0 | 0.518 | 1.11 | 2.18 | **190** |
| test_ood | weak / full | GD (3e-4) | 17.7 | 0.559 | 1.48 | 1.79 | 454 (64 % reach) |
J* = best final J of either method on the same case. VPT threshold = 0.4 × climatological std = 5.62.
Full table (all patterns, val): tmp/results/e1/summary.md; evals-to-target: tmp/results/e1/evals_to_target.csv.
Figures: tmp/figs/m2_e1_forecast_rmse.{png,pdf}, tmp/figs/m2_e1_convergence.{png,pdf}.

### Observations (paper-ready)
1. No divergence in any final run (0/200 in all 24 configurations); median J_min matches the χ² expectation m/2,
   so the analyses are statistically consistent with B and R.
2. 4D-Var reduces the IC error by ~50 % (0.43 vs 0.90) and extends the valid prediction time from 0.6 to ≈ 2.2 TU.
3. Adagrad dominates fixed-step GD: equal or better accuracy at 2–14× fewer gradient evaluations (to 1 % of J*).
   GD's best step is small and stalls (zig-zag in the weak problem), reaching the target in only 20–64 % of the
   weak-constraint cases.
4. The held-out lobe-transition region is harder: background forecast error doubles (7.66 vs 3.83 at 1 TU), Adagrad
   needs ~1.8× more evaluations, and the analysis forecast error at 1 TU roughly doubles (0.98 vs 0.54) — a genuine
   extrapolation test for the meta learners.
5. Weak ≈ strong constraint for M1, as expected in a perfect-model twin (no model error to absorb); the weak
   formulation will matter for the emulator (E3/E4). Adagrad's first weak-constraint step increases J
   (step lr·sign(g) = 0.1 = σ_q on η) before recovering.
6. Cost metric: convergence curves plateau long before the 10⁻⁶ stopping rule fires, so "evaluations to reach
   (1 + tol) J*" is the fair cost measure for the North Star gate. Wall-clock is batch-level (≈ 0.5 s per case per
   1000 iterations on CPU) and not per-case; it will be reported per configuration.

### Cost-gate reference (resolved by pilot, 2026-11-10)
The North Star cost gate is reported against both classical optimizers, and the claim is anchored on the strongest
one, Adagrad: a meta optimizer must reach J ≤ 1.01 J* in ≤ 1/10 of Adagrad's gradient evaluations (≈ 7 vs 70 for
strong/full test; ≈ 14 vs 138 weak/full; ≈ 13 vs 125 strong/full test_ood). GD targets (≈ 26 / 62 / 22) are reported
alongside.
