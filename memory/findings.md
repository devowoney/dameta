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

---
## Milestone 3 — emulator M2, gradient fidelity G1, 4D-Var through the emulator (E3) (branch m3-emulator, merged)

### Emulator (methods, paper-ready draft)
M2 replaces the Lorenz-63 vector field by a multilayer perceptron h_φ (3 → 64 → 64 → 3, SiLU activations, inputs
standardized by the training-state statistics, outputs scaled by the training tendency standard deviation) inside
the same fourth-order Runge–Kutta scheme (Δt = 0.01), so that 4D-Var gradients through M2 come from automatic
differentiation of a smooth map. It is trained on the 1000 D1 training trajectories (601 states each) with a
multi-step rollout loss (standardized MSE averaged over n_roll steps; curriculum n_roll = 1 → 10 over the first
half of training), Adam with cosine decay (10⁻³ → 10⁻⁵, 10 000 steps, batch 512), gradient clipping, and the
best checkpoint selected on the D1 validation trajectories (50-step rollout RMSE). Training takes 7.5 min on CPU.
Deliberate model error is introduced only through the teacher model generating the training trajectories (same
initial conditions): biased parameters and/or a scaled −xz term in dy/dt.

### Clean emulator diagnostics (test / test_ood, medians [IQR]; tmp/results/emulator/clean_diag.json)
| Metric | test | test_ood |
|---|---|---|
| one-step RMSE | 9.1e-4 | 9.6e-4 |
| free-forecast RMSE from true x0 @ 0.5 / 1 / 2 TU | 0.024 / 0.051 / 0.145 | 0.055 / 0.054 / 0.180 |
| valid prediction time (0.4 × clim. std) | 5.1 TU [4.3, 6.0] | 5.0 TU [4.2, 6.0] |
| leading Lyapunov exponent (M1: 0.904 ± 0.012) | 0.927 [0.888, 0.938] | 0.902 [0.875, 0.925] |
| TL fidelity over the window: cos / rel. error | 0.9998 / 0.050 | 0.9999 / 0.029 |
| **G1** strong/full at x₀ᵇ: cos / rel. error | **0.999 / 0.069** | 0.9999 / 0.022 |
| **G1** strong/full at x₀ᵗ (near minimum): cos / rel. error | **0.997 / 0.145** (86 % cases cos > 0.9) | 0.9996 / 0.177 (90 %) |
| **G1** weak/full at (x₀ᵇ, η = 0): cos / rel. error | 0.999 / 0.065 | 0.9998 / 0.030 |
| **G1** strong/x-only at x₀ᵇ: cos / rel. error | 0.999 / 0.066 | 0.9999 / 0.025 |

Observations: the clean emulator is accurate (valid prediction time ≈ 5 TU from the true state, ≫ the 2.2 TU of the
E1 analyses) and slightly over-chaotic (λ₁ +2.5 %). Its 4D-Var gradients are nearly parallel to the exact ones far
from the minimum (cos ≈ 0.999), but near the minimum, where the true gradient is small, the emulator error dominates
(relative error 15–18 %, 10–14 % of cases with cos < 0.9). Emulator-based 4D-Var is therefore expected to converge
to a slightly displaced minimum — the gap a meta learner would have to close.

### Candidate deliberate degradations (teacher model vs M1 truth, test ICs)
| Teacher | λ₁ | RMSE from true x0 @ 0.5 / 1 TU | VPT [TU] |
|---|---|---|---|
| A: ρ = 26 | 0.852 | 1.45 / 3.05 | 1.17 |
| B: xz_scale = 0.9 | 0.890 | 1.84 / 4.25 | 0.94 |
| B′: xz_scale = 0.95 | 0.911 | 0.88 / 2.15 | 1.62 |
| C: ρ = 26 & xz_scale = 0.9 | 0.857 | 1.01 / 2.14 | 1.68 |
All remain chaotic. Within the 1-TU window, B's model error exceeds the observation error (σ_o = 1.41) — a stress
test where strong constraint must fail; B′ is comparable to the observation error. Choice pending with the pilot.

### E3 with the clean emulator vs E1 (test / test_ood, 200 cases each; tmp/results/e3_clean/compare_e1.md)
Learning rates re-selected on val with the pre-registered rule (3-point grids around the E1 values): identical to E1
(Adagrad 0.1; GD 10⁻³ strong, 3·10⁻⁴ weak). E3 analyses are judged with the true model (P1 = J_M1 of the analysis;
forecast RMSE with M1 from the analysis).
| split | constraint / obs | method | P1: J_true E3 (J E1) | IC RMSE E3 (E1) | fc RMSE @1 TU, M1 forecast: E3 (E1) | emulator forecast E3 | paired gap E3−E1 @1 TU | evals to 1 % (E3) |
|---|---|---|---|---|---|---|---|---|
| test | strong / full | Adagrad | 17.0 (16.5) | 0.438 (0.429) | 0.690 (0.542) | 0.654 | 0.035 | 79 |
| test | strong / full | GD | 17.2 (17.0) | 0.452 (0.429) | 0.835 (0.704) | 0.755 | 0.016 | 234 |
| test | weak / full | Adagrad | 16.7 (16.4) | 0.431 (0.428) | 0.685 (0.558) | 0.596 | 0.026 | 138 |
| test | strong / x-only | Adagrad | 5.52 (5.43) | 0.610 (0.608) | 1.34 (1.34) | 1.33 | 0.019 | 41 |
| test_ood | strong / full | Adagrad | 17.3 (17.1) | 0.499 (0.494) | 1.02 (0.975) | 0.880 | 0.0005 | 153 |
| test_ood | weak / full | Adagrad | 17.2 (17.0) | 0.512 (0.518) | 1.09 (1.11) | 1.11 | 0.013 | 198 |
(all 16 configurations: 0 diverged; full table in compare_e1.md; figure tmp/figs/m3_e3_clean_forecast.{png,pdf})

Observations (paper-ready):
1. A well-trained emulator makes emulator-based 4D-Var almost as good as 4D-Var with the exact adjoint: the analysis
   is slightly displaced (P1 cost +1–3 %, IC error +0–5 %), consistent with the G1 degradation near the minimum, and
   the iteration cost is unchanged (79 vs 70 evaluations to 1 % for Adagrad, strong/full).
2. The case-paired forecast gap E3 − E1 is small (median 0.0005–0.035 at 1 TU) while the medians differ more
   (0.69 vs 0.54): the emulator penalty is concentrated in a minority of cases (heavy right tail).
3. Consequence for the North Star: with the clean emulator the gap-closure ratio (E3 − E4)/(E3 − E1) has a near-zero
   denominator and would be dominated by noise. Testing "the meta learner re-learns missing dynamics" requires the
   deliberately degraded emulator.

### Degraded emulators (pilot choice: B′ xz_scale = 0.95 main, B xz_scale = 0.9 stress)
Both trained like the clean emulator (10 000 steps, ≈ 12 min; best val rollout RMSE vs their own teacher 0.030 / 0.034).
Diagnostics vs the TRUE model M1 (test / test_ood medians [IQR]; tmp/results/emulator/<name>_diag.json):
| Metric | xz095 test | xz095 test_ood | xz090 test | xz090 test_ood |
|---|---|---|---|---|
| free-forecast RMSE @1 TU from true x0 | 2.12 | 1.15 | 4.19 | 2.43 |
| VPT [TU] | 1.63 | 2.50 | 0.93 | 1.72 |
| λ₁ | 0.908 | 0.890 | 0.910 | 0.881 |
| TL cos / rel. err (window) | 0.945 / 0.44 | 0.988 / 0.20 | 0.783 / 0.87 | 0.949 / 0.39 |
| **G1** cos at x₀ᵇ (strong/full) | 0.980 [0.82, 0.995] | 0.9995 | **0.909 [−0.36, 0.98]** | 0.998 |
| **G1** rel. err at x₀ᵇ | 0.68 | 0.12 | 1.16 | 0.24 |
| **G1** cos at x₀ᵗ (near minimum) | **0.24 [−0.88, 0.94]** | 0.92 | **−0.13 [−0.88, 0.86]** | 0.71 |
| one-obs-interval model error std (x, y, z) | 0.38, 0.88, 0.46 | | 0.77, 1.80, 0.96 | |
(clean emulator: one-obs-interval error std 0.004 / 0.010 / 0.010)

lr re-selected on val (full obs) with the pre-registered rule; strong-constraint grids extended at both edges to
make the choice interior: Adagrad 0.3 (strong) / 0.1 (weak); GD 3·10⁻⁴ (strong and weak), identical for both
degraded emulators.

### E3 with degraded emulators vs E1 (Adagrad, full obs, test; tmp/results/e3_<name>/compare_e1.md)
| | E1 (M1) | E3 clean | E3 xz095 | E3 xz090 |
|---|---|---|---|---|
| P1: J_true of analysis (χ² ref 16.5) | 16.5 | 17.0 | **31.9** | **72.2** |
| IC RMSE (background 0.895) | 0.429 | 0.438 | 0.771 | 1.17 |
| fc RMSE @1 TU, M1 forecast from analysis (background 3.83) | 0.542 | 0.690 | **3.10** | **5.02** |
| fc RMSE @1 TU, emulator forecast from its analysis | — | 0.654 | 1.36 | 2.47 |
| VPT [TU], M1 forecast (background 0.61) | 2.27 | 2.15 | 0.57 | 0.42 |
| paired gap E3 − E1 @1 TU (gap-closure denominator) | — | 0.035 | **2.09** | **3.77** |
| evals to 1 % of own J* | 70 | 79 | 30 | 46 |
Weak constraint, xz095: fc 3.46, gap 2.10; test_ood xz095 strong: fc 2.20 vs E1 0.975, gap 0.66.

Observations (paper-ready):
1. Missing dynamics destroys gradient fidelity long before forecast skill: with 5 % of the −xz coupling removed, the
   emulator still forecasts for 1.6 TU, but near the minimum its 4D-Var gradient is essentially uncorrelated with the
   true one (median cos 0.24; xz090: −0.13, with > 25 % of cases pointing uphill already at the background).
2. Emulator-based 4D-Var then converges quickly (30–46 evaluations) to a minimum of the WRONG cost: the analysis fits
   the emulator dynamics (emulator forecast 1.36) but not the truth (M1 forecast 3.10, valid prediction time no better
   than the background). P1 doubles (31.9) and quadruples (72.2) relative to the χ² reference.
3. This gives a large, well-defined gap-closure denominator (2.09 / 3.77 at 1 TU on test), so the North Star test
   "meta learner re-learns missing dynamics" is now measurable (E4 vs E3 vs E1).
4. The degradation is region-dependent: the missing term −xz in dy/dt scales with x, which is small near the z-axis,
   so test_ood (lobe-transition region) is LESS affected by this model error (G1 cos 0.9995, gap 0.66) even though it
   is dynamically harder. To be stated as a confound when interpreting OOD results for E3/E4.
5. Weak constraint does not help with the degraded emulators because Q is mis-specified: σ_q = 0.1 is 4–9× (xz095)
   and 8–18× (xz090) smaller than the actual one-interval model error (but 10× larger than the clean emulator's).
   Decision for the pilot: keep Q fixed (as approved) or re-estimate Q per emulator from train/val data.

---
## Consolidated report — 4D-Var initial-condition estimation (E1, E3) [paper-ready, 2026-11-11]
Reproduce: `python src/report_ic.py` → tmp/report/table{1,2,3}_*.{md,csv}, tmp/report/fig{1,2,3}_*.{png,pdf}.
This section supersedes the per-run E1/E3 numbers above where they differ (single cost definition, bootstrap CIs).

### Evaluation protocol
Each split (test: 200 in-distribution cases; test_ood: 200 cases from the held-out lobe-transition region
r_xy < 3) is a set of independent twin experiments sharing the same truth, observations and background across all
configurations, so every comparison is paired by case. The analysis x₀ᵃ is judged against the truth by (i) the
initial-condition RMSE, (ii) the RMSE of the forecast obtained by integrating x₀ᵃ with the true model M1 over 5 time
units (TU) beyond the 1-TU window, (iii) the valid prediction time (VPT), the first lead at which that forecast RMSE
exceeds 0.4 × the climatological standard deviation (5.62), and (iv) the 4D-Var cost of x₀ᵃ evaluated with M1 (P1).
Cost is the number of gradient evaluations (one forward + one adjoint integration) needed to reach
J ≤ 1.01 J*, where J* is the best final cost reached by either classical optimizer on the same case with the same
dynamics. Summaries are medians over cases with 95 % percentile-bootstrap confidence intervals (2000 resamples);
E3–E1 differences are paired medians with bootstrap CIs and two-sided Wilcoxon signed-rank tests. Emulator-based
analyses are always evaluated with M1, so that differences reflect the quality of the estimated initial condition.

### Table 1 — Classical 4D-Var with the exact model (E1), median [95 % CI], 200 cases per split
| split | constraint / obs | optimizer (lr) | IC RMSE | fc RMSE @1 TU | VPT [TU] | J (m/2) | evals to 1 % of J* (reach) |
|---|---|---|---|---|---|---|---|
| test | background (no DA) | — | 0.895 [0.836, 0.949] | 3.83 [3.34, 4.56] | 0.61 [0.50, 0.90] | — | — |
| test | strong / full | Adagrad (0.1) | 0.429 [0.385, 0.483] | 0.54 [0.43, 0.78] | 2.27 [2.06, 2.68] | 16.5 (16.5) | **70** [56, 88] (92 %) |
| test | strong / full | GD (10⁻³) | 0.429 [0.382, 0.522] | 0.70 [0.53, 0.85] | 2.13 [1.94, 2.32] | 17.0 | 261 [224, 283] (92 %) |
| test | weak / full | Adagrad (0.1) | 0.428 [0.379, 0.492] | 0.56 [0.42, 0.70] | 2.25 [2.07, 2.64] | 16.4 | **138** [95, 171] (99 %) |
| test | weak / full | GD (3·10⁻⁴) | 0.525 [0.478, 0.556] | 0.76 [0.56, 0.94] | 2.17 [2.01, 2.37] | 17.0 | 617 [526, 682] (54 %) |
| test | strong / x-only | Adagrad (0.1) | 0.608 [0.567, 0.661] | 1.34 [1.11, 1.66] | 1.58 [1.37, 1.82] | 5.43 (5.5) | **40** [30, 51] (98 %) |
| test | strong / x-only | GD (10⁻³) | 0.611 [0.568, 0.680] | 1.36 [1.03, 1.57] | 1.46 [1.33, 1.78] | 5.47 | 549 [481, 580] (90 %) |
| test_ood | background (no DA) | — | 0.853 [0.814, 0.904] | 7.66 [5.06, 10.28] | 0.74 [0.66, 0.81] | — | — |
| test_ood | strong / full | Adagrad (0.1) | 0.494 [0.445, 0.569] | 0.98 [0.68, 1.37] | 2.24 [2.00, 2.39] | 17.2 | **125** [101, 184] (84 %) |
| test_ood | strong / full | GD (10⁻³) | 0.517 [0.444, 0.605] | 1.94 [1.15, 2.98] | 1.72 [1.57, 2.06] | 18.3 | 219 [186, 278] (82 %) |
| test_ood | weak / full | Adagrad (0.1) | 0.518 [0.452, 0.568] | 1.11 [0.77, 1.75] | 2.18 [1.85, 2.37] | 17.0 | **190** [148, 234] (94 %) |
| test_ood | weak / full | GD (3·10⁻⁴) | 0.559 [0.489, 0.624] | 1.48 [1.02, 2.40] | 1.79 [1.66, 2.17] | 17.7 | 454 [318, 563] (64 %) |
All 24 E1 runs: 0 / 200 diverged. Full table incl. x-only and weak/x-only rows: tmp/report/table1_e1_classical.md.

### Table 2 — 4D-Var through the emulator (E3) vs E1, Adagrad, paired by case
| emulator | split | constraint / obs | IC RMSE E3 (E1) | Δ fc @1 TU, E3 − E1 [95 % CI] | p (Wilcoxon) | E3 worse | VPT E3 (E1) | P1: J_M1 E3 / E1 |
|---|---|---|---|---|---|---|---|---|
| clean | test | strong / full | 0.438 (0.429) | 0.035 [0.010, 0.065] | 3·10⁻³ | 59 % | 2.15 (2.27) | 1.00 |
| clean | test | weak / full | 0.431 (0.428) | 0.026 [0.004, 0.050] | 1·10⁻² | 58 % | 2.12 (2.25) | 1.00 |
| clean | test_ood | strong / full | 0.499 (0.494) | 0.000 [−0.008, 0.011] | 0.50 | 50 % | 2.31 (2.24) | 1.00 |
| **xz095** | test | strong / full | 0.771 (0.429) | **2.09** [1.73, 2.79] | 1·10⁻²⁴ | 89 % | 0.57 (2.27) | 1.75 |
| xz095 | test | weak / full | 0.722 (0.428) | 2.10 [1.85, 2.78] | 1·10⁻²³ | 87 % | 0.56 (2.25) | 1.73 |
| xz095 | test | strong / x-only | 0.792 (0.608) | 1.56 [1.08, 1.82] | 3·10⁻¹³ | 74 % | 0.68 (1.58) | 1.31 |
| xz095 | test_ood | strong / full | 0.679 (0.494) | 0.66 [0.45, 0.94] | 3·10⁻¹² | 76 % | 1.40 (2.24) | 1.17 |
| xz095 | test_ood | weak / full | 0.637 (0.518) | 0.76 [0.59, 1.13] | 2·10⁻⁹ | 73 % | 1.34 (2.18) | 1.19 |
| **xz090** | test | strong / full | 1.172 (0.429) | **3.77** [3.32, 4.40] | 7·10⁻²⁸ | 94 % | 0.42 (2.27) | 4.30 |
| xz090 | test | weak / full | 1.114 (0.428) | 3.85 [3.17, 4.50] | 3·10⁻²⁶ | 92 % | 0.41 (2.25) | 3.98 |
| xz090 | test_ood | strong / full | 1.107 (0.494) | 2.09 [1.42, 2.80] | 9·10⁻¹⁹ | 86 % | 0.81 (2.24) | 1.79 |
All 40 E3 runs: 0 / 200 diverged. Full table (20 rows): tmp/report/table2_e3_vs_e1_paired.md.

### Table 3 — Emulator fidelity (test; test_ood in brackets)
| emulator (teacher xz_scale) | free-fc VPT from x₀ᵗ [TU] | λ₁ (M1 0.904) | TL cos (window) | G1 cos @ x₀ᵇ | G1 cos @ x₀ᵗ (cases < 0) | model error per obs interval (x, y, z) |
|---|---|---|---|---|---|---|
| clean (1.00) | 5.14 (4.98) | 0.927 | 1.000 | 0.999 (1.000) | 0.997 (6 %) | 0.004, 0.010, 0.010 |
| xz095 (0.95) | 1.62 (2.50) | 0.908 | 0.945 | 0.980 (0.999) | **0.24** (47 %) [OOD 0.92] | 0.38, 0.88, 0.46 |
| xz090 (0.90) | 0.93 (1.71) | 0.910 | 0.783 | 0.908 (0.998) | **−0.13** (50 %) [OOD 0.71] | 0.77, 1.80, 0.96 |
Weak-constraint Q is kept at σ_q = 0.1 (pilot decision) — 4–18× below the degraded emulators' model error.

### Figures (tmp/report/)
- **Fig. 1** `fig1_ic_estimation_overview` — Strong-constraint 4D-Var with full observations and Adagrad, 200
  cases per split (rows: test, test_ood). (a, d) Distribution of the initial-condition RMSE for the background, E1
  and E3 with the three emulators. (b, e) Median RMSE of the forecast integrated with the true model from each
  analysis, versus lead time beyond the window (bands: interquartile range for E1 and E3-xz095; dashed: VPT
  threshold). (c, f) Distribution of the valid prediction time. Boxes: interquartile range; whiskers: 1.5 × IQR.
- **Fig. 2** `fig2_e1_optimizer_cost` — Cost of classical 4D-Var with the exact model (E1). (a, b) Median
  relative excess cost J/J* − 1 versus gradient evaluations (band: IQR) for Adagrad and gradient descent, strong and
  weak constraint, test split. (c) Gradient evaluations to reach 1 % of J* (median, 95 % bootstrap CI) for all
  eight configurations (T = test, OOD = test_ood; st/we = strong/weak; full/x = observed variables).
- **Fig. 3** `fig3_e3_gradient_fidelity` — Gradient fidelity of the emulators (G1, strong constraint, full
  observations). (a, b) Cosine between the 4D-Var gradient computed through the emulator and the exact gradient,
  at the background and at the true initial condition (test). (c) Median relative gradient error at the true
  initial condition versus the median paired forecast penalty E3 − E1 at 1 TU (95 % bootstrap CIs; filled = test,
  open = test_ood).

### Results (paper-ready text)
**Classical baseline.** With the exact model, strong-constraint 4D-Var halves the initial-condition error of the
background (0.43 vs 0.90, test) and extends the valid prediction time from 0.6 to 2.3 TU; the median final cost
matches the χ² expectation m/2 in every configuration, and no case diverges (Table 1, Fig. 1). Adagrad reaches
1 % of the attainable cost in 70 gradient evaluations against 261 for the best fixed-step gradient descent, and
the advantage grows for the weak-constraint and partially observed problems (up to 14×), where gradient descent
fails to reach the target in 46–80 % of the cases (Fig. 2). The held-out lobe-transition region is harder for the
optimizer (125 vs 70 evaluations) and for the forecast (RMSE at 1 TU 0.98 vs 0.54), while the initial-condition
error increases only moderately (0.49 vs 0.43).

**4D-Var through an emulator.** A well-trained emulator (clean) is nearly a drop-in replacement for the exact model:
the paired forecast penalty is 0.035 [0.010, 0.065] at 1 TU on test and indistinguishable from zero on test_ood
(p = 0.50), the cost of the analysis evaluated with the true model is unchanged (ratio 1.00), and the optimizer
needs the same number of iterations (Table 2). When 5 % of the −xz coupling is missing from the emulator's training
dynamics (xz095), the emulator still forecasts for 1.6 TU from the true state, but the analysis degrades sharply:
the initial-condition error rises from 0.43 to 0.77, the forecast penalty is 2.09 [1.73, 2.79] at 1 TU (89 % of
cases worse, p ≈ 10⁻²⁴), and the valid prediction time collapses to the level of the background (0.57 TU). The
stress emulator (xz090) roughly doubles these effects (penalty 3.77, P1 cost ×4.3). The penalty is systematic in
strong and weak constraint alike: with σ_q fixed at 0.1, the weak-constraint control cannot absorb a model error
4–18 times larger.

**Mechanism.** The degraded emulators' gradients are accurate far from the minimum (cosine 0.98 for xz095 at the
background) but lose their direction close to it (median cosine 0.24; 47 % of cases point uphill at the true
initial condition), so emulator-based 4D-Var converges quickly to the minimum of a different cost (Table 3,
Fig. 3a–b). Across emulators and splits, the median analysis penalty increases monotonically with the median
relative gradient error at the true initial condition (Fig. 3c). At the level of individual cases, gradient error
is only a weak predictor of the penalty (Spearman ρ ≤ 0.38 for the degraded emulators, ≤ 0.23 for the clean one),
indicating that the penalty reflects the displaced minimum of the emulator cost rather than local gradient noise.

**Out-of-distribution region.** The missing term −xz in dy/dt scales with x, which is small near the z-axis. The
held-out region is therefore less affected by this particular model error (xz095 gap 0.66 vs 2.09; G1 cosine 0.92
vs 0.24 at the true initial condition) even though it is dynamically harder. OOD comparisons of E3/E4 must be read
with this confound in mind.

### Implications for Milestones 4–5
- Cost targets for the meta optimizers (≤ 1/10 of Adagrad, claim): ≈ 7 (strong/full), 14 (weak/full), 4
  (strong/x-only) evaluations on test; 13 (strong/full) and 19 (weak/full) on test_ood.
- Gap-closure denominators (median paired gap at 1 TU, strong/full): xz095 2.09 (test), 0.66 (test_ood);
  xz090 3.77 (test), 2.09 (test_ood). The clean emulator (0.035) is a control, not a gap-closure case.
- Limitations: one dataset seed and one emulator seed per configuration; learning rates tuned on 50 validation
  cases; forecast RMSE is averaged over the three variables; σ_q deliberately not re-estimated.
