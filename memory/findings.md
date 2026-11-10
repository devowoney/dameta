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
