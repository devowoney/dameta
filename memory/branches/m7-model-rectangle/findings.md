# Findings — branch m7-model-rectangle

## Model rectangle: current development and configuration (2026-11-23)

### Purpose
Define five dynamical models that separate the two questions of the study:
- **Problem 1, vertical axis (F1 → F2, E1 → E2):** how increasing dynamical model error affects 4D-Var.
- **Problem 2, horizontal axis (F → E):** how replacing the model by an imperfect differentiable emulator affects
  4D-Var.

```
          F1 ──(emulator approximation)──▶ E1          L1 = truth (observations, evaluation, reference)
          │ (physical misspecification)    │
          F2 ──(emulator approximation)──▶ E2
```

### Configuration (src/configs/models.yaml)
| model | definition | parameters / setting |
|---|---|---|
| L1 | Lorenz-63, RK4, Δt 0.01, float64 | θ_L = (σ, ρ, β) = (10, 28, 8/3) |
| F1 | L63 with θ_F1 = θ_L ⊙ (1 + ε1 d) | ε1 = 0.03 → (10.1732, 28.4850, 2.7129), ‖θ − θ_L‖₂ = 0.52 |
| F2 | L63 with θ_F2 = θ_L ⊙ (1 + ε2 d) | ε2 = 0.18 → (11.0392, 30.9098, 2.9438), ‖θ − θ_L‖₂ = 3.10 |
| E1 | MLP tendency + RK4, trained on F1 trajectories | 10 training ICs, MLP 1 hidden layer × 8 (SiLU), file e1_d10w8d1 |
| E2 | MLP tendency + RK4, trained on F2 trajectories | same setting as E1, file e2_d10w8d1 |

- Direction d = (1, 1, 1)/√3, relative to θ_L: every parameter is increased by ε/√3.
- ε selection, pre-registered:
  - criterion: one-obs-interval model error vs L1, m = RMS over (x, y, z) of the pooled error std / σ_o, on D1 v2
    val;
  - F1 = grid ε with m closest to 0.15 (m = 0.155);
  - F2 = grid ε with m closest to 1.0 (m = 0.958);
  - grid ε ∈ {0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20, 22, 25} %;
  - requirements: stable over 20 TU and λ₁ > 0;
  - screen: tmp/results/models/false_screen.{csv,md}.
- Emulator training recipe (otherwise unchanged from M3):
  - teacher trajectories from the first n D1 train ICs;
  - multi-step rollout loss with a 1 → 10-step curriculum;
  - 10 000 Adam steps (lr 1e-3, cosine to 1e-5);
  - best validation-rollout checkpoint;
  - the teacher θ is taken from the registry (`+registry_model`).
- Data: D1 v2 truth / observations unchanged.
  - window 110 steps (1.1 TU), observations every 10 steps (12 times), σ_o² = 2;
  - each model has its own time-lagged background (0.5-TU first guess);
  - hybrid B tuned per model (tmp/data/hybrid_B_seed42.json; E1 at the s² upper grid edge, F2 / E2 at α = 1).
- The previous well-trained emulators (1000 ICs, MLP 2 × 64) are kept as tmp/emulators/e1.pt and e2.pt.

### Emulator degradation grid (pilot 2026-11-23)
- Target: emulator-vs-own-teacher error m_h ≈ F1's model error (0.15 σ_o), so the horizontal step is comparable to
  the mild physical step L1 → F1.
- m_h is the one-obs-interval difference emulator − teacher from true states (RMS over x, y, z of the pooled std,
  / σ_o), on 200 D1 v2 val cases.
- RMSE is emulator vs teacher from the true x0 at 1 TU (mean ± std over cases). λ₁ is over 50 ICs and 20 TU.
- Pick rule: m_h closest to 0.15, stable over 20 TU, λ₁ > 0. The setting is chosen on F1 and applied unchanged to
  F2.

| teacher | emulator | train ICs | MLP (depth × width) | m_h [σ_o] | RMSE vs teacher @1 TU | stable 20 TU | λ₁ |
|---|---|---|---|---|---|---|---|
| F1 | e1 (previous) | 1000 | 2 × 64 | 0.008 | 0.141 ± 0.581 | yes | 0.922 ± 0.096 |
| F1 | e1_d100 | 100 | 2 × 64 | 0.0085 | 0.160 ± 0.756 | yes | 0.918 ± 0.095 |
| F1 | e1_d30 | 30 | 2 × 64 | 0.0097 | 0.251 ± 1.425 | yes | 0.918 ± 0.102 |
| F1 | e1_d10 | 10 | 2 × 64 | 0.012 | 0.287 ± 1.331 | yes | 0.906 ± 0.096 |
| F1 | e1_w16 | 1000 | 2 × 16 | 0.021 | 0.375 ± 1.455 | yes | 0.917 ± 0.111 |
| F1 | e1_w8 | 1000 | 2 × 8 | 0.068 | 1.064 ± 2.077 | yes | 0.839 ± 0.180 |
| F1 | e1_w8d1 | 1000 | 1 × 8 | 0.107 | 1.326 ± 2.898 | yes | 0.875 ± 0.110 |
| F1 | e1_d30w16 | 30 | 2 × 16 | 0.024 | 0.382 ± 1.580 | yes | 0.906 ± 0.097 |
| F1 | **e1_d10w8d1 (E1)** | 10 | 1 × 8 | **0.142** | 1.642 ± 2.480 | yes | 0.922 ± 0.133 |
| F2 | e2 (previous) | 1000 | 2 × 64 | 0.016 | 0.167 ± 0.269 | yes | 0.995 ± 0.108 |
| F2 | **e2_d10w8d1 (E2)** | 10 | 1 × 8 | **0.126** | 1.181 ± 1.002 | yes | 1.018 ± 0.092 |

Sources: tmp/results/models/emulator_screen_{f1,f2}.{csv,md}; checkpoints tmp/emulators/*.pt; training logs
tmp/logs/train_e*.log.

### Ensemble diagnostic (truth-free; configuration only)
- Members: member_i = M^τ(x_true(t0 − τ) + ε_i), with ε_i ~ N(0, 2 I) and τ = 0.5 TU.
- Control: M^τ(x_true(t0 − τ)), the same short forecast without noise.
- Error: member − own control, over 0 to 5 TU of lead time.
- Decorrelation ρ(t) = 1 − |d|² / (2 Σ σ²_clim), with σ²_clim from L1 (D1 v2 train). Decorrelation time = first
  lead with ρ < 0.5.
- The same noise draws are used for all models.
- Study 1: 100 windows (test cases 0–99), one run each. Study 2: fixed window (test case 0), 100 runs.
- Figures: tmp/figs/ens_{windows,fixed}_{growth,decorr,summary,dist}.{png,pdf}; arrays:
  tmp/results/models/ensemble_{windows,fixed}.npz.

### Code (this branch)
| file | role |
|---|---|
| src/configs/models.yaml, src/dameta/dynamics/models.py | model registry: `perturbed_params`, `build_model`, `model_params`, `teacher_config` |
| src/screen_teachers.py `--sweep` | ε screen along d (point-wise and invariant-measure metrics) |
| src/train_emulator.py `+registry_model`, `+n_train_ic` | teacher taken from the registry; training-IC subset |
| src/screen_emulators.py | emulator vs own teacher screen (m_h, RMSE, stability, λ₁) |
| src/ensemble_models.py | truth-free ensemble error growth / decorrelation figures |
| src/report_models.py | pairwise model-edge characterization (rectangle) |
| src/run_4dvar_v2.py, src/configs/fourdvar_v2.yaml | `model ∈ {l1, f1, f2, e1, e2}` (alias truth; legacy `emulator`); BPE prior = model's own θ |
| src/gen_data_v2.py, src/configs/gen_data_v2.yaml | `registry_models`, `backgrounds_only` |
| src/tune_hybrid_b.py | default models f1, f2, e1, e2 |
| src/tests/test_models.py | registry tests (PASS) |
