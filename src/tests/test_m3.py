"""
Milestone 3 verification — emulator M2, emulator diagnostics / G1, 4D-Var through the emulator (E3 runner path).

Plain-python test script (no pytest in env `oceanai`). Run from the project root:   python src/tests/test_m3.py
Uses a deterministic randomly initialized emulator, so it does not depend on a trained checkpoint.
"""

import sys
import tempfile
import time
from pathlib import Path

import torch
import xarray as xr
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on the path

from dameta.da import CostConfig, FourDVarCost  # noqa: E402
from dameta.dynamics import L63Model  # noqa: E402
from dameta.dynamics.emulator import L63Emulator, TeacherConfig, teacher_rollout  # noqa: E402
from dameta.eval import gradient_fidelity, tl_fidelity  # noqa: E402

torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parents[2]
DS = xr.open_dataset(ROOT / "tmp/data/l63_val_seed42.nc").isel(ic=slice(0, 8))
XT = torch.tensor(DS.x_true.values)
M, CFG = L63Model(), CostConfig()
RESULTS = []


def report(name, value, thr, ok):
    RESULTS.append((name, value, thr, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<50s} value={value:.3e}  ({thr})")
    assert ok, name


def random_emulator() -> L63Emulator:
    torch.manual_seed(0)
    emu = L63Emulator()
    emu.set_normalization(XT)
    return emu


def test_teacher_default_is_l63():
    """Default (clean) teacher trajectory == M1 trajectory (D1 x_true) to rounding."""
    err = (teacher_rollout(XT[:, 0], 600, TeacherConfig()) - XT).abs().max().item()
    report("clean teacher rollout == M1 x_true", err, "max abs < 1e-8", err < 1e-8)


def test_save_load_roundtrip():
    """Saved/loaded emulator gives bit-identical steps and keeps teacher provenance."""
    emu = random_emulator()
    with tempfile.TemporaryDirectory() as d:
        emu.save(Path(d) / "e.pt", TeacherConfig(rho=26.0), {"best_val_rmse": 0.0})
        emu2 = L63Emulator.load(Path(d) / "e.pt")
    with torch.no_grad():
        err = (emu.step(XT[:, 0]) - emu2.step(XT[:, 0])).abs().max().item()
    ok = err == 0.0 and emu2.teacher.rho == 26.0
    report("save/load round trip (+ teacher provenance)", err, "== 0", ok)


def test_emulator_4dvar_gradient_fd():
    """4D-Var gradient through the emulator: autodiff vs central finite differences (rule for M2)."""
    emu = random_emulator()
    for p in emu.parameters():
        p.requires_grad_(False)
    cost = FourDVarCost(emu.step, torch.tensor(DS.y_obs.values), torch.tensor(DS.obs_mask.sel(pattern="full").values),
                        torch.tensor(DS.x0_b.values), CFG)
    x0 = torch.tensor(DS.x0_b.values)
    _, g = cost.grad(x0)
    d = torch.randn(x0.shape, generator=torch.Generator().manual_seed(3))
    h = 1e-6
    fd = (cost(x0 + h * d) - cost(x0 - h * d)) / (2 * h)
    rel = ((fd - (g * d).sum(1)).abs() / (g * d).sum(1).abs()).max().item()
    report("emulator 4D-Var gradient: autodiff vs FD", rel, "relative < 1e-5", rel < 1e-5)


def test_fidelity_metrics_self_consistency():
    """G1 and TL fidelity of M1 against itself: cos = 1, relative error = 0."""
    g1 = gradient_fidelity(M.step, M.step, torch.tensor(DS.y_obs.values), torch.tensor(DS.obs_mask.sel(pattern="full")
                           .values), torch.tensor(DS.x0_b.values), CFG, torch.tensor(DS.x0_b.values), weak=False)
    tl = tl_fidelity(M.step, M.step, XT[:, 0], CFG.window)
    err = max((g1["cos"] - 1).abs().max().item(), g1["rel_err"].max().item(), (tl["cos"] - 1).abs().max().item(),
              tl["rel_err"].max().item())
    report("G1 / TL fidelity self-consistency (M1 vs M1)", err, "< 1e-12", err < 1e-12)


def test_e3_runner_path():
    """run_4dvar with dynamics=emulator runs and adds the true-model checks (J_true_final, rmse_fc_m1)."""
    sys.path.insert(0, str(ROOT / "src"))
    import run_4dvar  # noqa: E402

    emu = random_emulator()
    with tempfile.TemporaryDirectory() as d:
        emu.save(Path(d) / "rand.pt", TeacherConfig(), {"best_val_rmse": 0.0})
        cfg = OmegaConf.load(ROOT / "src/configs/fourdvar.yaml")
        cfg = OmegaConf.merge(cfg, {"dynamics": "emulator", "emulator": "rand", "emulator_dir": d, "n_cases": 4,
                                    "max_iter": 5, "method": "adagrad", "lr": 0.1, "data_dir": str(ROOT / "tmp/data"),
                                    "out_dir": d, "hydra": None})
        out = run_4dvar.run(cfg)
    ok = {"J_true_final", "rmse_fc_m1", "rmse_fc"} <= set(out.data_vars) and out.attrs["configuration"] == "E3:rand"
    report("E3 runner path (true-model checks present)", 0.0 if ok else 1.0, "== 0", ok)


if __name__ == "__main__":
    t0 = time.time()
    for name, fn in [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]:
        fn()
    print(f"\nAll {len(RESULTS)} checks passed in {time.time() - t0:.1f}s")
