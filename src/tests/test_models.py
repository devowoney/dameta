"""
Model-rectangle registry checks (src/dameta/dynamics/models.py, src/configs/models.yaml).

Plain-python test script. Run from the project root:  python src/tests/test_models.py
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dameta.dynamics import L63Model, L63Params  # noqa: E402
from dameta.dynamics.models import (build_model, load_registry, model_params, perturbed_params,  # noqa: E402
                                    teacher_config, theta_distance)

torch.set_default_dtype(torch.float64)


def test_eps0_is_truth():
    reg = load_registry()
    p = perturbed_params(reg.theta_L, reg.direction, 0.0, reg.dt)
    assert p == L63Params(), p
    assert model_params("l1", reg) == L63Params() and model_params("truth", reg) == L63Params()


def test_l1_step_bit_identical():
    reg = load_registry()
    step, desc, p = build_model("l1", reg)
    x = torch.randn(64, 3) * 8
    ref = L63Model(L63Params()).step(x)
    assert torch.equal(step(x), ref), desc


def test_direction_and_distance():
    reg = load_registry(eps1=0.01, eps2=0.05)
    p1, p2 = model_params("f1", reg), model_params("f2", reg)
    f = 1 + 0.01 / math.sqrt(3)
    assert abs(p1.sigma - 10 * f) < 1e-12 and abs(p1.rho - 28 * f) < 1e-12 and abs(p1.beta - 8 / 3 * f) < 1e-12
    # same line in relative parameter space: (theta_F2 / theta_L - 1) = 5 x (theta_F1 / theta_L - 1)
    for a, b, t in ((p1.sigma, p2.sigma, 10), (p1.rho, p2.rho, 28), (p1.beta, p2.beta, 8 / 3)):
        assert abs((b / t - 1) - 5 * (a / t - 1)) < 1e-12
    assert abs(theta_distance("f1", reg)["rel"] - 0.01) < 1e-12
    assert theta_distance("f2", reg)["abs"] > theta_distance("f1", reg)["abs"]
    # emulator teachers resolve to their false model
    assert teacher_config("e1", reg).rho == p1.rho and teacher_config("e2", reg).rho == p2.rho


def test_unset_eps_fails_loudly():
    reg = load_registry()
    if reg.eps1 is None:
        try:
            model_params("f1", reg)
        except ValueError:
            return
        raise AssertionError("expected ValueError for unset eps1")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
