"""
Model rectangle registry (pilot 2026-11-20): the 5 dynamical models of the study, defined in src/configs/models.yaml.

    L1  true Lorenz-63, theta_L = (sigma 10, rho 28, beta 8/3)           truth, observations, perfect-model reference
    F1  L63 with theta_F1 = theta_L * (1 + eps1 d)   (mild misspecification)
    F2  L63 with theta_F2 = theta_L * (1 + eps2 d)   (strong misspecification, same direction d, eps2 > eps1)
    E1  MLP-tendency + RK4 emulator trained on F1 trajectories
    E2  MLP-tendency + RK4 emulator trained on F2 trajectories

Vertical axis F1 -> F2 (and E1 -> E2) = physical model error (Problem 1); horizontal axis F -> E = emulator
approximation (Problem 2). d is a RELATIVE direction (element-wise), normalized to unit length.
"""

from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf

from .emulator import L63Emulator, TeacherConfig
from .l63 import L63Model, L63Params

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "configs" / "models.yaml"


def load_registry(path: Path = REGISTRY_PATH, **overrides) -> DictConfig:
    """Load models.yaml; overrides (e.g. eps1=0.01) replace top-level keys before interpolation."""
    reg = OmegaConf.load(path)
    for k, v in overrides.items():
        reg[k] = v
    return reg


def perturbed_params(theta_L, direction, eps: float, dt: float) -> L63Params:
    """theta_L * (1 + eps * d) with d = direction / ||direction|| (relative, element-wise perturbation)."""
    d = torch.tensor(list(direction), dtype=torch.float64)
    d = d / d.norm()
    base = torch.tensor([theta_L["sigma"], theta_L["rho"], theta_L["beta"]], dtype=torch.float64)
    th = base * (1.0 + float(eps) * d)
    return L63Params(sigma=float(th[0]), rho=float(th[1]), beta=float(th[2]), dt=float(dt))


def canonical(name: str, reg: DictConfig) -> str:
    name = name.lower()
    return str(reg.get("aliases", {}).get(name, name))


def model_params(name: str, reg: DictConfig) -> L63Params:
    """L63 parameters of an l63-kind model, or of the TEACHER of an emulator-kind model."""
    name = canonical(name, reg)
    spec = reg.models[name]
    if spec.kind == "emulator":
        return model_params(spec.teacher, reg)
    if spec.eps is None:
        raise ValueError(f"model {name}: eps not set in models.yaml (screen + pilot sign-off pending)")
    return perturbed_params(reg.theta_L, reg.direction, spec.eps, reg.dt)


def teacher_config(name: str, reg: DictConfig) -> TeacherConfig:
    """TeacherConfig generating the training trajectories of an emulator (or of an l63-kind model itself)."""
    p = model_params(name, reg)
    return TeacherConfig(sigma=p.sigma, rho=p.rho, beta=p.beta, dt=p.dt)


def build_model(name: str, reg: DictConfig, emulator_dir: Optional[str] = None
                ) -> Tuple[Callable[[torch.Tensor], torch.Tensor], str, Optional[L63Params]]:
    """(one-step map, description, L63 params or None for emulators). Emulator weights are frozen."""
    name = canonical(name, reg)
    spec = reg.models[name]
    if spec.kind == "l63":
        p = model_params(name, reg)
        desc = f"{name.upper()}: L63(sigma={p.sigma:.6g}, rho={p.rho:.6g}, beta={p.beta:.6g}, eps={float(spec.eps):g})"
        return L63Model(p).step, desc, p
    if spec.kind == "emulator":
        assert emulator_dir is not None, "emulator_dir needed for emulator models"
        emu = L63Emulator.load(Path(emulator_dir) / f"{spec.file}.pt")
        for q in emu.parameters():
            q.requires_grad_(False)
        assert abs(emu.cfg.dt - float(reg.dt)) < 1e-12, "emulator dt != registry dt"
        return emu.step, f"{name.upper()}: emulator {spec.file} (teacher {spec.teacher.upper()}, {emu.teacher})", None
    raise ValueError(f"unknown model kind {spec.kind!r} for {name}")


def theta_distance(name: str, reg: DictConfig) -> dict:
    """||theta - theta_L||_2 (absolute) and relative size eps of an l63-kind model (or an emulator's teacher)."""
    p = model_params(name, reg)
    t = reg.theta_L
    diff = torch.tensor([p.sigma - t.sigma, p.rho - t.rho, p.beta - t.beta])
    return {"abs": float(diff.norm()), "rel": float((diff / torch.tensor([t.sigma, t.rho, t.beta])).norm())}
