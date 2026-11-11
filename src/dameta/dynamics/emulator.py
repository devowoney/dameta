"""
M2 — auto-differentiable ML emulator of Lorenz-63 (approved D2 design, 2026-11-10).

    dx/dt = h_phi(x),  h_phi = MLP 3 -> width -> width -> 3 with smooth activations (SiLU), RK4 at the same dt as M1.

Smooth activations keep the emulator Jacobian smooth (no ReLU kinks), which matters for 4D-Var gradients through the
emulator (gradient fidelity G1). Inputs are standardized with the training-state mean/std; outputs are scaled by the
training tendency std, so the MLP works with O(1) numbers. Everything is float64, like M1.

Shapes: same convention as M1 (state [..., 3]); `step` is a drop-in replacement for L63Model.step in 4D-Var.

Deliberate degradation ("missing dynamics", pilot 2026-11-10) is NOT inside the network: the emulator is trained on
trajectories of a modified teacher model started from the same D1 training initial conditions (see
`teacher_tendency`): parameters (sigma, rho, beta) can be biased and the x*z coupling in dy/dt can be scaled
(xz_scale < 1 removes part of that nonlinear term). The clean emulator uses the exact L63 teacher.
"""

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .l63 import rk4_step


@dataclass(frozen=True)
class TeacherConfig:
    """Model that generates the emulator's training trajectories (defaults = exact L63 = clean emulator)."""

    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    xz_scale: float = 1.0   # factor on the -x*z term of dy/dt (1 = exact; e.g. 0.9 drops 10 % of the coupling)
    dt: float = 0.01


def teacher_tendency(x: torch.Tensor, t: TeacherConfig) -> torch.Tensor:
    """Tendency of the (possibly degraded) teacher model; equals l63_tendency when TeacherConfig is default."""
    x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
    dx1 = t.sigma * (x2 - x1)
    dx2 = t.rho * x1 - t.xz_scale * x1 * x3 - x2
    dx3 = x1 * x2 - t.beta * x3
    return torch.stack([dx1, dx2, dx3], dim=-1)


@torch.no_grad()
def teacher_rollout(x0: torch.Tensor, n_steps: int, t: TeacherConfig) -> torch.Tensor:
    """Teacher trajectory [..., n_steps + 1, 3] from x0 (RK4, dt = t.dt)."""
    states, x = [x0], x0
    for _ in range(n_steps):
        x = rk4_step(lambda v: teacher_tendency(v, t), x, t.dt)
        states.append(x)
    return torch.stack(states, dim=-2)


@dataclass(frozen=True)
class EmulatorConfig:
    width: int = 64
    depth: int = 2          # hidden layers
    dt: float = 0.01


class L63Emulator(nn.Module):
    """RK4-wrapped neural tendency. Normalization buffers are set once from training data (set_normalization)."""

    def __init__(self, cfg: EmulatorConfig = EmulatorConfig()):
        super().__init__()
        self.cfg = cfg
        layers, d_in = [], 3
        for _ in range(cfg.depth):
            layers += [nn.Linear(d_in, cfg.width), nn.SiLU()]
            d_in = cfg.width
        layers.append(nn.Linear(d_in, 3))
        self.mlp = nn.Sequential(*layers).double()
        self.register_buffer("x_mean", torch.zeros(3, dtype=torch.float64))
        self.register_buffer("x_std", torch.ones(3, dtype=torch.float64))
        self.register_buffer("f_std", torch.ones(3, dtype=torch.float64))   # tendency scale

    @torch.no_grad()
    def set_normalization(self, traj: torch.Tensor) -> None:
        """traj [n, T, 3] training trajectories: state mean/std and finite-difference tendency std."""
        flat = traj.reshape(-1, 3)
        self.x_mean.copy_(flat.mean(0))
        self.x_std.copy_(flat.std(0))
        self.f_std.copy_(((traj[:, 1:] - traj[:, :-1]) / self.cfg.dt).reshape(-1, 3).std(0))

    def tendency(self, x: torch.Tensor) -> torch.Tensor:
        """h_phi(x) [..., 3]."""
        return self.f_std * self.mlp((x - self.x_mean) / self.x_std)

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """x_{t+1} = RK4(h_phi)(x_t) — drop-in replacement for L63Model.step."""
        return rk4_step(self.tendency, x, self.cfg.dt)

    def rollout(self, x0: torch.Tensor, n_steps: int) -> torch.Tensor:
        states, x = [x0], x0
        for _ in range(n_steps):
            x = self.step(x)
            states.append(x)
        return torch.stack(states, dim=-2)

    # --- persistence: one .pt file with weights, buffers, configs and the training record ---
    def save(self, path, teacher: TeacherConfig, record: dict) -> None:
        torch.save({"state_dict": self.state_dict(), "emulator_cfg": asdict(self.cfg), "teacher_cfg": asdict(teacher),
                    "record": record}, path)

    @classmethod
    def load(cls, path) -> "L63Emulator":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        emu = cls(EmulatorConfig(**ckpt["emulator_cfg"]))
        emu.load_state_dict(ckpt["state_dict"])
        emu.teacher = TeacherConfig(**ckpt["teacher_cfg"])   # provenance, kept on the object
        emu.record = ckpt["record"]
        return emu.eval()
