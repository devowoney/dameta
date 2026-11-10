"""
M1 — physical Lorenz-63 model integrated with classical RK4.

Data shape convention (used everywhere in dameta):
    state x: [..., 3]  (any leading batch dims; last dim = (x, y, z)), dtype float64
    trajectory: [..., T + 1, 3]  (time axis just before the variable axis; index 0 = initial condition)

The tangent-linear (TL) and adjoint models are NOT hand-coded: they are obtained from PyTorch autodiff
(torch.func.jvp / torch.func.vjp) through the exact RK4 scheme. They are therefore the exact derivatives of the
discrete model, which is what 4D-Var needs. src/tests/test_m1.py checks them (TL Taylor test, dot-product test).
"""

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class L63Params:
    """Lorenz-63 parameters and time step (classical chaotic regime by default)."""

    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    dt: float = 0.01


def l63_tendency(x: torch.Tensor, p: L63Params) -> torch.Tensor:
    """dx/dt of Lorenz-63 for a batch of states x [..., 3]."""
    x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
    dx1 = p.sigma * (x2 - x1)
    dx2 = x1 * (p.rho - x3) - x2
    dx3 = x1 * x2 - p.beta * x3
    return torch.stack([dx1, dx2, dx3], dim=-1)


def l63_jacobian(x: torch.Tensor, p: L63Params) -> torch.Tensor:
    """Analytic Jacobian d(tendency)/dx, shape [..., 3, 3]. Used only to validate autodiff in the tests."""
    x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
    zero = torch.zeros_like(x1)
    row1 = torch.stack([zero - p.sigma, zero + p.sigma, zero], dim=-1)
    row2 = torch.stack([p.rho - x3, zero - 1.0, -x1], dim=-1)
    row3 = torch.stack([x2, x1, zero - p.beta], dim=-1)
    return torch.stack([row1, row2, row3], dim=-2)


def rk4_step(f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, dt: float) -> torch.Tensor:
    """One classical 4th-order Runge-Kutta step of dx/dt = f(x). Shared by M1 and (later) the emulator M2."""
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class L63Model:
    """
    Discrete Lorenz-63 resolvent M: x_t -> x_{t+1} (one RK4 step of size dt).

    Differentiable end to end, so J(x0) built on rollout() can be differentiated with torch.autograd to obtain the
    exact adjoint gradient used by classical 4D-Var (E1) and by the meta learners (E2).
    """

    def __init__(self, params: L63Params = L63Params()):
        self.params = params

    def tendency(self, x: torch.Tensor) -> torch.Tensor:
        """Continuous-time vector field f(x)."""
        return l63_tendency(x, self.params)

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """x_{t+1} = M(x_t)."""
        return rk4_step(self.tendency, x, self.params.dt)

    def rollout(self, x0: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        Integrate n_steps from x0 [..., 3]; returns the full trajectory [..., n_steps + 1, 3] (index 0 = x0).
        Keeps the autograd graph (needed for 4D-Var gradients); wrap in torch.no_grad() for data generation.
        """
        states = [x0]
        x = x0
        for _ in range(n_steps):
            x = self.step(x)
            states.append(x)
        return torch.stack(states, dim=-2)

    def final_state(self, x0: torch.Tensor, n_steps: int) -> torch.Tensor:
        """M_{0->n}(x0) without storing the trajectory (used by TL/adjoint tests and spin-up)."""
        x = x0
        for _ in range(n_steps):
            x = self.step(x)
        return x
