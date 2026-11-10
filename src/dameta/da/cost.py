"""
4D-Var cost functions (strong and weak constraint), batched over initial conditions.

Data shapes (n = number of IC cases solved in parallel; each case is an independent 4D-Var problem):
    x0      [n, 3]               control: initial condition
    eta     [n, n_obs - 1, 3]    weak-constraint control: model-error increments added at obs times t_obs[1:]
    y       [n, n_obs, 3]        observations (all variables stored; H applied by `mask`)
    mask    [n_obs, 3] bool      observation operator H as a selection mask (pattern "full" or "x_only")
    x0_b    [n, 3]               background
    returns J [n]                one cost per case (sum it before autograd: cases do not interact)

Costs (approved DA configuration v0, B = sigma_b^2 I, R = sigma_o2 I, Q = sigma_q^2 I):
    strong:  J(x0)      = 1/2 |x0 - x0_b|^2 / sigma_b^2 + 1/2 sum_i |H(x_{t_i}) - y_i|^2 / sigma_o2
    weak:    J(x0, eta) = strong terms + 1/2 sum_i |eta_i|^2 / sigma_q^2
             with the trajectory  x_{t+1} = M(x_t)  and, at every obs time t_i (i >= 1),  x_{t_i} <- x_{t_i} + eta_i

`step` is any differentiable resolvent x_t -> x_{t+1} on [n, 3] (M1 physical L63 now, M2 emulator in milestone 3),
so the same cost serves configurations E1-E4.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import torch

Step = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class CostConfig:
    """Error statistics and window geometry (defaults = approved DA configuration v0)."""

    window: int = 100       # L, model steps
    obs_every: int = 10     # obs at t = 0, obs_every, ..., L
    sigma_o2: float = 2.0   # R = sigma_o2 I
    sigma_b: float = 1.0    # B = sigma_b^2 I
    sigma_q: float = 0.1    # Q = sigma_q^2 I (weak constraint only)

    @property
    def n_obs(self) -> int:
        return self.window // self.obs_every + 1


def window_trajectory(step: Step, x0: torch.Tensor, cfg: CostConfig, eta: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Analysis-window trajectory [n, L + 1, 3]. With eta (weak constraint), eta_i is added to the state at obs time
    t_i = i * obs_every (i = 1 .. n_obs - 1) before that state is observed and propagated further.
    """
    states, x = [x0], x0
    for t in range(1, cfg.window + 1):
        x = step(x)
        if eta is not None and t % cfg.obs_every == 0:
            x = x + eta[:, t // cfg.obs_every - 1]
        states.append(x)
    return torch.stack(states, dim=1)


class FourDVarCost:
    """J(x0[, eta]) per case, plus its term breakdown. Weak constraint iff `weak=True`."""

    def __init__(self, step: Step, y: torch.Tensor, mask: torch.Tensor, x0_b: torch.Tensor, cfg: CostConfig,
                 weak: bool = False):
        self.step, self.y, self.x0_b, self.cfg, self.weak = step, y, x0_b, cfg, weak
        self.mask = mask.to(y.dtype)                          # [n_obs, 3], 1 = observed
        self.t_obs = torch.arange(0, cfg.window + 1, cfg.obs_every)
        self.n_grad_evals = 0                                 # cost counter: forward + adjoint runs (see grad())

    def terms(self, x0: torch.Tensor, eta: Optional[torch.Tensor] = None) -> dict:
        """Return {"Jb", "Jo", "Jq", "J"} each [n]."""
        traj = window_trajectory(self.step, x0, self.cfg, eta if self.weak else None)
        innov = (traj[:, self.t_obs] - self.y) * self.mask    # H(x) - y on observed components only
        jb = 0.5 * ((x0 - self.x0_b) ** 2).sum(-1) / self.cfg.sigma_b**2
        jo = 0.5 * (innov**2).sum((-1, -2)) / self.cfg.sigma_o2
        jq = 0.5 * (eta**2).sum((-1, -2)) / self.cfg.sigma_q**2 if self.weak else torch.zeros_like(jb)
        return {"Jb": jb, "Jo": jo, "Jq": jq, "J": jb + jo + jq}

    def __call__(self, x0: torch.Tensor, eta: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.terms(x0, eta)["J"]

    def grad(self, control: torch.Tensor) -> tuple:
        """
        J and dJ/d(control) for the packed control (see pack/unpack). One call = one forward model run plus one
        adjoint (reverse-mode) run per case: this is the unit of computational cost counted in n_grad_evals.
        """
        c = control.detach().requires_grad_(True)
        j = self(*self.unpack(c))
        (g,) = torch.autograd.grad(j.sum(), c)
        self.n_grad_evals += 1
        return j.detach(), g.detach()

    # Packed control vector [n, d]: strong d = 3 (x0); weak d = 3 + 3 (n_obs - 1) (x0 then eta, flattened).
    def pack(self, x0: torch.Tensor, eta: Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.cat([x0, eta.flatten(1)], dim=1) if self.weak else x0

    def unpack(self, control: torch.Tensor) -> tuple:
        if not self.weak:
            return (control,)
        return control[:, :3], control[:, 3:].reshape(control.shape[0], self.cfg.n_obs - 1, 3)
