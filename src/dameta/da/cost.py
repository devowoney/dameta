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
    q_var: Optional[tuple] = None   # (q_x, q_y, q_z): diagonal Q (variances) per obs interval; overrides sigma_q
                                    # (M8 2026-11-26: per-model Q from the model's own one-interval error)

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
                 weak: bool = False, b_inv: Optional[torch.Tensor] = None):
        self.step, self.y, self.x0_b, self.cfg, self.weak = step, y, x0_b, cfg, weak
        self.b_inv = b_inv                                    # [n, 3, 3] case-specific B^-1 (M5); None = sigma_b^2 I
        self.mask = mask.to(y.dtype)                          # [n_obs, 3], 1 = observed
        self.t_obs = torch.arange(0, cfg.window + 1, cfg.obs_every)
        self.n_grad_evals = 0                                 # cost counter: forward + adjoint runs (see grad())

    def terms(self, x0: torch.Tensor, eta: Optional[torch.Tensor] = None) -> dict:
        """Return {"Jb", "Jo", "Jq", "J"} each [n]."""
        traj = window_trajectory(self.step, x0, self.cfg, eta if self.weak else None)
        innov = (traj[:, self.t_obs] - self.y) * self.mask    # H(x) - y on observed components only
        jb = self.jb(x0)
        jo = 0.5 * (innov**2).sum((-1, -2)) / self.cfg.sigma_o2
        jq = self.jq(eta) if self.weak else torch.zeros_like(jb)
        return {"Jb": jb, "Jo": jo, "Jq": jq, "J": jb + jo + jq}

    def jq(self, eta: torch.Tensor) -> torch.Tensor:
        """Model-error term: 1/2 sum_i eta_i^T Q^-1 eta_i, Q = sigma_q^2 I or diag(q_var)."""
        if self.cfg.q_var is None:
            return 0.5 * (eta**2).sum((-1, -2)) / self.cfg.sigma_q**2
        q = torch.as_tensor(self.cfg.q_var, dtype=eta.dtype)
        return 0.5 * (eta**2 / q).sum((-1, -2))

    def jb(self, x0: torch.Tensor) -> torch.Tensor:
        """Background term: 1/2 |x0 - x0_b|^2 / sigma_b^2, or 1/2 (x0 - x0_b)^T B^-1 (x0 - x0_b) with b_inv."""
        d = x0 - self.x0_b
        if self.b_inv is None:
            return 0.5 * (d**2).sum(-1) / self.cfg.sigma_b**2
        return 0.5 * torch.einsum("ni,nij,nj->n", d, self.b_inv, d)

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


class ParamFourDVarCost(FourDVarCost):
    """
    BPE (M5, F1 only): strong-constraint 4D-Var with the L63 parameters in the control,
        control c = [x0, theta],  theta = (sigma, rho, beta) per case,
        J(x0, theta) = J_b(x0) + J_o(trajectory of L63(theta)) + J_p,  J_p = 1/2 sum_k ((theta_k - theta_b,k) / s_k)^2
    The dynamics is the F1 knowledge-model structure (L63 RK4) with the CONTROL parameters; theta_b = F1's own
    parameters (prior), s = parameter prior std. Packed control [n, 6].
    """

    def __init__(self, y: torch.Tensor, mask: torch.Tensor, x0_b: torch.Tensor, cfg: CostConfig, theta_b: torch.Tensor,
                 theta_std: torch.Tensor, dt: float = 0.01, b_inv: Optional[torch.Tensor] = None):
        super().__init__(None, y, mask, x0_b, cfg, weak=False, b_inv=b_inv)
        self.theta_b, self.theta_std, self.dt = theta_b, theta_std, dt

    def step_theta(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        from ..dynamics.l63 import l63_tendency_theta, rk4_step
        return rk4_step(lambda v: l63_tendency_theta(v, theta), x, self.dt)

    def trajectory(self, x0: torch.Tensor, theta: torch.Tensor, n_steps: int) -> torch.Tensor:
        states, x = [x0], x0
        for _ in range(n_steps):
            x = self.step_theta(x, theta)
            states.append(x)
        return torch.stack(states, dim=1)

    def terms(self, x0: torch.Tensor, theta: torch.Tensor) -> dict:
        traj = self.trajectory(x0, theta, self.cfg.window)
        innov = (traj[:, self.t_obs] - self.y) * self.mask
        jb = self.jb(x0)
        jo = 0.5 * (innov**2).sum((-1, -2)) / self.cfg.sigma_o2
        jp = 0.5 * (((theta - self.theta_b) / self.theta_std) ** 2).sum(-1)
        return {"Jb": jb, "Jo": jo, "Jp": jp, "Jq": torch.zeros_like(jb), "J": jb + jo + jp}

    def __call__(self, x0: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return self.terms(x0, theta)["J"]

    def pack(self, x0: torch.Tensor, theta: Optional[torch.Tensor] = None) -> torch.Tensor:
        if theta is None:
            theta = self.theta_b.expand(x0.shape[0], 3)
        return torch.cat([x0, theta], dim=1)

    def unpack(self, control: torch.Tensor) -> tuple:
        return control[:, :3], control[:, 3:6]
