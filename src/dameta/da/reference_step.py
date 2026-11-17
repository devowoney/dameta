"""
Reference (teacher) update rule f of  c^{k+1} = c^k - f(c^k)  — ported from bice `gd_optimic/ic_optimizer.py`.

    method "gd"      : f = lr * g
    method "adagrad" : f = torch.optim.Adagrad step on g, with its own state (sum of g^2), advanced once per call

How it works for torch optimizers (as in bice): the optimizer owns a private "shadow" copy of the control. Each call
copies c^k into it, takes one optimizer step with g as its gradient, and reads back f = c^k - shadow. The real control
is never touched. Call exactly once per outer gradient so the optimizer state evolves exactly as in the baseline run.
The same object is the classical optimizer of E1/E3 AND the teacher step stretched by Meta-B (milestone 4).

Changes vs bice: ocean mask removed; optional `active` mask [n] freezes converged cases (their step is set to 0).
Adagrad accumulators are element-wise, so batched cases stay independent.
"""

from typing import Optional

import torch


class LBFGSDirection:
    """
    Batched L-BFGS direction (2026-11-16, teacher option "lbfgs"): per case, two-loop recursion d = H_k g_k over the
    last `history` curvature pairs s = c^k - c^{k-1}, y = g^k - g^{k-1} taken from the moves ACTUALLY applied (so it
    stays consistent when a learner rescales the step). A pair is kept only if s.y > eps |s| |y| (H stays positive
    definite -> g.d > 0). H_0 = gamma I, gamma = s.y / y.y of the newest pair; first iteration d = g / |g| (unit
    step, as torch.optim.LBFGS). No line search: the caller applies f = lr * d.
    """

    def __init__(self, history: int = 10, eps: float = 1e-10):
        self.m, self.eps = history, eps
        self.S, self.Y = [], []                       # lists of [n, d]; pair validity encoded by zeroed rows
        self.c_prev, self.g_prev = None, None

    def __call__(self, c: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if self.c_prev is not None:
            s, y = c - self.c_prev, g - self.g_prev
            sy = (s * y).sum(1)
            ok = (sy > self.eps * s.norm(dim=1) * y.norm(dim=1)) & (s.norm(dim=1) > 0)
            self.S.append(s * ok[:, None])
            self.Y.append(y * ok[:, None])
            self.S, self.Y = self.S[-self.m:], self.Y[-self.m:]
        self.c_prev, self.g_prev = c.clone(), g.clone()
        if not self.S:
            return g / g.norm(dim=1, keepdim=True).clamp_min(1e-300)
        q, alphas = g.clone(), []
        for s, y in zip(reversed(self.S), reversed(self.Y)):
            sy = (s * y).sum(1)
            rho = torch.where(sy > 0, 1.0 / sy.clamp_min(1e-300), torch.zeros_like(sy))   # invalid pair -> no-op
            a = rho * (s * q).sum(1)
            q = q - a[:, None] * y
            alphas.append((a, rho))
        # gamma from the newest VALID pair per case (fallback 1 / |g|: unit-length first-step scale)
        gamma = 1.0 / g.norm(dim=1).clamp_min(1e-300)
        for s, y in zip(self.S, self.Y):
            sy, yy = (s * y).sum(1), (y * y).sum(1)
            gamma = torch.where(sy > 0, sy / yy.clamp_min(1e-300), gamma)
        r = gamma[:, None] * q
        for (s, y), (a, rho) in zip(zip(self.S, self.Y), reversed(alphas)):
            b = rho * (y * r).sum(1)
            r = r + (a - b)[:, None] * s
        return r


class ReferenceStep:
    def __init__(self, control_like: torch.Tensor, method: str = "gd", lr: float = 1e-3, **optim_kwargs):
        self.method, self.lr = method.lower(), float(lr)
        self.optimizer, self._shadow, self._lbfgs = None, None, None
        if self.method == "adagrad":
            self._shadow = control_like.detach().clone().requires_grad_(True)  # private leaf tensor (holds state)
            self.optimizer = torch.optim.Adagrad([self._shadow], lr=self.lr, **optim_kwargs)
        elif self.method == "lbfgs":
            self._lbfgs = LBFGSDirection(**optim_kwargs)
        elif self.method != "gd":
            raise ValueError(f"unknown reference method {method!r} (expected 'gd', 'adagrad' or 'lbfgs')")

    def describe(self) -> str:
        name = "L-BFGS(fixed step)" if self._lbfgs else ("GD" if self.optimizer is None else "Adagrad")
        return f"{name}(lr={self.lr:g})"

    @torch.no_grad()
    def __call__(self, control: torch.Tensor, grad: torch.Tensor, active: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return f(c^k) [n, d] for control c^k and gradient g^k; zero rows where active is False."""
        if self._lbfgs is not None:
            step = self.lr * self._lbfgs(control.detach(), grad.detach())
        elif self.optimizer is None:
            step = self.lr * grad
        else:
            g = grad if active is None else grad * active[:, None]   # frozen cases: no state accumulation
            self._shadow.copy_(control.detach())
            self._shadow.grad = g.detach().clone()
            self.optimizer.step()
            step = control.detach() - self._shadow
            self._shadow.grad = None
        if active is not None:
            step = step * active[:, None]
        return step.detach()
