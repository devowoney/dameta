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


class ReferenceStep:
    def __init__(self, control_like: torch.Tensor, method: str = "gd", lr: float = 1e-3, **optim_kwargs):
        self.method, self.lr = method.lower(), float(lr)
        self.optimizer, self._shadow = None, None
        if self.method == "adagrad":
            self._shadow = control_like.detach().clone().requires_grad_(True)  # private leaf tensor (holds state)
            self.optimizer = torch.optim.Adagrad([self._shadow], lr=self.lr, **optim_kwargs)
        elif self.method != "gd":
            raise ValueError(f"unknown reference method {method!r} (expected 'gd' or 'adagrad')")

    def describe(self) -> str:
        return f"{'GD' if self.optimizer is None else 'Adagrad'}(lr={self.lr:g})"

    @torch.no_grad()
    def __call__(self, control: torch.Tensor, grad: torch.Tensor, active: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return f(c^k) [n, d] for control c^k and gradient g^k; zero rows where active is False."""
        if self.optimizer is None:
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
