"""
Adjoint-free gradient providers (M5 baselines for the cell-8 claim: "IC estimation without the exact adjoint of
the model"). Each wraps a FourDVarCost and exposes the interface run_fourdvar uses (weak, cfg, pack, unpack,
__call__, grad, n_grad_evals), so the classical outer loop is unchanged. Model-run accounting per gradient call:

    FDGradient      central finite differences of J of the wrapped (forward-only) model:
                    2 d + 1 forward runs, no adjoint (d = control size: 3 strong, 3 + 3 (n_obs - 1) weak)
    HybridGradient  J from the wrapped model (1 forward run, e.g. F1 = knowledge numerical model) and the gradient
                    from a SURROGATE cost (e.g. the emulator: 1 forward + 1 adjoint run of the surrogate)

n_fwd_model counts forward runs of the wrapped (non-differentiated) model; n_surrogate_grads counts surrogate
gradient evaluations. n_grad_evals keeps counting outer gradient calls (iterations).
"""

import torch

from .cost import FourDVarCost


class _Wrapped:
    def __init__(self, cost: FourDVarCost):
        self.cost = cost
        self.weak, self.cfg = cost.weak, cost.cfg
        self.n_grad_evals, self.n_fwd_model = 0, 0

    def pack(self, *a):
        return self.cost.pack(*a)

    def unpack(self, c):
        return self.cost.unpack(c)

    def __call__(self, *a):
        return self.cost(*a)

    def terms(self, *a):
        return self.cost.terms(*a)


class FDGradient(_Wrapped):
    def __init__(self, cost: FourDVarCost, eps: float = 1e-6):
        super().__init__(cost)
        self.eps = eps

    @torch.no_grad()
    def grad(self, control: torch.Tensor) -> tuple:
        c = control.detach()
        j = self.cost(*self.cost.unpack(c))
        g = torch.empty_like(c)
        for i in range(c.shape[1]):
            h = self.eps * torch.clamp(c[:, i].abs(), min=1.0)
            cp, cm = c.clone(), c.clone()
            cp[:, i] += h
            cm[:, i] -= h
            g[:, i] = (self.cost(*self.cost.unpack(cp)) - self.cost(*self.cost.unpack(cm))) / (2 * h)
        self.n_grad_evals += 1
        self.n_fwd_model += 2 * c.shape[1] + 1
        return j, g


class HybridGradient(_Wrapped):
    def __init__(self, cost: FourDVarCost, surrogate: FourDVarCost):
        super().__init__(cost)
        self.surrogate = surrogate
        self.n_surrogate_grads = 0

    def grad(self, control: torch.Tensor) -> tuple:
        with torch.no_grad():
            j = self.cost(*self.cost.unpack(control.detach()))
        _, g = self.surrogate.grad(control)
        self.n_grad_evals += 1
        self.n_fwd_model += 1
        self.n_surrogate_grads += 1
        return j, g
