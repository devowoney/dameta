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
    """
    Methods: "gd", "adagrad", "lbfgs" (fixed step lr * d) and "lbfgs_wolfe" (2026-11-17).

    "lbfgs_wolfe" = the classical REFERENCE L-BFGS step, i.e. one iteration of torch.optim.LBFGS with
    line_search_fn="strong_wolfe" and the settings of run_dense_v2.opt_lbfgs (history 10, lr 1, tolerance_change 1e-14,
    c1 1e-4, c2 0.9, max_ls 25): direction d = -H g (two-loop over the pairs of the moves actually applied), first
    iteration d = -g with initial t = min(1, 1 / |g|_1) * lr, then t = lr, accepted t from torch's own _strong_wolfe.
    Batched: the first trial t is evaluated for all cases in one batched gradient; cases that already satisfy the
    strong Wolfe conditions there (torch's first-trial exit) are accepted, the others call torch's _strong_wolfe
    per case (its evaluations are counted, the shared first trial once). Needs cost (FourDVarCost) and __call__(...,
    j=J(c)). Cost bookkeeping:
        n_ls [n]      cumulative line-search gradient evaluations (each trial = forward + adjoint)
        is_free(c)    cases whose control equals the last accepted point: their J and gradient are already known
                      from the line search (the classical method reuses them), so the outer gradient is not counted.
    """

    def __init__(self, control_like: torch.Tensor, method: str = "gd", lr: float = 1e-3, cost=None,
                 **optim_kwargs):
        self.method, self.lr = method.lower(), float(lr)
        self.optimizer, self._shadow, self._lbfgs = None, None, None
        self.cost, self._case_costs, self._acc, self._first = cost, {}, None, True
        self.n_ls = torch.zeros(control_like.shape[0], dtype=torch.long)
        if self.method == "lbfgs_wolfe":
            if cost is None:
                raise ValueError("method 'lbfgs_wolfe' needs cost")
            self._lbfgs = LBFGSDirection(**optim_kwargs)
        elif self.method == "adagrad":
            self._shadow = control_like.detach().clone().requires_grad_(True)  # private leaf tensor (holds state)
            self.optimizer = torch.optim.Adagrad([self._shadow], lr=self.lr, **optim_kwargs)
        elif self.method == "lbfgs":
            self._lbfgs = LBFGSDirection(**optim_kwargs)
        elif self.method != "gd":
            raise ValueError(f"unknown reference method {method!r} (expected gd, adagrad, lbfgs or lbfgs_wolfe)")

    def describe(self) -> str:
        if self.method == "lbfgs_wolfe":
            return "L-BFGS(strong Wolfe line search)"
        name = "L-BFGS(fixed step)" if self._lbfgs else ("GD" if self.optimizer is None else "Adagrad")
        return f"{name}(lr={self.lr:g})"

    def is_free(self, c: torch.Tensor) -> torch.Tensor:
        """[n] bool: c equals the last accepted line-search point (gradient already evaluated there)."""
        if self._acc is None:
            return torch.zeros(c.shape[0], dtype=torch.bool)
        tol = 1e-12 * (1.0 + self._acc.abs().amax(1))
        return (c - self._acc).abs().amax(1) <= tol

    @torch.no_grad()
    def __call__(self, control: torch.Tensor, grad: torch.Tensor, active: Optional[torch.Tensor] = None,
                 j: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return f(c^k) [n, d] for control c^k and gradient g^k; zero rows where active is False."""
        if self.method == "lbfgs_wolfe":
            step = self._wolfe(control.detach(), grad.detach(), j, active)
        elif self._lbfgs is not None:
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

    def _case_cost(self, i: int):
        if i not in self._case_costs:
            cs = self.cost
            self._case_costs[i] = type(cs)(cs.step, cs.y[i:i + 1], cs.mask, cs.x0_b[i:i + 1], cs.cfg, weak=cs.weak,
                                           b_inv=None if cs.b_inv is None else cs.b_inv[i:i + 1])
        return self._case_costs[i]

    def _wolfe(self, c, g, j, active):
        from torch.optim.lbfgs import _strong_wolfe
        if j is None:
            raise ValueError("method 'lbfgs_wolfe' needs the current cost j")
        hg = self._lbfgs(c, g)                                   # H g (LBFGSDirection keeps the applied-move pairs)
        if self._first:                                          # torch: d = -g, t = min(1, 1 / |g|_1) * lr
            hg = g.clone()
            t0 = torch.clamp(1.0 / g.abs().sum(1), max=1.0) * self.lr
        else:
            t0 = torch.full_like(j, self.lr)
        self._first = False
        d = -hg
        gtd = (g * d).sum(1)
        pend = torch.isfinite(j) & torch.isfinite(d).all(1) & (gtd < -1e-14)
        if active is not None:
            pend = pend & active.bool()
        t0 = torch.where(pend, t0, torch.zeros_like(t0))
        x_acc = c + t0[:, None] * d
        with torch.enable_grad():
            j_t, g_t = self.cost.grad(x_acc)                     # shared first trial, batched
        self.n_ls += pend.long()
        ok = pend & torch.isfinite(j_t) & (j_t <= j + 1e-4 * t0 * gtd) & ((g_t * d).sum(1).abs() <= -0.9 * gtd)
        for i in torch.nonzero(pend & ~ok).flatten().tolist():
            ci = self._case_cost(i)

            def obj(x, t, dd, ci=ci):
                with torch.enable_grad():
                    jj, gg = ci.grad((x + t * dd)[None])
                return float(jj[0]), gg[0]

            _, _, t, nev = _strong_wolfe(obj, c[i], float(t0[i]), d[i], float(j[i]), g[i], gtd[i],
                                         tolerance_change=1e-14, max_ls=25)
            self.n_ls[i] += int(nev) - 1                         # first trial already counted
            x_acc[i] = c[i] + float(t) * d[i]
        self._acc = x_acc.clone()
        return c - x_acc
