"""
M5 cell 8 — ADJOINT-FREE LEARNED GRADIENT (pilot 2026-11-14): IC estimation with a model that has no adjoint.

Only the observation term needs an adjoint: grad J_b = B^-1 (x0 - x0_b) and grad J_q = eta / sigma_q^2 are analytic.
The network predicts the observation-term gradient from FORWARD information only:

    d^k   = (H M_F(x^k)_{t_i} - y_i) / sigma_o  (masked)      innovations from ONE forward run of the false model F
    g_o^  = g_scale * G_theta(x_std^k, emb(k), [d^k, mask])   ~ grad J_o(x^k)
    g^    = grad J_b(x^k) [+ grad J_q] + g_o^                 fed to the GD / Adagrad teacher (ReferenceStep)

Learner input = running state x^k, iteration k, and the case information carried by the innovations (the observations
enter only through d^k, from a forward run — no adjoint). Supervised target: grad J_o of the EMULATOR (autodiff),
evaluated at the same x^k. Training states: visited by the adjoint-free hybrid classical run (J from F, gradient from
the emulator adjoint) on train ICs, so no adjoint of F is used anywhere. Best epoch on val pairs.
Inference cost: one forward run of F per iteration (J for the stopping rule and the innovations), counted in
n_fwd_evals; no gradient evaluation. Diagnostics (not counted): cos(g^, exact grad of F's cost).
"""

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from ..da.cost import FourDVarCost, window_trajectory
from ..da.fourdvar import run_fourdvar
from ..da.gradients import HybridGradient
from ..da.reference_step import ReferenceStep
from .networks import GradNet


@dataclass
class LGConfig:
    hidden: int = 128
    n_layers: int = 3
    emb_dim: int = 8
    log_k: bool = True
    lr: float = 1e-3
    epochs: int = 50
    batch_size: int = 1024
    seed: int = 0


@torch.no_grad()
def forward_features(cost: FourDVarCost, c: torch.Tensor):
    """ONE forward run: J [n] and innovation features [n, 2 * n_obs * 3] = (d / sigma_o (masked), mask)."""
    x0, *rest = cost.unpack(c)
    eta = rest[0] if cost.weak else None
    traj = window_trajectory(cost.step, x0, cost.cfg, eta)
    d = (traj[:, cost.t_obs] - cost.y) * cost.mask / cost.cfg.sigma_o2 ** 0.5
    jo = 0.5 * (d ** 2).sum((-1, -2))
    jq = 0.5 * (eta ** 2).sum((-1, -2)) / cost.cfg.sigma_q ** 2 if cost.weak else torch.zeros_like(jo)
    j = cost.jb(x0) + jo + jq
    m = cost.mask.expand(c.shape[0], -1, -1)
    return j, torch.cat([d.flatten(1), m.flatten(1)], 1)


def analytic_grad(cost: FourDVarCost, c: torch.Tensor) -> torch.Tensor:
    """grad of J_b (+ J_q) w.r.t. the packed control — no model involved."""
    x0, *rest = cost.unpack(c.detach())
    dx = x0 - cost.x0_b
    gb = torch.einsum("nij,nj->ni", cost.b_inv, dx) if cost.b_inv is not None else dx / cost.cfg.sigma_b ** 2
    if not cost.weak:
        return gb
    return torch.cat([gb, (rest[0] / cost.cfg.sigma_q ** 2).flatten(1)], 1)


def jo_grad(cost: FourDVarCost, c: torch.Tensor) -> torch.Tensor:
    """grad J_o by autodiff of `cost` (used with the EMULATOR cost only: training targets)."""
    cc = c.detach().requires_grad_(True)
    (g,) = torch.autograd.grad(cost.terms(*cost.unpack(cc))["Jo"].sum(), cc)
    return g.detach()


def collect_pairs(cost_f: FourDVarCost, cost_emu: FourDVarCost, x0_b: torch.Tensor, method: str, lr: float,
                  max_iter: int = 1000, tol: float = 1e-6, chunk: int = 4096) -> dict:
    """States of the adjoint-free hybrid classical run (J: F, gradient: emulator) -> (x, k, features, target)."""
    xs, ks, case = [], [], []

    def hook(k, c, g, active):
        a = active.bool()
        xs.append(c[a].clone())
        ks.append(torch.full((int(a.sum()),), k, dtype=c.dtype))
        case.append(torch.nonzero(a).flatten())

    run_fourdvar(HybridGradient(cost_f, cost_emu), x0_b, method, lr, max_iter, tol, on_step=hook)
    x, k, idx = torch.cat(xs), torch.cat(ks), torch.cat(case)
    feats, tgt = [], []
    for b0 in range(0, len(k), chunk):                       # per-pair costs: gather each pair's case data
        i = idx[b0:b0 + chunk]
        cf = _case_cost(cost_f, i)
        ce = _case_cost(cost_emu, i)
        feats.append(forward_features(cf, x[b0:b0 + chunk])[1])
        tgt.append(jo_grad(ce, x[b0:b0 + chunk]))
    return {"x": x, "k": k, "obs": torch.cat(feats), "g": torch.cat(tgt)}


def _case_cost(cost: FourDVarCost, idx) -> FourDVarCost:
    return FourDVarCost(cost.step, cost.y[idx], cost.mask, cost.x0_b[idx], cost.cfg, weak=cost.weak,
                        b_inv=None if cost.b_inv is None else cost.b_inv[idx])


class LearnedGradient:
    def __init__(self, net: GradNet, cfg: LGConfig, ctrl_mean, ctrl_std, g_scale):
        self.net, self.cfg, self.ctrl_mean, self.ctrl_std, self.g_scale = net, cfg, ctrl_mean, ctrl_std, g_scale

    def predict_std(self, x, k, obs):
        return self.net((x - self.ctrl_mean) / self.ctrl_std, k, obs)

    @torch.no_grad()
    def grad_obs(self, x, k, obs):
        self.net.eval()
        return self.predict_std(x, k, obs) * self.g_scale

    def fit(self, train: dict, val: dict, log=None) -> list:
        cfg, gen = self.cfg, torch.Generator().manual_seed(self.cfg.seed)
        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        yt, yv = train["g"] / self.g_scale, val["g"] / self.g_scale
        n, best, best_state, hist = yt.shape[0], float("inf"), None, []
        for ep in range(cfg.epochs):
            t0, tot = time.perf_counter(), 0.0
            self.net.train()
            perm = torch.randperm(n, generator=gen)
            for b0 in range(0, n, cfg.batch_size):
                i = perm[b0:b0 + cfg.batch_size]
                opt.zero_grad()
                loss = (self.predict_std(train["x"][i], train["k"][i], train["obs"][i]) - yt[i]).pow(2).mean()
                loss.backward()
                opt.step()
                tot += float(loss.detach()) * len(i)
            self.net.eval()
            with torch.no_grad():
                pv = self.predict_std(val["x"], val["k"], val["obs"])
                vl = float((pv - yv).pow(2).mean())
                vcos = float(torch.nn.functional.cosine_similarity(pv * self.g_scale, val["g"], dim=1).median())
            h = {"epoch": ep, "train_mse": tot / n, "val_mse": vl, "val_cos_med": vcos, "s": time.perf_counter() - t0}
            hist.append(h)
            if log is not None:
                log(h)
            if vl < best:
                best, best_state = vl, {key: v.clone() for key, v in self.net.state_dict().items()}
        self.net.load_state_dict(best_state)
        self.net.eval()
        return hist

    def state_dict(self) -> dict:
        n = self.net
        return {"net": n.state_dict(), "cfg": asdict(self.cfg), "ctrl_mean": self.ctrl_mean, "ctrl_std": self.ctrl_std,
                "g_scale": self.g_scale, "net_kwargs": dict(d=n.d, hidden=self.cfg.hidden, n_layers=self.cfg.n_layers,
                                                             emb_dim=self.cfg.emb_dim, log_k=self.cfg.log_k,
                                                             d_obs=n.d_obs)}

    @classmethod
    def from_state_dict(cls, sd: dict) -> "LearnedGradient":
        net = GradNet(**sd["net_kwargs"]).to(sd["ctrl_mean"].dtype)
        net.load_state_dict(sd["net"])
        net.eval()
        return cls(net, LGConfig(**sd["cfg"]), sd["ctrl_mean"], sd["ctrl_std"], sd["g_scale"])


def build_lg(d: int, n_obs: int, cfg: LGConfig, ctrl_mean, ctrl_std, train_pairs: dict) -> LearnedGradient:
    torch.manual_seed(cfg.seed)
    net = GradNet(d, cfg.hidden, cfg.n_layers, cfg.emb_dim, cfg.log_k, d_obs=2 * n_obs * 3).to(ctrl_mean.dtype)
    return LearnedGradient(net, cfg, ctrl_mean, ctrl_std, train_pairs["g"].std(0).clamp_min(1e-12))


def run_learned_gradient(cost: FourDVarCost, x0_init: torch.Tensor, learner: LearnedGradient, method: str, lr: float,
                         max_iter: int = 1000, tol: float = 1e-6, diag_exact: bool = True) -> dict:
    """Frozen inference; record compatible with run_fourdvar (+ n_fwd_evals, cos_ghat_grad)."""
    n = x0_init.shape[0]
    eta0 = torch.zeros(n, cost.cfg.n_obs - 1, 3, dtype=x0_init.dtype) if cost.weak else None
    c = cost.pack(x0_init, eta0).detach().clone()
    teacher = ReferenceStep(c, method=method, lr=lr)
    active, n_fwd = torch.ones(n, dtype=c.dtype), torch.zeros(n, dtype=torch.long)
    x0_hist, j_hist, cos_hist, cosg_hist = [c[:, :3].clone()], [], [], []
    t0, j_prev = time.perf_counter(), None
    for k in range(max_iter):
        j, obs = forward_features(cost, c)
        n_fwd += active.long()
        j_hist.append(j.clone() if j_prev is None else torch.where(active.bool(), j, j_hist[-1]))
        if j_prev is not None:
            rel = (j_prev - j).abs() / j_prev.abs().clamp_min(1e-300)
            active = active * (~(active.bool() & ((rel < tol) | ~torch.isfinite(j)))).to(c.dtype)
        if active.sum() == 0:
            break
        g_hat = analytic_grad(cost, c) + learner.grad_obs(c, k, obs)
        f = teacher(c, g_hat, active)
        if diag_exact:                                          # diagnostics only, NOT counted
            g_ex = cost.grad(c)[1]
            cos_hist.append(torch.nn.functional.cosine_similarity(f, g_ex, dim=1))
            cosg_hist.append(torch.nn.functional.cosine_similarity(g_hat, g_ex, dim=1))
        c = c - f
        x0_hist.append(c[:, :3].clone())
        j_prev = j
    wall = time.perf_counter() - t0
    with torch.no_grad():
        j_final = cost(*cost.unpack(c))
    j_hist.append(j_final)
    k_run = len(x0_hist) - 1
    pad = max_iter - k_run
    x0_iter = torch.stack(x0_hist, 1)
    x0_iter = torch.cat([x0_iter, x0_iter[:, -1:].expand(n, pad, 3)], 1) if pad > 0 else x0_iter
    j_iter = torch.stack(j_hist[: k_run + 1], 1)
    j_iter = torch.cat([j_iter, j_iter[:, -1:].expand(n, pad)], 1) if pad > 0 else j_iter

    def _pad(h):
        h = torch.stack(h, 1) if h else torch.zeros(n, 0, dtype=c.dtype)
        return torch.cat([h, torch.full((n, max_iter - h.shape[1]), float("nan"), dtype=c.dtype)], 1).numpy()

    out = {"x0_iter": x0_iter.numpy(), "J_iter": j_iter.numpy(), "cos_step_grad": _pad(cos_hist),
           "cos_ghat_grad": _pad(cosg_hist), "n_grad_evals": np.zeros(n, dtype=np.int64), "n_fwd_evals": n_fwd.numpy(),
           "converged": ((active == 0) & (n_fwd < max_iter) & torch.isfinite(j_final)).numpy(),
           "x0_a": c[:, :3].numpy(), "wall_s": wall, "control_final": c.numpy()}
    if cost.weak:
        out["eta_final"] = cost.unpack(c)[1].numpy()
    return out
