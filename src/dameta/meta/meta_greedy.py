"""
M5 cell 3 — learned OPTIMIZER, UNSUPERVISED, NO gradient input: a free update rule iterated like an optimizer,

    x^{k+1} = x^k - Delta_theta(x^k, k [, obs]),   Delta = step_scale * ctrl_std * U_theta(x_std, emb(k) [, obs])

trained ONLINE across training ICs (episodes as Meta-B: batch of cases x episode_iters iterations, theta and its
Adam state persist) on the greedy one-step cost ratio  l = mean_cases J(x^k - Delta_theta) / J(x^k)  — each meta
step backpropagates through the model once (training only). Inference: frozen theta, iterate until the classical
stopping rule (relative J decrease < tol) or max_iter; cost = one FORWARD run per iteration (J for the stopping rule),
counted in n_fwd_evals; no gradient evaluation. Best epoch on val: median J after select_k frozen iterations.
"""

import copy
import time
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import numpy as np
import torch

from ..da.cost import FourDVarCost
from .meta_a import obs_features
from .networks import FreeUpdateNet

TINY = torch.finfo(torch.float64).tiny


@dataclass
class GreedyConfig:
    obs_input: bool = False
    step_scale: float = 0.1
    meta_lr: float = 1e-3
    num_meta_steps: int = 1          # Adam steps on theta per outer iteration
    n_epochs: int = 3
    batch: int = 100
    episode_iters: int = 100
    grad_clip: float = 1.0
    select_k: int = 10
    seed: int = 0


class GreedyOptimizer:
    def __init__(self, net: FreeUpdateNet, cfg: GreedyConfig, ctrl_mean, ctrl_std):
        self.net, self.cfg, self.ctrl_mean, self.ctrl_std = net, cfg, ctrl_mean, ctrl_std
        self.opt = torch.optim.Adam(net.parameters(), lr=cfg.meta_lr)

    def delta(self, x, k, obs):
        return self.cfg.step_scale * self.ctrl_std * self.net((x - self.ctrl_mean) / self.ctrl_std, None, k, obs)

    def _obs(self, cost):
        return obs_features(cost, self.ctrl_mean, self.ctrl_std) if self.cfg.obs_input else None

    def episode(self, cost: FourDVarCost, x0: torch.Tensor) -> dict:
        """One online training episode (theta updated before every move)."""
        obs, x, ratios = self._obs(cost), x0.detach(), []
        self.net.train()
        for k in range(self.cfg.episode_iters):
            with torch.no_grad():
                j = cost(*cost.unpack(x))
            for _ in range(self.cfg.num_meta_steps):
                self.opt.zero_grad()
                l = (cost(*cost.unpack(x - self.delta(x, k, obs))) / j.clamp_min(TINY)).mean()
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
                self.opt.step()
            ratios.append(float(l.detach()))
            with torch.no_grad():
                x = x - self.delta(x, k, obs)
        self.net.eval()
        with torch.no_grad():
            j_end = cost(*cost.unpack(x))
        return {"ratio_mean": float(np.mean(ratios)), "J_end_med": float(j_end.median())}

    def fit(self, make_cost: Callable, x0_train, val_cost, x0_val, log=None) -> list:
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        n, hist, best = x0_train.shape[0], [], (float("inf"), None, -1)
        for ep in range(cfg.n_epochs):
            perm = rng.permutation(n)
            for b0 in range(0, n, cfg.batch):
                idx = np.sort(perm[b0:b0 + cfg.batch])
                t0 = time.perf_counter()
                h = {"epoch": ep, "batch": b0 // cfg.batch, **self.episode(make_cost(idx), x0_train[idx]),
                     "s": time.perf_counter() - t0}
                hist.append(h)
                if log is not None:
                    log(h)
            r = run_greedy(val_cost, None, self, max_iter=cfg.select_k, tol=0.0, x0_ctrl=x0_val, diag=False)
            score = float(np.nanmedian(r["J_iter"][:, -1]))
            if score < best[0]:
                best = (score, copy.deepcopy(self.net.state_dict()), ep)
            row = {"epoch": ep, "val_score": score, "best_epoch": best[2]}
            hist.append(row)
            if log is not None:
                log(row)
        self.net.load_state_dict(best[1])
        self.net.eval()
        return hist

    def state_dict(self) -> dict:
        n = self.net
        return {"net": n.state_dict(), "cfg": asdict(self.cfg), "ctrl_mean": self.ctrl_mean, "ctrl_std": self.ctrl_std,
                "net_kwargs": dict(d=n.d, use_grad=False, d_obs=n.d_obs, hidden=n.head.in_features,
                                   n_layers=sum(isinstance(l, torch.nn.Linear) for l in n.body),
                                   emb_dim=n.emb.net[2].out_features, output_scale=n.output_scale,
                                   log_k=n.emb.log_input)}

    @classmethod
    def from_state_dict(cls, sd: dict) -> "GreedyOptimizer":
        net = FreeUpdateNet(**sd["net_kwargs"], zero_init=False).to(sd["ctrl_mean"].dtype)
        net.load_state_dict(sd["net"])
        net.eval()
        return cls(net, GreedyConfig(**sd["cfg"]), sd["ctrl_mean"], sd["ctrl_std"])


def build_greedy(d: int, cfg: GreedyConfig, ctrl_mean, ctrl_std, n_obs: int = 11, **net_kwargs) -> GreedyOptimizer:
    torch.manual_seed(cfg.seed)
    net = FreeUpdateNet(d, use_grad=False, d_obs=2 * n_obs * 3 if cfg.obs_input else 0, **net_kwargs)
    return GreedyOptimizer(net.to(ctrl_mean.dtype), cfg, ctrl_mean, ctrl_std)


@torch.no_grad()
def run_greedy(cost: FourDVarCost, x0_init, learner: GreedyOptimizer, max_iter: int = 1000, tol: float = 1e-6,
               eta_init=None, x0_ctrl: Optional[torch.Tensor] = None, diag: bool = True) -> dict:
    """Frozen inference with the classical stopping rule; record compatible with run_fourdvar (+ n_fwd_evals)."""
    if x0_ctrl is None:
        n = x0_init.shape[0]
        if cost.weak and eta_init is None:
            eta_init = torch.zeros(n, cost.cfg.n_obs - 1, 3, dtype=x0_init.dtype)
        x0_ctrl = cost.pack(x0_init, eta_init)
    x, n = x0_ctrl.detach().clone(), x0_ctrl.shape[0]
    obs = learner._obs(cost)
    active, n_fwd = torch.ones(n, dtype=x.dtype), torch.zeros(n, dtype=torch.long)
    x0_hist, j_hist, cos_hist = [x[:, :3].clone()], [], []
    t0, j_prev = time.perf_counter(), None
    for k in range(max_iter):
        j = cost(*cost.unpack(x))
        n_fwd += active.long()
        j_hist.append(j.clone() if j_prev is None else torch.where(active.bool(), j, j_hist[-1]))
        if j_prev is not None:
            rel = (j_prev - j).abs() / j_prev.abs().clamp_min(1e-300)
            active = active * (~(active.bool() & ((rel < tol) | ~torch.isfinite(j)))).to(x.dtype)
        if active.sum() == 0:
            break
        step = learner.delta(x, k, obs) * active[:, None]
        if diag:                                              # G2 vs the true gradient: diagnostics, not counted
            with torch.enable_grad():
                g = cost.grad(x)[1]
            cos_hist.append(torch.nn.functional.cosine_similarity(step, g, dim=1))
        x = x - step
        x0_hist.append(x[:, :3].clone())
        j_prev = j
    wall = time.perf_counter() - t0
    j_final = cost(*cost.unpack(x))
    j_hist.append(j_final)
    k_run = len(x0_hist) - 1
    pad = max_iter - k_run
    x0_iter = torch.stack(x0_hist, 1)
    x0_iter = torch.cat([x0_iter, x0_iter[:, -1:].expand(n, pad, 3)], 1) if pad > 0 else x0_iter
    j_iter = torch.stack(j_hist[: k_run + 1], 1)
    j_iter = torch.cat([j_iter, j_iter[:, -1:].expand(n, pad)], 1) if pad > 0 else j_iter
    cos = torch.stack(cos_hist, 1) if cos_hist else torch.zeros(n, 0, dtype=x.dtype)
    cos = torch.cat([cos, torch.full((n, max_iter - cos.shape[1]), float("nan"), dtype=x.dtype)], 1)
    out = {"x0_iter": x0_iter.numpy(), "J_iter": j_iter.numpy(), "cos_step_grad": cos.numpy(),
           "n_grad_evals": np.zeros(n, dtype=np.int64), "n_fwd_evals": n_fwd.numpy(),
           "converged": ((active == 0) & (n_fwd < max_iter) & torch.isfinite(j_final)).numpy(),
           "x0_a": x[:, :3].numpy(), "wall_s": wall}
    if cost.weak:
        out["eta_final"] = cost.unpack(x)[1].numpy()
    return out
