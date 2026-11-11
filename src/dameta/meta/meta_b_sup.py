"""
Meta-B, SUPERVISED variant (pilot 2026-11-11): imitation of the teacher's running gradient from the running state.

Training data = the classical teacher run (run_fourdvar, same method / lr / stopping rule as E1/E3) on TRAIN ICs:
every (running state c^k, iteration k, running gradient g^k = dJ/dc(c^k)) of every still-active case.
Model:   g_hat = G_theta(x_std(c^k), emb(k)) * g_scale            (inputs: state + iteration ONLY)
Loss:    MSE( G_theta(...), g^k / g_scale )                       (standardised units, per component)
         g_scale = per-component std of g over the training pairs; best epoch selected on VAL-IC teacher pairs.
Inference (frozen):   c^{k+1} = c^k - f(c^k; g_hat)   with the same ReferenceStep teacher fed g_hat
         (Adagrad state is built from the predicted gradients). NO gradient evaluation is used: J for the stopping
         rule (same rule as run_fourdvar) costs one forward run per iteration, counted in n_fwd_evals.
Diagnostics (not counted as cost, flag diag_true_grad): cos(step, true grad) = G2, cos(g_hat, true grad).

Note: G_theta sees neither the observations nor the gradient, so the case-specific minimum is not identifiable from
its inputs; it can only learn the average teacher behaviour around the attractor. Reported as-is.
"""

import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch

from ..da.cost import FourDVarCost
from ..da.fourdvar import run_fourdvar
from ..da.reference_step import ReferenceStep
from .networks import GradNet


@dataclass
class SupConfig:
    hidden: int = 128
    n_layers: int = 3
    emb_dim: int = 8
    log_k: bool = True
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 50
    batch_size: int = 1024
    seed: int = 0


def collect_teacher_pairs(cost: FourDVarCost, x0_b: torch.Tensor, method: str, lr: float, max_iter: int = 1000,
                          tol: float = 1e-6) -> dict:
    """(c^k, k, g^k) of every active case along the classical teacher run. Returns tensors c [N, d], k [N], g [N, d]."""
    cs, ks, gs = [], [], []

    def hook(k, c, g, active):
        a = active.bool()
        cs.append(c[a].clone())
        gs.append(g[a].clone())
        ks.append(torch.full((int(a.sum()),), k, dtype=c.dtype))

    run_fourdvar(cost, x0_b, method, lr, max_iter, tol, on_step=hook)
    return {"c": torch.cat(cs), "k": torch.cat(ks), "g": torch.cat(gs)}


class MetaBSup:
    def __init__(self, net: GradNet, cfg: SupConfig, ctrl_mean: torch.Tensor, ctrl_std: torch.Tensor,
                 g_scale: torch.Tensor):
        self.net, self.cfg = net, cfg
        self.ctrl_mean, self.ctrl_std, self.g_scale = ctrl_mean, ctrl_std, g_scale

    def predict_std(self, c: torch.Tensor, k) -> torch.Tensor:
        return self.net((c - self.ctrl_mean) / self.ctrl_std, k)

    @torch.no_grad()
    def predict(self, c: torch.Tensor, k) -> torch.Tensor:
        """Predicted running gradient g_hat [n, d] (physical units)."""
        self.net.eval()
        return self.predict_std(c, k) * self.g_scale

    def fit(self, train: dict, val: dict, log=None) -> list:
        """Mini-batch Adam on standardised MSE; keeps the best-val-epoch weights. Returns per-epoch history."""
        cfg = self.cfg
        gen = torch.Generator().manual_seed(cfg.seed)
        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        yt, yv = train["g"] / self.g_scale, val["g"] / self.g_scale
        best, best_state, hist = float("inf"), None, []
        n = yt.shape[0]
        for ep in range(cfg.epochs):
            t0 = time.perf_counter()
            self.net.train()
            perm, tot = torch.randperm(n, generator=gen), 0.0
            for b0 in range(0, n, cfg.batch_size):
                i = perm[b0:b0 + cfg.batch_size]
                opt.zero_grad()
                loss = (self.predict_std(train["c"][i], train["k"][i]) - yt[i]).pow(2).mean()
                loss.backward()
                opt.step()
                tot += float(loss.detach()) * len(i)
            self.net.eval()
            with torch.no_grad():
                pv = self.predict_std(val["c"], val["k"])
                vl = float((pv - yv).pow(2).mean())
                vcos = float(torch.nn.functional.cosine_similarity(pv * self.g_scale, val["g"], dim=1).median())
            h = {"epoch": ep, "train_mse": tot / n, "val_mse": vl, "val_cos_med": vcos, "s": time.perf_counter() - t0}
            hist.append(h)
            if log is not None:
                log(h)
            if vl < best:
                best, best_state = vl, {k: v.clone() for k, v in self.net.state_dict().items()}
        self.net.load_state_dict(best_state)
        self.net.eval()
        return hist

    def state_dict(self) -> dict:
        n = self.net
        return {"net": n.state_dict(), "cfg": asdict(self.cfg), "ctrl_mean": self.ctrl_mean, "ctrl_std": self.ctrl_std,
                "g_scale": self.g_scale, "net_kwargs": dict(d=n.d, hidden=self.cfg.hidden, n_layers=self.cfg.n_layers,
                                                             emb_dim=self.cfg.emb_dim, log_k=self.cfg.log_k)}

    @classmethod
    def from_state_dict(cls, sd: dict) -> "MetaBSup":
        net = GradNet(**sd["net_kwargs"]).to(sd["ctrl_mean"].dtype)
        net.load_state_dict(sd["net"])
        net.eval()
        return cls(net, SupConfig(**sd["cfg"]), sd["ctrl_mean"], sd["ctrl_std"], sd["g_scale"])


def build_sup(d: int, cfg: SupConfig, ctrl_mean: torch.Tensor, ctrl_std: torch.Tensor, train_pairs: dict) -> MetaBSup:
    torch.manual_seed(cfg.seed)
    net = GradNet(d, cfg.hidden, cfg.n_layers, cfg.emb_dim, cfg.log_k).to(ctrl_mean.dtype)
    g_scale = train_pairs["g"].std(0).clamp_min(1e-12)
    return MetaBSup(net, cfg, ctrl_mean, ctrl_std, g_scale)


@torch.no_grad()
def run_meta_b_sup(cost: FourDVarCost, x0_init: torch.Tensor, learner: MetaBSup, method: str, lr: float,
                   max_iter: int = 1000, tol: float = 1e-6, eta_init: Optional[torch.Tensor] = None,
                   diag_true_grad: bool = True) -> dict:
    """Frozen supervised Meta-B outer loop; record compatible with run_fourdvar (+ n_fwd_evals, cos_ghat_grad)."""
    n = x0_init.shape[0]
    if cost.weak and eta_init is None:
        eta_init = torch.zeros(n, cost.cfg.n_obs - 1, 3, dtype=x0_init.dtype)
    c = cost.pack(x0_init, eta_init).detach().clone()
    teacher = ReferenceStep(c, method=method, lr=lr)
    active = torch.ones(n, dtype=c.dtype)
    n_fwd = torch.zeros(n, dtype=torch.long)
    x0_hist, j_hist, cos_hist, cosg_hist = [c[:, :3].clone()], [], [], []
    t0, j_prev = time.perf_counter(), None
    for k in range(max_iter):
        j = cost(*cost.unpack(c))
        n_fwd += active.long()
        j_hist.append(j.clone() if j_prev is None else torch.where(active.bool(), j, j_hist[-1]))
        if j_prev is not None:
            rel = (j_prev - j).abs() / j_prev.abs().clamp_min(1e-300)
            newly_done = active.bool() & ((rel < tol) | ~torch.isfinite(j))
            active = active * (~newly_done).to(c.dtype)
        if active.sum() == 0:
            break
        g_hat = learner.predict(c, k)
        f = teacher(c, g_hat, active)
        if diag_true_grad:                                    # diagnostics only, NOT counted as cost
            with torch.enable_grad():
                _, g_true = cost.grad(c)
            cos_hist.append(torch.nn.functional.cosine_similarity(f, g_true, dim=1))
            cosg_hist.append(torch.nn.functional.cosine_similarity(g_hat, g_true, dim=1))
        c = c - f
        x0_hist.append(c[:, :3].clone())
        j_prev = j
    wall = time.perf_counter() - t0

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

    out = {
        "x0_iter": x0_iter.numpy(), "J_iter": j_iter.numpy(), "cos_step_grad": _pad(cos_hist),
        "cos_ghat_grad": _pad(cosg_hist), "n_grad_evals": np.zeros(n, dtype=np.int64), "n_fwd_evals": n_fwd.numpy(),
        "converged": ((active == 0) & (n_fwd < max_iter) & torch.isfinite(j_final)).numpy(),
        "x0_a": c[:, :3].numpy(), "wall_s": wall,
    }
    if cost.weak:
        out["eta_final"] = cost.unpack(c)[1].numpy()
    return out
