"""
Learned OPTIMIZATION (M5 cells 5 and 7; cell 5 = Meta-A of original.md): a free update rule unrolled for a fixed
budget of K steps and trained end-to-end on the 4D-Var cost after K steps (unsupervised: obs + background only).

    x^0 = first guess (background control: x0_b, eta = 0)          -- the learner never sees x0_b as an input
    x^{m+1} = x^m - Delta_m,   Delta_m = step_scale * ctrl_std * U_theta(x_std^m [, g_std^m], emb(m) [, obs])
    loss    = mean_cases J(x^K) / J(x^0)                              (scale-free across cases)
              loss="sum" (2026-11-16 speed variant): mean_cases mean_{m=1..K} J(x^m) / J(x^0)

use_grad=True  (cell 5): g^m = grad J(x^m), one gradient evaluation (forward + adjoint) per step; first order by
               default (g^m enters as data: the theta-gradient flows through the x-path and through the final J
               only); second_order=True keeps d g^m / d theta (Hessian-vector products through the model).
use_grad=False (cell 7): no gradient input; inference costs NO model call (K network evaluations). Training still
               backpropagates through the model once, through J(x^K).
obs_input=True (no-gradient option): appends the case observations y (standardised, masked) and their mask.

Inference is a fixed budget (no stopping rule): n_grad_evals = K (cell 5) or 0 (cell 7). J is evaluated at every
iterate for reporting / cost-to-target only (not counted), as the reporting forward of the other runners.

base="lbfgs" (2026-11-19, ICE table cell 3): the move is the L-BFGS direction of the learner's OWN moves plus the
learned bounded correction,
    x^{m+1} = x^m - (H(M^m) g^m + step_scale * ctrl_std * U_theta(...)),
H(M^m) g^m = two-loop recursion over the 10 most recent (s, y) pairs of the moves actually made (reference_step.
LBFGSDirection; unit-length first move). The memory is treated as data (first order). Zero-initialised U: the
untrained learner is fixed-step L-BFGS (lr 1); training (loss sum) learns where to deviate from it.

safeguard="backtrack" (2026-11-20, ICE cell 3, as cells 1 / 2 at inference): without a line search, fixed-step
L-BFGS blows up on some chaotic cases (J -> inf -> NaN), which poisoned the batch-mean loss (NaN weights from the first
epoch in the first ICE run). With backtrack, the gradient evaluation at a trial point x^{m+1} also gives J(x^{m+1});
a move with non-finite or increased J is REJECTED and halved from the last accepted point (no extra model run). The
memory, the next move and the loss / record use ACCEPTED iterates only; the last trial point (after K gradients) is
kept only if its J (reporting evaluation, not counted) is not larger. Accepted J <= J(x^0): the loss ratio is bounded.
"""

import copy
import time
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import numpy as np
import torch

from ..da.cost import FourDVarCost
from ..da.reference_step import LBFGSDirection
from .networks import FreeUpdateNet

TINY = torch.finfo(torch.float64).tiny


@dataclass
class UnrollConfig:
    K: int = 20
    use_grad: bool = True
    second_order: bool = False
    obs_input: bool = False
    step_scale: float = 0.1          # Delta = step_scale * ctrl_std * out, |out| <= output_scale
    lr: float = 1e-3
    epochs: int = 30
    batch: int = 100
    grad_clip: float = 1.0
    seed: int = 0
    loss: str = "final"              # final: J(x^K)/J(x^0) | sum: mean_m J(x^m)/J(x^0), m = 1..K (rewards fast descent)
    base: str = "none"               # none | lbfgs (ICE cell 3: move = L-BFGS direction of own moves + correction)
    safeguard: str = "none"          # none | backtrack (base lbfgs: reject + halve a move whose J did not decrease)


def standardize_grad(g: torch.Tensor, d: int) -> torch.Tensor:
    """g / RMS(g) per case and per control block (x0 | eta) — as Meta-B."""
    out = torch.empty_like(g)
    for a, b in [(0, 3)] + ([(3, d)] if d > 3 else []):
        blk = g[:, a:b]
        out[:, a:b] = blk / blk.pow(2).mean(1, keepdim=True).sqrt().clamp_min(TINY)
    return out


def obs_features(cost: FourDVarCost, ctrl_mean: torch.Tensor, ctrl_std: torch.Tensor) -> torch.Tensor:
    """[n, 2 * n_obs * 3]: standardised masked observations, then the mask (flag obs_input)."""
    n = cost.y.shape[0]
    m = cost.mask.expand(n, -1, -1)
    y = (cost.y - ctrl_mean[:3]) / ctrl_std[:3] * m
    return torch.cat([y.flatten(1), m.flatten(1)], 1)


class LearnedOptimization:
    def __init__(self, net: FreeUpdateNet, cfg: UnrollConfig, ctrl_mean: torch.Tensor, ctrl_std: torch.Tensor):
        self.net, self.cfg, self.ctrl_mean, self.ctrl_std = net, cfg, ctrl_mean, ctrl_std

    def _obs(self, cost):
        return obs_features(cost, self.ctrl_mean, self.ctrl_std) if self.cfg.obs_input else None

    def delta(self, x, g, m, obs):
        g_std = standardize_grad(g, self.net.d) if self.cfg.use_grad else None
        out = self.net((x - self.ctrl_mean) / self.ctrl_std, g_std, m, obs)
        return self.cfg.step_scale * self.ctrl_std * out

    def unroll(self, cost: FourDVarCost, x0: torch.Tensor, train: bool = False, K: Optional[int] = None,
               diag: bool = False) -> dict:
        """K steps from x0. Returns x_K (graph kept if train), iterates, J of the gradient calls, n_grad, steps."""
        K = self.cfg.K if K is None else K
        obs = self._obs(cost)
        x, xs, js, gs, n_grad = x0, [x0.detach()], [], [], 0
        xg = []                                           # graph-carrying iterates (train, loss="sum")
        lbfgs = None
        if getattr(self.cfg, "base", "none") == "lbfgs":
            assert self.cfg.use_grad and not self.cfg.second_order, "base lbfgs: gradient input, first order"
            lbfgs = LBFGSDirection(history=10)            # memory of the learner's own moves (data)
        if lbfgs is not None and getattr(self.cfg, "safeguard", "none") == "backtrack":
            return self._unroll_backtrack(cost, x0, train, K, diag, obs, lbfgs)
        for m in range(K):
            g = None
            if self.cfg.use_grad:
                if train and self.cfg.second_order:
                    xr = x if x.requires_grad else x.detach().requires_grad_(True)
                    j = cost(*cost.unpack(xr))
                    (g,) = torch.autograd.grad(j.sum(), xr, create_graph=True)
                    x = xr
                else:
                    j, g = cost.grad(x.detach())          # first order: g is data
                n_grad += 1
                js.append(j.detach())
                if diag:
                    gs.append(g.detach())
            elif diag:
                gs.append(cost.grad(x.detach())[1])      # diagnostics only (not counted)
            if lbfgs is not None:
                x = x - (lbfgs(x.detach(), g.detach()) + self.delta(x, g, m, obs))
            else:
                x = x - self.delta(x, g, m, obs)
            xs.append(x.detach())
            if train:
                xg.append(x)
        return {"x": x, "xs": xs, "J_grad": js, "n_grad": n_grad, "g": gs, "x_graph": xg}

    def _unroll_backtrack(self, cost, x0, train, K, diag, obs, lbfgs) -> dict:
        """Base lbfgs + backtrack safeguard (first order: J, g at the trial points are data)."""
        x, xs, js, gs, xg = x0, [x0.detach()], [], [], []
        x_acc = j_acc = g_acc = None
        for m in range(K):
            j, g = cost.grad(x.detach())
            js.append(j.detach())
            if diag:
                gs.append(g.detach())
            if m == 0:                                    # first guess: accepted by definition
                acc = torch.ones_like(j, dtype=torch.bool)
                x_acc, j_acc, g_acc = x, j, g
            else:
                acc = torch.isfinite(j) & torch.isfinite(g).all(1) & (j <= j_acc)
                x_acc = torch.where(acc[:, None], x, x_acc)
                j_acc = torch.where(acc, j, j_acc)
                g_acc = torch.where(acc[:, None], torch.nan_to_num(g), g_acc)
                xs.append(x_acc.detach())
                if train:
                    xg.append(x_acc)
            # accepted cases: new move from the accepted point; rejected cases: halve the rejected move
            step = lbfgs(x_acc.detach(), g_acc.detach()) + self.delta(x_acc, g_acc, m, obs)
            x = torch.where(acc[:, None], x_acc - step, x_acc + (x - x_acc) / 2)
        with torch.no_grad():                             # last trial point: reporting evaluation, not counted
            j_last = cost(*cost.unpack(x.detach()))
        keep = torch.isfinite(j_last) & (j_last <= j_acc)
        x_fin = torch.where(keep[:, None], x, x_acc)
        xs.append(x_fin.detach())
        if train:
            xg.append(x_fin)
        return {"x": x_fin, "xs": xs, "J_grad": js, "n_grad": K, "g": gs, "x_graph": xg}

    # ---- training --------------------------------------------------------------------------------------------
    def loss(self, cost: FourDVarCost, x0: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            j0 = cost(*cost.unpack(x0))
        r = self.unroll(cost, x0, train=True)
        if getattr(self.cfg, "loss", "final") == "sum":
            jm = torch.stack([cost(*cost.unpack(xm)) for xm in r["x_graph"]], 1)       # [n, K]
            return (jm / j0.clamp_min(TINY)[:, None]).mean()
        return (cost(*cost.unpack(r["x"])) / j0.clamp_min(TINY)).mean()

    def fit(self, make_cost: Callable, x0_train: torch.Tensor, val_cost: FourDVarCost, x0_val: torch.Tensor,
            log=None) -> list:
        """Mini-batch Adam over train ICs; keep the epoch with the lowest val loss (mean J(x^K)/J(x^0))."""
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        best, best_state, hist = float("inf"), None, []
        n = x0_train.shape[0]
        for ep in range(cfg.epochs):
            t0, tot, n_skip = time.perf_counter(), 0.0, 0
            self.net.train()
            perm = rng.permutation(n)
            for b0 in range(0, n, cfg.batch):
                idx = np.sort(perm[b0:b0 + cfg.batch])
                opt.zero_grad()
                loss = self.loss(make_cost(idx), x0_train[idx])
                if not torch.isfinite(loss):                  # NaN guard (2026-11-20): skip a non-finite batch
                    n_skip += 1
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.grad_clip)
                if all(torch.isfinite(p.grad).all() for p in self.net.parameters() if p.grad is not None):
                    opt.step()
                else:
                    n_skip += 1
                tot += float(loss.detach()) * len(idx)
            self.net.eval()
            vl = float(self.loss(val_cost, x0_val).detach())   # grad enabled: the unroll needs grad J
            h = {"epoch": ep, "train_ratio": tot / n, "val_ratio": vl, "skipped_batches": n_skip,
                 "s": time.perf_counter() - t0}
            hist.append(h)
            if log is not None:
                log(h)
            if np.isfinite(vl) and vl < best:
                best, best_state = vl, copy.deepcopy(self.net.state_dict())
        if best_state is None:                            # no finite val epoch: raise instead of a silent bad learner
            raise RuntimeError("unroll fit: validation loss never finite")
        self.net.load_state_dict(best_state)
        self.net.eval()
        return hist

    # ---- checkpoint ------------------------------------------------------------------------------------------
    def state_dict(self) -> dict:
        n = self.net
        return {"net": n.state_dict(), "cfg": asdict(self.cfg), "ctrl_mean": self.ctrl_mean, "ctrl_std": self.ctrl_std,
                "net_kwargs": dict(d=n.d, use_grad=n.use_grad, d_obs=n.d_obs, hidden=n.head.in_features,
                                   n_layers=sum(isinstance(l, torch.nn.Linear) for l in n.body),
                                   emb_dim=n.emb.net[2].out_features, output_scale=n.output_scale,
                                   log_k=n.emb.log_input)}

    @classmethod
    def from_state_dict(cls, sd: dict) -> "LearnedOptimization":
        net = FreeUpdateNet(**sd["net_kwargs"], zero_init=False).to(sd["ctrl_mean"].dtype)
        net.load_state_dict(sd["net"])
        net.eval()
        return cls(net, UnrollConfig(**sd["cfg"]), sd["ctrl_mean"], sd["ctrl_std"])


def build_unroll(d: int, cfg: UnrollConfig, ctrl_mean, ctrl_std, n_obs: int = 11, **net_kwargs) -> LearnedOptimization:
    torch.manual_seed(cfg.seed)
    d_obs = 2 * n_obs * 3 if cfg.obs_input else 0
    net = FreeUpdateNet(d, use_grad=cfg.use_grad, d_obs=d_obs, **net_kwargs).to(ctrl_mean.dtype)
    return LearnedOptimization(net, cfg, ctrl_mean, ctrl_std)


@torch.no_grad()
def run_meta_a(cost: FourDVarCost, x0_init: torch.Tensor, learner: LearnedOptimization,
               eta_init: Optional[torch.Tensor] = None, K: Optional[int] = None) -> dict:
    """Frozen K-step inference; record compatible with run_fourdvar (kstep length = K)."""
    n = x0_init.shape[0]
    if cost.weak and eta_init is None:
        eta_init = torch.zeros(n, cost.cfg.n_obs - 1, 3, dtype=x0_init.dtype)
    x0 = cost.pack(x0_init, eta_init).detach().clone()
    t0 = time.perf_counter()
    with torch.enable_grad():
        r = learner.unroll(cost, x0, train=False, K=K, diag=True)
    wall = time.perf_counter() - t0
    xs = torch.stack(r["xs"], 1)                                     # [n, K + 1, d]
    j = torch.stack([cost(*cost.unpack(xs[:, m])) for m in range(xs.shape[1])], 1)   # reporting only
    steps = xs[:, :-1] - xs[:, 1:]
    g = torch.stack(r["g"], 1)
    cos = torch.nn.functional.cosine_similarity(steps, g, dim=2)
    xk = xs[:, -1]
    out = {"x0_iter": xs[:, :, :3].numpy(), "J_iter": j.numpy(), "cos_step_grad": cos.numpy(),
           "n_grad_evals": np.full(n, r["n_grad"], dtype=np.int64), "n_fwd_evals": np.zeros(n, dtype=np.int64),
           "converged": torch.isfinite(j[:, -1]).numpy(), "x0_a": xk[:, :3].numpy(), "wall_s": wall,
           # gradient evaluations spent to reach iterate m (one per move; J at iterates is reporting only)
           "grad_cum": np.tile(np.arange(xs.shape[1]) * (1 if learner.cfg.use_grad else 0), (n, 1))}
    if cost.weak:
        out["eta_final"] = cost.unpack(xk)[1].numpy()
    return out
