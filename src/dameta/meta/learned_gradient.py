"""
M5 cell 8 — ADJOINT-FREE LEARNED GRADIENT (pilot 2026-11-14): IC estimation with a model that has no adjoint.

Only the observation term needs an adjoint: grad J_b = B^-1 (x0 - x0_b) and grad J_q = eta / sigma_q^2 are analytic.
One FORWARD run of the false model F at the iterate x^k gives J, the states at the observation times x_{t_i} and the
residuals r_i = H_i^T R^-1 (H x_{t_i} - y_i). The learner predicts grad J_o from that forward information:

  arch "mlp"      g_o^ = g_scale * G_theta(x_std^k, emb(k), [d^k, mask])           (first version; overfits)
  arch "adjoint"  learned local tangent-linear operators A_theta(x_{t_i}) ~ d x_{t_{i+1}} / d x_{t_i} (3 x 3, one obs
                  interval, A = I + MLP(state)), combined by the EXACT adjoint recursion
                      lambda_L = r_L,   lambda_i = r_i + A_theta(x_{t_i})^T lambda_{i+1},
                      grad_x0 J_o = lambda_0,   grad_eta_i J_o = lambda_i (weak constraint)
                  -> linear in the innovations like a true adjoint; the network only learns state -> Jacobian, which
                  is a property of the dynamics (not of the case). No iteration input: a local tangent-linear
                  operator does not depend on the optimiser iteration (deviation from the x / grad / k input rule).
  g^ = grad J_b(x^k) [+ grad J_q] + g_o^   is fed to the GD / Adagrad teacher (ReferenceStep).

Supervised target: grad J_o of the EMULATOR (autodiff) at the same control; for the adjoint arch the default loss
matches each operator to the emulator's one-interval Jacobian at the same forward-run state (loss "jacobian": local
and well-posed — training through the 11-fold product of chaotic Jacobians, loss "gradient", is ill-conditioned);
the best epoch is chosen on the median relative validation GRADIENT error (both archs). Training controls:
  "trajectory" — states visited by the adjoint-free hybrid classical run (J: F, gradient: emulator)  (mlp version)
  "sampled"    — decorrelated controls x0 = x0_b + s L z (L L^T = B, s in perturb_scales), eta ~ N(0, (s sigma_q)^2),
                 n per case, plus a thinned hybrid trajectory (stride 10); no truth and no adjoint of F needed.
Inference cost: one forward run of F per iteration (n_fwd_evals); no gradient evaluation. Diagnostics (not counted):
cos(g^, exact grad of F's cost).
"""

import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn

from ..da.cost import FourDVarCost, window_trajectory
from ..da.fourdvar import run_fourdvar
from ..da.gradients import HybridGradient
from ..da.reference_step import ReferenceStep
from .networks import GradNet


@dataclass
class LGConfig:
    arch: str = "mlp"                 # mlp | adjoint
    data: str = "trajectory"          # trajectory | sampled
    samples_per_case: int = 64        # sampled data: controls per training case
    perturb_scales: list = field(default_factory=lambda: [0.5, 1.0, 2.0])
    traj_stride: int = 10             # sampled data: also keep every n-th hybrid-trajectory state (0 = none)
    loss: str = "jacobian"            # adjoint arch: jacobian (match the emulator's one-interval Jacobians at the
                                      # forward-run states; local, well-posed) | gradient (through the recursion)
    hidden: int = 128
    n_layers: int = 3
    emb_dim: int = 8
    log_k: bool = True
    lr: float = 1e-3
    epochs: int = 50
    batch_size: int = 1024
    seed: int = 0


# ---- forward information (one forward run) ------------------------------------------------------------------------
@torch.no_grad()
def forward_run(cost: FourDVarCost, c: torch.Tensor) -> dict:
    """ONE forward run: J [n], innovation features [n, 2 n_obs 3], states at obs times [n, n_obs, 3], residuals."""
    x0, *rest = cost.unpack(c)
    eta = rest[0] if cost.weak else None
    traj = window_trajectory(cost.step, x0, cost.cfg, eta)
    x_obs = traj[:, cost.t_obs]
    innov = (x_obs - cost.y) * cost.mask
    d = innov / cost.cfg.sigma_o2 ** 0.5
    jo = 0.5 * (d ** 2).sum((-1, -2))
    jq = 0.5 * (eta ** 2).sum((-1, -2)) / cost.cfg.sigma_q ** 2 if cost.weak else torch.zeros_like(jo)
    m = cost.mask.expand(c.shape[0], -1, -1)
    return {"J": cost.jb(x0) + jo + jq, "obs": torch.cat([d.flatten(1), m.flatten(1)], 1), "x_obs": x_obs,
            "r": innov / cost.cfg.sigma_o2}


def forward_features(cost: FourDVarCost, c: torch.Tensor):
    """Backward-compatible: (J, innovation features)."""
    f = forward_run(cost, c)
    return f["J"], f["obs"]


def analytic_grad(cost: FourDVarCost, c: torch.Tensor) -> torch.Tensor:
    """grad of J_b (+ J_q) w.r.t. the packed control — no model involved."""
    x0, *rest = cost.unpack(c.detach())
    dx = x0 - cost.x0_b
    gb = torch.einsum("nij,nj->ni", cost.b_inv, dx) if cost.b_inv is not None else dx / cost.cfg.sigma_b ** 2
    if not cost.weak:
        return gb
    return torch.cat([gb, (rest[0] / cost.cfg.sigma_q ** 2).flatten(1)], 1)


def jo_grad(cost: FourDVarCost, c: torch.Tensor) -> torch.Tensor:
    """grad J_o by autodiff of `cost` (training targets with the EMULATOR cost; exact references in tests)."""
    cc = c.detach().requires_grad_(True)
    (g,) = torch.autograd.grad(cost.terms(*cost.unpack(cc))["Jo"].sum(), cc)
    return g.detach()


def adjoint_recursion(a: torch.Tensor, r: torch.Tensor, weak: bool) -> torch.Tensor:
    """a [n, n_obs - 1, 3, 3] (a[:, i] ~ d x_{t_{i+1}} / d x_{t_i}), r [n, n_obs, 3] -> grad J_o [n, d]."""
    lam = r[:, -1]
    lams = [lam]
    for i in range(r.shape[1] - 2, -1, -1):
        lam = r[:, i] + torch.einsum("nji,nj->ni", a[:, i], lam)
        lams.append(lam)
    lams = lams[::-1]                                              # lams[i] at obs time t_i
    if not weak:
        return lams[0]
    return torch.cat([lams[0]] + lams[1:], 1)                      # [x0 | eta_1 .. eta_{n_obs - 1}]


class AdjointNet(nn.Module):
    """A_theta(state) = I + MLP(state_std): learned 3 x 3 tangent-linear operator over one observation interval."""

    def __init__(self, hidden: int = 128, n_layers: int = 3):
        super().__init__()
        layers, width = [], 3
        for _ in range(n_layers):
            layers += [nn.Linear(width, hidden), nn.SiLU()]
            width = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, 9)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, s_std: torch.Tensor) -> torch.Tensor:
        return torch.eye(3, dtype=s_std.dtype) + self.head(self.body(s_std)).unflatten(-1, (3, 3))


# ---- training data --------------------------------------------------------------------------------------------------
def _case_cost(cost: FourDVarCost, idx) -> FourDVarCost:
    return FourDVarCost(cost.step, cost.y[idx], cost.mask, cost.x0_b[idx], cost.cfg, weak=cost.weak,
                        b_inv=None if cost.b_inv is None else cost.b_inv[idx])


def interval_jacobians(step, states: torch.Tensor, n_steps: int) -> torch.Tensor:
    """d M^(n_steps)(s) / d s for states [N, m, 3] -> [N, m, 3, 3] (autodiff of `step`; the EMULATOR in training)."""
    def interval(x):
        for _ in range(n_steps):
            x = step(x)
        return x
    jac = torch.func.vmap(torch.func.jacrev(interval))
    flat = states.reshape(-1, 3)
    return torch.cat([jac(flat[b:b + 8192]) for b in range(0, len(flat), 8192)]).reshape(*states.shape, 3)


def _pairs_at(cost_f, cost_emu, x, k, idx, chunk=4096, with_jac=False) -> dict:
    out = {"x": x, "k": k, "obs": [], "x_obs": [], "r": [], "g": [], "jac": []}
    for b0 in range(0, len(k), chunk):
        i = idx[b0:b0 + chunk]
        f = forward_run(_case_cost(cost_f, i), x[b0:b0 + chunk])
        for key in ("obs", "x_obs", "r"):
            out[key].append(f[key])
        out["g"].append(jo_grad(_case_cost(cost_emu, i), x[b0:b0 + chunk]))
        if with_jac:
            out["jac"].append(interval_jacobians(cost_emu.step, f["x_obs"][:, :-1], cost_f.cfg.obs_every))
    for key in ("obs", "x_obs", "r", "g") + (("jac",) if with_jac else ()):
        out[key] = torch.cat(out[key])
    if not with_jac:
        del out["jac"]
    return out


def _hybrid_states(cost_f, cost_emu, x0_b, method, lr, max_iter, tol, stride=1):
    xs, ks, case = [], [], []

    def hook(k, c, g, active):
        if k % stride:
            return
        a = active.bool()
        xs.append(c[a].clone())
        ks.append(torch.full((int(a.sum()),), k, dtype=c.dtype))
        case.append(torch.nonzero(a).flatten())

    run_fourdvar(HybridGradient(cost_f, cost_emu), x0_b, method, lr, max_iter, tol, on_step=hook)
    return torch.cat(xs), torch.cat(ks), torch.cat(case)


def collect_pairs(cost_f: FourDVarCost, cost_emu: FourDVarCost, x0_b: torch.Tensor, method: str, lr: float,
                  max_iter: int = 1000, tol: float = 1e-6, chunk: int = 4096) -> dict:
    """'trajectory' data: every state of the adjoint-free hybrid classical run."""
    x, k, idx = _hybrid_states(cost_f, cost_emu, x0_b, method, lr, max_iter, tol)
    return _pairs_at(cost_f, cost_emu, x, k, idx, chunk)


def collect_sampled_pairs(cost_f: FourDVarCost, cost_emu: FourDVarCost, x0_b: torch.Tensor, method: str, lr: float,
                          cfg: LGConfig, max_iter: int = 1000, tol: float = 1e-6, chunk: int = 4096) -> dict:
    """'sampled' data: decorrelated controls around the first guess (B-shaped perturbations) + thinned trajectory."""
    gen = torch.Generator().manual_seed(cfg.seed + 1234)
    n = x0_b.shape[0]
    b = torch.linalg.inv(cost_f.b_inv) if cost_f.b_inv is not None else torch.eye(3).expand(n, 3, 3) * cost_f.cfg.sigma_b ** 2
    chol = torch.linalg.cholesky(b)
    m = cfg.samples_per_case
    idx = torch.arange(n).repeat_interleave(m)
    scales = torch.tensor(cfg.perturb_scales, dtype=x0_b.dtype)
    s = scales[torch.randint(len(scales), (n * m,), generator=gen)]
    z = torch.randn(n * m, 3, generator=gen, dtype=x0_b.dtype)
    x0 = x0_b[idx] + s[:, None] * torch.einsum("nij,nj->ni", chol[idx], z)
    if cost_f.weak:
        eta = s[:, None, None] * cost_f.cfg.sigma_q * torch.randn(n * m, cost_f.cfg.n_obs - 1, 3, generator=gen,
                                                                  dtype=x0_b.dtype)
        x = torch.cat([x0, eta.flatten(1)], 1)
    else:
        x = x0
    k = torch.zeros(n * m, dtype=x0_b.dtype)
    if cfg.traj_stride > 0:
        xt, kt, it = _hybrid_states(cost_f, cost_emu, x0_b, method, lr, max_iter, tol, cfg.traj_stride)
        x, k, idx = torch.cat([x, xt]), torch.cat([k, kt]), torch.cat([idx, it])
    return _pairs_at(cost_f, cost_emu, x, k, idx, chunk, with_jac=(cfg.arch == "adjoint" and cfg.loss == "jacobian"))


# ---- learner --------------------------------------------------------------------------------------------------------
class LearnedGradient:
    def __init__(self, net: nn.Module, cfg: LGConfig, ctrl_mean, ctrl_std, g_scale):
        self.net, self.cfg, self.ctrl_mean, self.ctrl_std, self.g_scale = net, cfg, ctrl_mean, ctrl_std, g_scale

    def _pred(self, x, k, fwd, weak):
        """Predicted grad J_o in PHYSICAL units."""
        if self.cfg.arch == "adjoint":
            st = (fwd["x_obs"][:, :-1] - self.ctrl_mean[:3]) / self.ctrl_std[:3]       # x_{t_0 .. t_{L-1}}
            return adjoint_recursion(self.net(st), fwd["r"], weak)
        return self.net((x - self.ctrl_mean) / self.ctrl_std, k, fwd["obs"]) * self.g_scale

    @torch.no_grad()
    def grad_obs(self, x, k, fwd, weak=False):
        self.net.eval()
        return self._pred(x, k, fwd, weak)

    def fit(self, train: dict, val: dict, weak: bool = False, log=None) -> list:
        cfg, gen = self.cfg, torch.Generator().manual_seed(self.cfg.seed)
        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        sub = lambda p, i: {"obs": p["obs"][i], "x_obs": p["x_obs"][i], "r": p["r"][i]}  # noqa: E731
        if "jac" in train:
            self.jac_scale = train["jac"].std((0, 1)).clamp_min(1e-12)        # per-entry Jacobian scale [3, 3]
        n, best, best_state, hist = train["g"].shape[0], float("inf"), None, []
        for ep in range(cfg.epochs):
            t0, tot = time.perf_counter(), 0.0
            self.net.train()
            perm = torch.randperm(n, generator=gen)
            for b0 in range(0, n, cfg.batch_size):
                i = perm[b0:b0 + cfg.batch_size]
                opt.zero_grad()
                if cfg.arch == "adjoint" and cfg.loss == "jacobian":
                    st = (train["x_obs"][i][:, :-1] - self.ctrl_mean[:3]) / self.ctrl_std[:3]
                    loss = ((self.net(st) - train["jac"][i]) / self.jac_scale).pow(2).mean()
                else:
                    pred = self._pred(train["x"][i], train["k"][i], sub(train, i), weak)
                    loss = ((pred - train["g"][i]) / self.g_scale).pow(2).mean()
                loss.backward()
                opt.step()
                tot += float(loss.detach()) * len(i)
            self.net.eval()
            with torch.no_grad():
                pv = self._pred(val["x"], val["k"], val, weak)
                vmse = float(((pv - val["g"]) / self.g_scale).pow(2).mean())
                vcos = float(torch.nn.functional.cosine_similarity(pv, val["g"], dim=1).median())
                vrel = float(((pv - val["g"]).norm(dim=1) / val["g"].norm(dim=1).clamp_min(1e-30)).median())
            # selection on the MEDIAN relative gradient error (the MSE is dominated by a few far-from-minimum cases)
            vl = vrel
            h = {"epoch": ep, "train_loss": tot / n, "val_mse": vmse, "val_relerr_med": vrel, "val_cos_med": vcos,
                 "s": time.perf_counter() - t0}
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
        kw = (dict(hidden=self.cfg.hidden, n_layers=self.cfg.n_layers) if self.cfg.arch == "adjoint" else
              dict(d=n.d, hidden=self.cfg.hidden, n_layers=self.cfg.n_layers, emb_dim=self.cfg.emb_dim,
                   log_k=self.cfg.log_k, d_obs=n.d_obs))
        return {"net": n.state_dict(), "cfg": asdict(self.cfg), "ctrl_mean": self.ctrl_mean, "ctrl_std": self.ctrl_std,
                "g_scale": self.g_scale, "net_kwargs": kw}

    @classmethod
    def from_state_dict(cls, sd: dict) -> "LearnedGradient":
        cfg = LGConfig(**sd["cfg"])
        net = (AdjointNet(**sd["net_kwargs"]) if cfg.arch == "adjoint" else GradNet(**sd["net_kwargs"]))
        net = net.to(sd["ctrl_mean"].dtype)
        net.load_state_dict(sd["net"])
        net.eval()
        return cls(net, cfg, sd["ctrl_mean"], sd["ctrl_std"], sd["g_scale"])


def build_lg(d: int, n_obs: int, cfg: LGConfig, ctrl_mean, ctrl_std, train_pairs: dict) -> LearnedGradient:
    torch.manual_seed(cfg.seed)
    if cfg.arch == "adjoint":
        net = AdjointNet(cfg.hidden, cfg.n_layers)
    else:
        net = GradNet(d, cfg.hidden, cfg.n_layers, cfg.emb_dim, cfg.log_k, d_obs=2 * n_obs * 3)
    return LearnedGradient(net.to(ctrl_mean.dtype), cfg, ctrl_mean, ctrl_std,
                           train_pairs["g"].std(0).clamp_min(1e-12))


def run_learned_gradient(cost: FourDVarCost, x0_init: torch.Tensor, learner, method: str, lr: float,
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
        fwd = forward_run(cost, c)
        j = fwd["J"]
        n_fwd += active.long()
        j_hist.append(j.clone() if j_prev is None else torch.where(active.bool(), j, j_hist[-1]))
        if j_prev is not None:
            rel = (j_prev - j).abs() / j_prev.abs().clamp_min(1e-300)
            active = active * (~(active.bool() & ((rel < tol) | ~torch.isfinite(j)))).to(c.dtype)
        if active.sum() == 0:
            break
        g_hat = analytic_grad(cost, c) + learner.grad_obs(c, k, fwd, cost.weak)
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
