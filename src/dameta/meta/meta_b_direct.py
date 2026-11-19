"""
delta_exact = SUPERVISED delta learned optimizer (2026-11-18, pilot; variant `delta_exact`): the network imitates one
step of the classical teacher, loss = mismatch between the teacher update Delta = x^{k+1} - x^k and the prediction
Delta_hat. Inference: the learner alone (no teacher). Unsupervised counterpart: variant `delta` (meta_b.py).

    input:              x_std = (x^k - mean) / std,  g_std = g^k / RMS(g^k) (per case and per block x0 | eta),  emb(k)
                        [obs_input: + observations of the case: y standardised by the train-truth climatology and
                        masked, the mask, and the observation times t_i / L  (3 n_obs + 3 n_obs + n_obs values)]
    output:             Delta_theta = - RMS_block(g^k) * u_theta,   per block (x0 | eta):
                            u_b = exp(l0_b + l_theta,b) * v_b / RMS(v_b),   v = g_std + MLP_dir
                        gradient units (g_std has no magnitude, the step scale comes from |g^k| per block); direction
                        and log-magnitude are separate heads because the teacher step in these units spans ~4 decades
                        (strong-Wolfe L-BFGS, dense weak: |u| ~ 4e-5 at k = 0 -> 1e-2 at k = 40). l0_b = median
                        log RMS(u_t,b) of the train pairs; zero-initialised heads -> start = preconditioned-free GD
                        with the median teacher step. A GD teacher is exactly representable (v = g_std, l = log lr).
    target:             Delta_t = x^{k+1} - x^k of the teacher run (same lr / stopping rule as classical)
    loss:               mean_pairs |Delta_theta - Delta_t|^2 / |Delta_t|^2          (relative, scale-free per case)

Training pairs come from ONE teacher run per train case (teacher forcing: states on the teacher trajectory; collected
once, then epochs over the pairs, best epoch = min MEDIAN relative error on val-IC pairs). Inference: Delta_theta alone — no teacher, no line
search, ONE gradient evaluation per iteration (the classical stopping rule). With the strong-Wolfe L-BFGS teacher this
removes the line-search gradients that cell 1 paid on top of the outer gradient.

Conditioning options (2026-11-18, pilot: "is the learner badly conditioned / under-trained?"):
    coords      raw  — as above (x0 standardised by climatology, eta by sigma_q; ocean-code scaling)
                cvt  — control-variable transform of 4D-Var: v0 = L^-1 (x0 - x0_b) with B = L L^T (case-specific
                       hybrid B), w = eta / sigma_q. The learner works entirely in (v0, w): input v, gradient
                       g_v = [L^T g_x0, sigma_q g_eta], output Delta_v (same law, -RMS_block(g_v) u), mapped back
                       Delta_x0 = L Delta_v0, Delta_eta = sigma_q Delta_w; loss = relative MSE in v coordinates.
                       J_b + J_q = |v|^2 / 2 -> Hessian I + (obs part): every eigenvalue >= 1.
    grad_input  unit — g / RMS(g) per case and block (magnitude dropped)
                fixed — g / (train-median RMS per block): a FIXED scale, the gradient magnitude stays visible
    obs_feat    y    — case observations [y_std masked, mask, t/L] (obs_time_features)
                innov — innovations at the current iterate [(H M(x^k) - y) / sigma_o masked, mask, t/L], same
                       layout (from the forward run of the gradient evaluation: no extra model run at inference)
    optimizer   adam (mini-batch) | lbfgs (full-batch torch L-BFGS with strong-Wolfe line search, lbfgs_iters
                inner iterations per epoch); max_train_pairs > 0 = random subset of the train pairs
"""

import copy
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch

from ..da.cost import FourDVarCost, window_trajectory
from ..da.fourdvar import _pad_cum
from .da_utils_obs import obs_time_features
from .meta_a import standardize_grad
from .networks import IterationEmbedding

TINY = torch.finfo(torch.float64).tiny


@dataclass
class DirectConfig:
    hidden: int = 64
    n_layers: int = 2
    emb_dim: int = 8
    log_k: bool = True
    obs_input: bool = False         # append the case observations (values, mask, times) to the input
    arch: str = "mlp"               # mlp (flat DirectNet) | temporal (one token per obs time + 1D conv; needs obs_input)
    kernel: int = 5                 # temporal: Conv1d kernel along the observation times
    max_log_dev: float = 10.0       # |l_theta| bound (tanh) around l0
    lr: float = 1e-3
    epochs: int = 50
    batch_size: int = 1024
    grad_clip_norm: float = 1.0
    seed: int = 0
    coords: str = "raw"             # raw | cvt (control-variable transform v0 = L^-1 (x0 - x0_b), w = eta / sigma_q)
    grad_input: str = "unit"        # unit (g / RMS per case) | fixed (g / train-median RMS per block, keeps |g|)
    obs_feat: str = "y"             # y (case observations) | innov (innovations at the current iterate)
    optimizer: str = "adam"         # adam (mini-batch) | lbfgs (full batch, strong Wolfe)
    lbfgs_iters: int = 20           # lbfgs: inner iterations per epoch
    lbfgs_history: int = 50
    max_train_pairs: int = 0        # > 0: random subset of the train pairs (seeded)
    hist_m: int = 0                 # > 0: optimiser history input = the m most recent (s, y) pairs (L-BFGS memory)
                                    # (HistTemporalNet; replaces the observation input; raw coordinates)
    base: str = "grad"              # hist: base descent direction of the output law: grad (-g) | lbfgs (two-loop
                                    # -H g of the same memory: zero-initialised learner = fixed-step L-BFGS)
    horizon: int = 1                # target = teacher displacement over h steps, x^{k+h} - x^k = Delta_k + .. +
                                    # Delta_{k+h-1} (delayed: known once the teacher made h steps; capped at its end)
    train_mode: str = "offline"     # offline (shuffled mini-batches of pairs) | online (teacher-run order: one Adam
                                    # step per teacher iteration on a batch of cases, delayed by the horizon)
    online_cases: int = 100         # online: cases per episode
    train_loss: str = "mean"      # mean relative error | logmean: mean log(1 + rel) (balances rare large errors)
    select: str = "median"          # best-epoch statistic of the val relative errors: median | logmean | mean
    first_step: str = "learner"     # learner | teacher: iteration 0 = one classical step (strong-Wolfe L-BFGS
                                    # iteration from the initial gradient; its gradients counted), learner from k = 1
    K: int = 1                     # inner rollout: Delta_hat = sum of K sub-steps of 1/K (state input re-evaluated,
                                    # gradient held fixed); 1 = delta_exact, unsupervised delta uses the delta.K chain


def grad_scale(g: torch.Tensor, d: int) -> torch.Tensor:
    """[n, d]: RMS of g per case and per control block (x0 | eta), broadcast to the block's components."""
    out = torch.empty_like(g)
    for a, b in blocks(d):
        out[:, a:b] = g[:, a:b].pow(2).mean(1, keepdim=True).sqrt().clamp_min(TINY)
    return out


def blocks(d: int) -> list:
    return [(0, 3)] + ([(3, d)] if d > 3 else [])


PAIR_KEYS = ("x", "g", "k", "case", "target", "obs_pair", "S", "Y")
LBFGS_EPS = 1e-10                   # pair kept iff s.y > eps |s| |y| (as reference_step.LBFGSDirection)


def lbfgs_pair(s: torch.Tensor, y: torch.Tensor) -> tuple:
    """Curvature pair as stored by LBFGSDirection: zeroed when s.y <= eps |s| |y| or s = 0."""
    sy = (s * y).sum(-1)
    ok = (sy > LBFGS_EPS * s.norm(dim=-1) * y.norm(dim=-1)) & (s.norm(dim=-1) > 0)
    return s * ok[..., None], y * ok[..., None]


def lbfgs_two_loop(g: torch.Tensor, S: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """H g by the L-BFGS two-loop recursion over the memory S, Y [n, m, d] (oldest -> newest, zero rows = no pair);
    identical to reference_step.LBFGSDirection (gamma from the newest valid pair, fallback 1 / |g|)."""
    q, alphas, m = g.clone(), [], S.shape[1]
    for i in reversed(range(m)):
        s, y = S[:, i], Y[:, i]
        sy = (s * y).sum(1)
        rho = torch.where(sy > 0, 1.0 / sy.clamp_min(1e-300), torch.zeros_like(sy))
        a = rho * (s * q).sum(1)
        q = q - a[:, None] * y
        alphas.append((a, rho))
    gamma = 1.0 / g.norm(dim=1).clamp_min(1e-300)
    for i in range(m):
        sy, yy = (S[:, i] * Y[:, i]).sum(1), (Y[:, i] * Y[:, i]).sum(1)
        gamma = torch.where(sy > 0, sy / yy.clamp_min(1e-300), gamma)
    r = gamma[:, None] * q
    for i, (a, rho) in zip(range(m), reversed(alphas)):
        b = rho * (Y[:, i] * r).sum(1)
        r = r + (a - b)[:, None] * S[:, i]
    return r


def attach_history(p: dict, m: int) -> dict:
    """Add the L-BFGS memory of every teacher pair: S, Y [P, m, d] = the m most recent moves s = x^{i+1} - x^i and
    gradient differences y = g^{i+1} - g^i before iteration k of the same case (oldest -> newest, zero-padded,
    invalid pairs zeroed as LBFGSDirection). Pairs of a case must cover k = 0..K-1 contiguously (teacher run)."""
    P, d = p["x"].shape
    S, Y = torch.zeros(P, m, d, dtype=p["x"].dtype), torch.zeros(P, m, d, dtype=p["x"].dtype)
    for c in p["case"].unique():
        idx = torch.nonzero(p["case"] == c).flatten()
        idx = idx[p["k"][idx].argsort()]
        assert bool((p["k"][idx] == torch.arange(len(idx), dtype=p["k"].dtype)).all()), "non-contiguous teacher pairs"
        X, G = p["x"][idx], p["g"][idx]
        s, y = lbfgs_pair(X[1:] - X[:-1], G[1:] - G[:-1])                     # moves 0 .. K-2
        z = torch.zeros(m, d, dtype=X.dtype)
        ms, my = torch.cat([z, s]), torch.cat([z, y])                          # padded: row k + m - 1 = move k - 1
        for kk in range(len(idx)):
            S[idx[kk]], Y[idx[kk]] = ms[kk:kk + m], my[kk:kk + m]
    return {**p, "S": S, "Y": Y}


def online_groups(p: dict, n_cases: int, gen: torch.Generator) -> list:
    """Online training order (2026-11-19, pilot): episodes = random batches of n_cases train cases; inside an episode
    the teacher run is replayed in iteration order: the k-th group = the pairs of those cases at iteration k, used
    once the teacher has made the steps of their target (delayed evaluation). Teacher forcing: the cached states and
    gradients are exactly those of a live teacher run, so replaying the cache is identical to running it online."""
    cases = p["case"].unique()
    out = []
    for b in cases[torch.randperm(len(cases), generator=gen)].split(max(1, n_cases)):
        inb = torch.isin(p["case"], b)
        for kk in p["k"][inb].unique().sort().values:
            out.append(torch.nonzero(inb & (p["k"] == kk)).flatten())
    return out


def attach_horizon(p: dict, h: int) -> dict:
    """Replace each target by the teacher displacement over h steps, x^{k+h} - x^k = sum of the next h teacher
    updates of the same case (capped at the teacher's last iterate). Pairs of a case cover k = 0..K-1."""
    if h <= 1:
        return p
    t = p["target"].clone()
    for c in p["case"].unique():
        idx = torch.nonzero(p["case"] == c).flatten()
        idx = idx[p["k"][idx].argsort()]
        cs = torch.cat([torch.zeros_like(t[idx[:1]]), p["target"][idx].cumsum(0)])   # x^k - x^0, k = 0..K
        K = len(idx)
        for kk in range(K):
            t[idx[kk]] = cs[min(kk + h, K)] - cs[kk]
    return {**p, "target": t}


def cvt_factor(cost: FourDVarCost) -> torch.Tensor:
    """[n, 3, 3] lower Cholesky factor L of B (B = L L^T) per case; B = sigma_b^2 I without b_inv."""
    n = cost.x0_b.shape[0]
    if cost.b_inv is None:
        return cost.cfg.sigma_b * torch.eye(3, dtype=cost.x0_b.dtype).expand(n, 3, 3).clone()
    return torch.linalg.cholesky(torch.linalg.inv(cost.b_inv))


def grad_innov(cost: FourDVarCost, control: torch.Tensor) -> tuple:
    """(J, dJ/dc, innovations [n, n_obs, 3] = (H M(x) - y) masked) from ONE forward + adjoint run (= cost.grad)."""
    c = control.detach().requires_grad_(True)
    args = cost.unpack(c)
    x0, eta = args[0], (args[1] if cost.weak else None)
    traj = window_trajectory(cost.step, x0, cost.cfg, eta)
    innov = (traj[:, cost.t_obs] - cost.y) * cost.mask
    j = cost.jb(x0) + 0.5 * (innov ** 2).sum((-1, -2)) / cost.cfg.sigma_o2
    if cost.weak:
        j = j + 0.5 * (eta ** 2).sum((-1, -2)) / cost.cfg.sigma_q ** 2
    (g,) = torch.autograd.grad(j.sum(), c)
    cost.n_grad_evals += 1
    return j.detach(), g.detach(), innov.detach()


def innov_features(cost: FourDVarCost, innov: torch.Tensor) -> torch.Tensor:
    """[n, 7 n_obs]: innovations / sigma_o (masked), mask, t_i / L (same layout as obs_time_features)."""
    n = innov.shape[0]
    t = (cost.t_obs.to(innov.dtype) / cost.cfg.window).expand(n, -1)
    return torch.cat([(innov / cost.cfg.sigma_o2 ** 0.5).flatten(1), cost.mask.expand(n, -1, -1).flatten(1), t], 1)


@torch.no_grad()
def pair_innov_features(cost: FourDVarCost, x: torch.Tensor, case: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    """Innovation features of training pairs (control x [P, d] of case `case` [P]); forward runs only (training)."""
    out = []
    for i in range(0, x.shape[0], chunk):
        c, cs = x[i:i + chunk], case[i:i + chunk]
        args = cost.unpack(c)
        traj = window_trajectory(cost.step, args[0], cost.cfg, args[1] if cost.weak else None)
        out.append(innov_features(cost, (traj[:, cost.t_obs] - cost.y[cs]) * cost.mask))
    return torch.cat(out)


class DirectNet(torch.nn.Module):
    """[x_std, g_std, emb(k) (, obs features)] -> u [n, d] (direction head + per-block log-magnitude head)."""

    def __init__(self, d: int, hidden: int = 64, n_layers: int = 2, emb_dim: int = 8, log_k: bool = True,
                 d_obs: int = 0, max_log_dev: float = 10.0):
        super().__init__()
        self.d, self.blocks, self.d_obs, self.max_log_dev = d, blocks(d), d_obs, max_log_dev
        nb = len(self.blocks)
        self.emb = IterationEmbedding(emb_dim, log_input=log_k)
        layers, width = [], 2 * d + emb_dim + d_obs
        for _ in range(n_layers):
            layers += [torch.nn.Linear(width, hidden), torch.nn.SiLU()]
            width = hidden
        self.body = torch.nn.Sequential(*layers)
        self.head_dir, self.head_mag = torch.nn.Linear(width, d), torch.nn.Linear(width, nb)
        for lin in (self.head_dir, self.head_mag):
            torch.nn.init.zeros_(lin.weight)
            torch.nn.init.zeros_(lin.bias)
        self.register_buffer("log_mag0", torch.zeros(nb))

    def forward(self, x_std, g_std, k, obs=None):
        parts = [x_std, g_std, self.emb(k, x_std.shape[0])] + ([obs] if self.d_obs else [])
        h = self.body(torch.cat(parts, 1))
        v = g_std + self.head_dir(h)
        lm = self.log_mag0 + self.max_log_dev * torch.tanh(self.head_mag(h) / self.max_log_dev)
        u = torch.empty_like(v)
        for i, (a, b) in enumerate(self.blocks):
            vb = v[:, a:b]
            u[:, a:b] = lm[:, i:i + 1].exp() * vb / vb.pow(2).mean(1, keepdim=True).sqrt().clamp_min(TINY)
        return u


def collect_delta_pairs(cost: FourDVarCost, x0_b: torch.Tensor, method: str, lr: float, max_iter: int = 1000,
                        tol: float = 1e-6, cache: Optional[str] = None) -> dict:
    """Teacher trajectory -> pairs (x^k, g^k, k, case, target Delta_t = x^{k+1} - x^k) over the active iterations
    (case = index of the case in `cost`, for its observation features). cache: optional .pt path (reused if it
    exists and has the case index, so the teacher runs once for all runs)."""
    if cache and Path(cache).exists():
        p = torch.load(cache, weights_only=False)
        if "case" in p:
            return p
    from ..da.fourdvar import run_fourdvar
    traj, grads, act = [], [], []
    res = run_fourdvar(cost, x0_b, method, lr, max_iter, tol,
                       on_step=lambda k, c, g, a: (traj.append(c.clone()), grads.append(g.clone()),
                                                   act.append(a.bool().clone())))
    eta = torch.tensor(res["eta_final"]) if cost.weak else None
    T = torch.stack(traj + [cost.pack(torch.tensor(res["x0_a"]), eta)], 1)     # [n, k_run + 1, d]
    G, A = torch.stack(grads, 1), torch.stack(act, 1)
    out = {key: [] for key in ("x", "g", "k", "case", "target")}
    for k in range(A.shape[1]):
        a = A[:, k]
        if not a.any():
            continue
        out["x"].append(T[a, k]); out["g"].append(G[a, k]); out["target"].append((T[:, k + 1] - T[:, k])[a])  # noqa: E702
        out["case"].append(torch.nonzero(a).flatten())
        out["k"].append(torch.full((int(a.sum()),), k, dtype=T.dtype))
    out = {key: torch.cat(v) for key, v in out.items()}
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, cache)
    return out


class DirectDelta:
    def __init__(self, net: DirectNet, cfg: DirectConfig, ctrl_mean: torch.Tensor, ctrl_std: torch.Tensor):
        self.net, self.cfg, self.ctrl_mean, self.ctrl_std = net, cfg, ctrl_mean, ctrl_std
        self.g_scale0 = torch.ones(len(net.blocks), dtype=ctrl_mean.dtype)   # grad_input fixed: per-block scale

    # ---- case context / features ------------------------------------------------------------------------------
    def obs_feats(self, cost: FourDVarCost) -> Optional[torch.Tensor]:
        """[n_cases, d_obs] observation features of the cases of `cost` (None without obs_input, and with innov
        features, which change every iteration: run_direct computes them from the gradient's forward run)."""
        if not self.net.d_obs or self.cfg.obs_feat == "innov":
            return None
        return obs_time_features(cost, self.ctrl_mean, self.ctrl_std)

    def case_ctx(self, cost: FourDVarCost) -> Optional[dict]:
        """Per-case quantities of the coordinate transform (cvt: Cholesky factor of B, background)."""
        return {"L": cvt_factor(cost), "xb": cost.x0_b} if self.cfg.coords == "cvt" else None

    def prepare_pairs(self, p: dict, cost: FourDVarCost) -> dict:
        """Attach what the loss needs: case context (cvt), observation or innovation features."""
        p = dict(p)
        ctx = self.case_ctx(cost)
        if ctx is not None:
            p.update(ctx)
        if self.net.d_obs:
            if self.cfg.obs_feat == "innov":
                p["obs_pair"] = pair_innov_features(cost, p["x"], p["case"])
            else:
                p["obs"] = obs_time_features(cost, self.ctrl_mean, self.ctrl_std)
        if self.cfg.hist_m:
            p = attach_history(p, self.cfg.hist_m)
        p = attach_horizon(p, self.cfg.horizon)
        if self.cfg.first_step == "teacher":                 # iteration 0 is the classical step: not learned
            keep = p["k"] > 0
            p = {key: (v[keep] if key in PAIR_KEYS else v) for key, v in p.items()}
        return p

    # ---- coordinates --------------------------------------------------------------------------------------------
    def _sq(self):
        return self.ctrl_std[3]                                       # eta block std = sigma_q (control_stats)

    def to_learner(self, x, g, ctx):
        """(x, g) -> learner coordinates: raw = unchanged; cvt = (v, g_v)."""
        if self.cfg.coords != "cvt":
            return x, g
        L, xb = ctx["L"], ctx["xb"]
        v0 = torch.linalg.solve_triangular(L, (x[:, :3] - xb)[..., None], upper=False)[..., 0]
        gv0 = torch.einsum("nji,nj->ni", L, g[:, :3])
        if x.shape[1] == 3:
            return v0, gv0
        return torch.cat([v0, x[:, 3:] / self._sq()], 1), torch.cat([gv0, g[:, 3:] * self._sq()], 1)

    def step_to_learner(self, dx, ctx):
        if self.cfg.coords != "cvt":
            return dx
        dv0 = torch.linalg.solve_triangular(ctx["L"], dx[:, :3, None], upper=False)[..., 0]
        return dv0 if dx.shape[1] == 3 else torch.cat([dv0, dx[:, 3:] / self._sq()], 1)

    def step_to_raw(self, dv, ctx):
        if self.cfg.coords != "cvt":
            return dv
        dx0 = torch.einsum("nij,nj->ni", ctx["L"], dv[:, :3])
        return dx0 if dv.shape[1] == 3 else torch.cat([dx0, dv[:, 3:] * self._sq()], 1)

    def _x_in(self, xl):
        return xl if self.cfg.coords == "cvt" else (xl - self.ctrl_mean) / self.ctrl_std

    def _g_in(self, gl):
        if self.cfg.grad_input == "fixed":
            sc = torch.empty_like(gl[0])
            for i, (a, b) in enumerate(self.net.blocks):
                sc[a:b] = self.g_scale0[i]
            return gl / sc
        return standardize_grad(gl, self.net.d)

    # ---- the update ---------------------------------------------------------------------------------------------
    def base_dir(self, g, hist) -> torch.Tensor:
        """History mode: base descent direction (-g, or -H g by the L-BFGS two-loop over the memory)."""
        return -lbfgs_two_loop(g, *hist) if self.cfg.base == "lbfgs" else -g

    def _delta_hist(self, x, g, k, hist) -> torch.Tensor:
        """Delta_hat = RMS_block(b) * u(x, g, b, k, memory), b = base direction (raw coordinates)."""
        d = self.net.d
        S, Y = hist
        b = self.base_dir(g, hist)
        bs, gs = grad_scale(b, d), grad_scale(g, d)
        feats = (torch.asinh(S / bs[:, None, :]), torch.asinh(Y / gs[:, None, :]),
                 (S.abs().sum(-1) > 0).to(x.dtype))
        return bs * self.net(self._x_in(x), g / gs, b / bs, k, feats)

    def delta_learner(self, x, g, k, obs=None, ctx=None, hist=None) -> torch.Tensor:
        """Delta_hat in learner coordinates. K > 1: inner rollout x_{m+1} = x_m + step(x_m, g, k) / K."""
        if self.cfg.hist_m:
            return self._delta_hist(x, g, k, hist)
        xl, gl = self.to_learner(x, g, ctx)
        gs, g_in = grad_scale(gl, self.net.d), self._g_in(gl)
        K = self.cfg.K
        if K == 1:
            return -gs * self.net(self._x_in(xl), g_in, k, obs)
        xm = xl
        for _ in range(K):
            xm = xm - gs * self.net(self._x_in(xm), g_in, k, obs) / K
        return xm - xl

    def delta(self, x: torch.Tensor, g: torch.Tensor, k, obs: Optional[torch.Tensor] = None,
              ctx: Optional[dict] = None, hist=None) -> torch.Tensor:
        """Delta_hat(x, g, k [, obs]) in raw control coordinates (ctx = case_ctx rows, needed with coords cvt;
        hist = (S, Y) L-BFGS memory [n, m, d], needed with hist_m)."""
        assert self.cfg.coords != "cvt" or ctx is not None, "coords cvt needs the case context"
        return self.step_to_raw(self.delta_learner(x, g, k, obs, ctx, hist), ctx)

    # ---- training -----------------------------------------------------------------------------------------------
    def _rows(self, p: dict, idx):
        x, g, k, t = (p[key][idx] for key in ("x", "g", "k", "target"))
        case = p["case"][idx]
        ctx = {"L": p["L"][case], "xb": p["xb"][case]} if self.cfg.coords == "cvt" else None
        if not self.net.d_obs:
            obs = None
        elif "obs_pair" in p:
            obs = p["obs_pair"][idx]
        else:
            obs = p["obs"][case]
        hist = (p["S"][idx], p["Y"][idx]) if self.cfg.hist_m else None
        return x, g, k, t, obs, ctx, hist

    def init_magnitude(self, p: dict):
        """l0_b = median over train pairs of log RMS(u_t,b): u_t = -Delta_t / RMS_block(g) (learner coordinates), or
        with history u_t = Delta_t / RMS_block(b) (b = base direction); g_scale0_b = median RMS_b(g) (grad_input fixed)."""
        x, g, k, t, obs, ctx, hist = self._rows(p, slice(None))
        _, gl = self.to_learner(x, g, ctx)
        if self.cfg.hist_m:
            u = t / grad_scale(self.base_dir(g, hist), self.net.d)
        else:
            u = -self.step_to_learner(t, ctx) / grad_scale(gl, self.net.d)
        with torch.no_grad():
            for i, (a, b) in enumerate(self.net.blocks):
                r = u[:, a:b].pow(2).mean(1).sqrt()
                r = r[torch.isfinite(r) & (r > 0)]
                self.net.log_mag0[i] = r.log().median()
                gr = gl[:, a:b].pow(2).mean(1).sqrt()
                self.g_scale0[i] = gr[torch.isfinite(gr) & (gr > 0)].median()

    def loss(self, p: dict, idx, reduce: str = "mean", space: str = "learner") -> torch.Tensor:
        """Relative MSE |Delta_hat - Delta_t|^2 / |Delta_t|^2 in learner coordinates (space raw: raw coordinates)."""
        x, g, k, t, obs, ctx, hist = self._rows(p, idx)
        if space == "learner":
            t = self.step_to_learner(t, ctx)
        den = t.pow(2).sum(1)
        ok = torch.isfinite(den) & (den > 0) & torch.isfinite(g).all(1) & torch.isfinite(x).all(1)
        x, g, k, t, den = x[ok], g[ok], k[ok], t[ok], den[ok]
        obs = obs[ok] if obs is not None else None
        ctx = {key: v[ok] for key, v in ctx.items()} if ctx is not None else None
        hist = (hist[0][ok], hist[1][ok]) if hist is not None else None
        pred = self.delta_learner(x, g, k, obs, ctx, hist) if space == "learner" else \
            self.delta(x, g, k, obs, ctx, hist)
        rel = (pred - t).pow(2).sum(1) / den
        if reduce == "median":
            return rel.median()
        return torch.log1p(rel).mean() if reduce == "logmean" else rel.mean()

    def fit(self, train: dict, val: dict, log=None) -> list:
        cfg = self.cfg
        gen = torch.Generator().manual_seed(cfg.seed)
        n = train["x"].shape[0]
        if cfg.max_train_pairs and n > cfg.max_train_pairs:
            sub = torch.randperm(n, generator=gen)[:cfg.max_train_pairs].sort().values
            train = {key: (v[sub] if key in PAIR_KEYS else v) for key, v in train.items()}
            n = cfg.max_train_pairs
        self.init_magnitude(train)
        params = list(self.net.parameters())
        if cfg.optimizer == "lbfgs":
            opt = torch.optim.LBFGS(params, lr=1.0, max_iter=cfg.lbfgs_iters, history_size=cfg.lbfgs_history,
                                    line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-14)
        else:
            opt = torch.optim.Adam(params, lr=cfg.lr)
        best, best_state, hist, n_steps = float("inf"), copy.deepcopy(self.net.state_dict()), [], 0
        vidx, all_idx = torch.arange(val["x"].shape[0]), torch.arange(n)
        if cfg.hist_m:      # epoch -1 = untrained learner (base rule with the median step) is a selectable candidate
            self.net.eval()
            with torch.no_grad():
                best = float(self.loss(val, vidx, cfg.select))
            hist.append({"epoch": -1, "train_rel_median": float(self.loss(train, all_idx, "median")),
                         "val_rel_median": best, "opt_steps": 0})
            if log is not None:
                log(hist[-1])
        for ep in range(cfg.epochs):
            t0, tot = time.perf_counter(), 0.0
            self.net.train()
            if cfg.optimizer == "lbfgs":
                def closure():
                    opt.zero_grad()
                    loss = self.loss(train, all_idx, cfg.train_loss)
                    if torch.isfinite(loss):
                        loss.backward()
                    return loss
                opt.step(closure)
                n_steps = int(opt.state[params[0]].get("n_iter", n_steps))
                with torch.no_grad():
                    tot = float(self.loss(train, all_idx)) * n
            else:
                if cfg.train_mode == "online":   # teacher-run order: at iteration k + h, one step on the moves made at k
                    groups = online_groups(train, cfg.online_cases, gen)
                else:
                    groups = torch.randperm(n, generator=gen).split(cfg.batch_size)
                for i in groups:
                    opt.zero_grad()
                    loss = self.loss(train, i, cfg.train_loss)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip_norm)
                    opt.step()
                    n_steps += 1
                    tot += float(loss.detach()) * len(i)
            self.net.eval()
            with torch.no_grad():
                vl = float(self.loss(val, vidx, cfg.select))    # median (default): mean dominated by a few pairs
                h = {"epoch": ep, "train_rel": tot / n, "train_rel_median": float(self.loss(train, all_idx, "median")),
                     "val_rel_median": vl, "opt_steps": n_steps, "s": time.perf_counter() - t0}
                if cfg.coords != "raw":
                    h["val_rel_median_raw"] = float(self.loss(val, vidx, "median", space="raw"))
            hist.append(h)
            if log is not None:
                log(h)
            if vl < best:
                best, best_state = vl, copy.deepcopy(self.net.state_dict())
        self.net.load_state_dict(best_state)
        self.net.eval()
        return hist

    def state_dict(self) -> dict:
        return {"net": self.net.state_dict(), "cfg": asdict(self.cfg), "d": self.net.d, "d_obs": self.net.d_obs,
                "ctrl_mean": self.ctrl_mean, "ctrl_std": self.ctrl_std, "g_scale0": self.g_scale0,
                "n_obs": int(getattr(self.net, "n_obs", 0))}

    @classmethod
    def from_state_dict(cls, sd: dict) -> "DirectDelta":
        lrn = build_direct(sd["d"], DirectConfig(**sd["cfg"]), sd["ctrl_mean"], sd["ctrl_std"],
                           n_obs=sd.get("n_obs", 0), d_obs=sd.get("d_obs", 0))
        lrn.net.load_state_dict(sd["net"])
        if "g_scale0" in sd:
            lrn.g_scale0 = sd["g_scale0"]
        lrn.net.eval()
        return lrn


def build_direct(d: int, cfg: DirectConfig, ctrl_mean, ctrl_std, n_obs: int = 0, d_obs: Optional[int] = None
                 ) -> DirectDelta:
    """n_obs: observation times of the problem (needed with cfg.obs_input: d_obs = 7 n_obs)."""
    torch.manual_seed(cfg.seed)
    if cfg.hist_m:                                    # history input (replaces the observations), raw coordinates
        from .temporal_net import HistTemporalNet
        assert cfg.coords == "raw" and not cfg.obs_input and n_obs, "hist_m: raw coordinates, no obs input, n_obs"
        net = HistTemporalNet(d, n_obs, cfg.hist_m, hidden=cfg.hidden, n_layers=cfg.n_layers, emb_dim=cfg.emb_dim,
                              kernel=cfg.kernel, log_k=cfg.log_k, max_log_dev=cfg.max_log_dev).to(ctrl_mean.dtype)
        return DirectDelta(net, cfg, ctrl_mean, ctrl_std)
    if d_obs is None:
        d_obs = 7 * n_obs if cfg.obs_input else 0
    if cfg.arch == "temporal":
        from .temporal_net import TemporalDeltaNet
        assert cfg.obs_input and d_obs, "arch temporal needs obs_input"
        net = TemporalDeltaNet(d, d_obs // 7, hidden=cfg.hidden, n_layers=cfg.n_layers, emb_dim=cfg.emb_dim,
                               kernel=cfg.kernel, log_k=cfg.log_k, max_log_dev=cfg.max_log_dev).to(ctrl_mean.dtype)
        return DirectDelta(net, cfg, ctrl_mean, ctrl_std)
    net = DirectNet(d, hidden=cfg.hidden, n_layers=cfg.n_layers, emb_dim=cfg.emb_dim, log_k=cfg.log_k,
                    d_obs=d_obs, max_log_dev=cfg.max_log_dev).to(ctrl_mean.dtype)
    return DirectDelta(net, cfg, ctrl_mean, ctrl_std)


@torch.no_grad()
def run_direct(cost: FourDVarCost, x0_init: torch.Tensor, learner: DirectDelta, max_iter: int = 1000,
               tol: float = 1e-6, eta_init: Optional[torch.Tensor] = None, safeguard: str = "none") -> dict:
    """Frozen inference: x^{k+1} = x^k + Delta_theta(x^k, g^k, k). One gradient per iteration, no teacher; same
    stopping rule and record layout as run_fourdvar / run_meta_b (grad_cum[k] = k gradients).
    safeguard "backtrack" (2026-11-19): the gradient evaluation at x^{k+1} also returns J(x^{k+1}); if J did not
    decrease (or is not finite) the move is REJECTED and halved, x^{k+2} = x^k + (x^{k+1} - x^k) / 2 (memory, stopping
    rule and accepted point unchanged; the rejected gradient is counted). No extra model run. J_iter records the J of
    the last accepted iterate, and the returned analysis is the last accepted iterate."""
    n = x0_init.shape[0]
    if cost.weak and eta_init is None:
        eta_init = torch.zeros(n, cost.cfg.n_obs - 1, 3, dtype=x0_init.dtype)
    c = cost.pack(x0_init, eta_init).detach().clone()
    learner.net.eval()
    obs, ctx = learner.obs_feats(cost), learner.case_ctx(cost)
    use_innov = bool(learner.net.d_obs) and learner.cfg.obs_feat == "innov"
    m = learner.cfg.hist_m
    S = Y = None
    if m:                                                      # own L-BFGS memory (oldest -> newest, zero = none)
        S, Y = torch.zeros(n, m, c.shape[1], dtype=c.dtype), torch.zeros(n, m, c.shape[1], dtype=c.dtype)
    c_prev = g_prev = None
    teacher = None
    if learner.cfg.first_step == "teacher":                    # iteration 0 = one strong-Wolfe L-BFGS iteration
        from ..da.reference_step import ReferenceStep
        teacher = ReferenceStep(c, method="lbfgs_wolfe", lr=1.0, cost=cost)
    outer = torch.zeros(n, dtype=torch.long)
    active = torch.ones(n, dtype=c.dtype)
    n_evals = torch.zeros(n, dtype=torch.long)
    x0_hist, j_hist, cos_hist, grad_hist = [c[:, :3].clone()], [], [], [torch.zeros(n, dtype=torch.long)]
    t0 = time.perf_counter()
    j_prev = None
    for k in range(max_iter):
        with torch.enable_grad():
            if use_innov:                                      # innovations from the gradient's own forward run
                j, g, inn = grad_innov(cost, c)
                obs = innov_features(cost, inn)
            else:
                j, g = cost.grad(c)
        n_evals += active.long()
        # cost: a gradient already evaluated by the first step's line search (accepted point) is not counted again
        free = teacher.is_free(c) if teacher is not None else torch.zeros(n, dtype=torch.bool)
        outer += (active.bool() & ~free).long()
        rej = torch.zeros(n, dtype=torch.bool)
        if safeguard == "backtrack" and j_prev is not None:
            rej = active.bool() & ~(torch.isfinite(j) & (j <= j_prev))
            j = torch.where(rej, j_prev, j)                     # J of the last accepted iterate
        j_hist.append(j.clone() if j_prev is None else torch.where(active.bool(), j, j_hist[-1]))
        if j_prev is not None:
            rel = (j_prev - j).abs() / j_prev.abs().clamp_min(1e-300)
            active = active * (~(active.bool() & ~rej & ((rel < tol) | ~torch.isfinite(j)))).to(c.dtype)
        if active.sum() == 0:
            break
        acc = ~rej
        if m and c_prev is not None:                            # memory update with the move actually applied
            s, y = lbfgs_pair(c - c_prev, g - g_prev)
            S = torch.where(acc[:, None, None], torch.cat([S[:, 1:], s[:, None]], 1), S)
            Y = torch.where(acc[:, None, None], torch.cat([Y[:, 1:], y[:, None]], 1), Y)
        if c_prev is None:
            c_prev, g_prev = c.clone(), g.clone()
        else:
            c_prev = torch.where(acc[:, None], c, c_prev)
            g_prev = torch.where(acc[:, None], g, g_prev)
        if k == 0 and teacher is not None:
            step = teacher(c, g, active, j=j)                   # classical first step from the initial gradient
        else:                                                   # accepted cases: c_prev = c, g_prev = g
            step = -learner.delta(c_prev, g_prev, k, obs, ctx, (S, Y) if m else None) * active[:, None]
        grad_hist.append(outer + (teacher.n_ls if teacher is not None else 0))
        cos_hist.append(torch.nn.functional.cosine_similarity(step, g, dim=1))
        # rejected cases: halve the rejected move from the last accepted point
        c = torch.where(rej[:, None], c_prev + (c - c_prev) / 2, c_prev - step)
        x0_hist.append(c[:, :3].clone())
        j_prev = j
    wall = time.perf_counter() - t0

    j_final = cost(*cost.unpack(c))
    if safeguard == "backtrack" and j_prev is not None:     # never return a trial point worse than the accepted one
        worse = ~(torch.isfinite(j_final) & (j_final <= j_prev))
        c = torch.where(worse[:, None], c_prev, c)
        j_final = torch.where(worse, j_prev, j_final)
    j_hist.append(j_final)
    k_run = len(x0_hist) - 1
    pad = max_iter - k_run
    x0_iter = torch.stack(x0_hist, 1)
    x0_iter = torch.cat([x0_iter, x0_iter[:, -1:].expand(n, pad, 3)], 1) if pad > 0 else x0_iter
    j_iter = torch.stack(j_hist[: k_run + 1], 1)
    j_iter = torch.cat([j_iter, j_iter[:, -1:].expand(n, pad)], 1) if pad > 0 else j_iter
    cos = torch.stack(cos_hist, 1) if cos_hist else torch.zeros(n, 0, dtype=c.dtype)
    cos = torch.cat([cos, torch.full((n, max_iter - cos.shape[1]), float("nan"), dtype=c.dtype)], 1)
    out = {"x0_iter": x0_iter.numpy(), "J_iter": j_iter.numpy(), "cos_step_grad": cos.numpy(),
           "n_grad_evals": n_evals.numpy(),
           "converged": ((active == 0) & (n_evals < max_iter) & torch.isfinite(j_final)).numpy(),
           "x0_a": c[:, :3].numpy(), "wall_s": wall, "control_final": c.numpy(),
           "grad_cum": _pad_cum(grad_hist, max_iter)}
    if cost.weak:
        out["eta_final"] = cost.unpack(c)[1].numpy()
    return out

