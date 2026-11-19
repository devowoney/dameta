"""
Temporal delta network (2026-11-18, pilot: observations "concatenated in time"): the weak-constraint control lives on
the observation times (x0 at t_0, eta_i at t_i, i = 1..n_obs-1), so the module works on one TOKEN per observation
time instead of one flat vector (flat obs input = 7 n_obs = 777 extra inputs for the dense problem: overfits).

    token_i = [ctrl_i (x0 or eta_i, standardised, 3), g_std_i (3), y_i standardised and masked (3), mask_i (3),
               t_i / L (1), emb(k)]                        strong constraint: ctrl_i, g_i = 0 for i >= 1 (no control)
    h_i     = SiLU(W_in token_i)                                                   shared over time
    h       <- h + SiLU(Conv1d_k(h))   x n_layers                                  local mixing along time (kernel 5)
    v_i     = g_std_i + W_dir h_i                                                  direction per token (zero-init)
    l_x0    = l0_x0  + bound(w_x0 . h_0),   l_eta = l0_eta + bound(w_eta . mean_{i>=1} h_i)   log step size per block
    u_b     = exp(l_b) v_b / RMS(v_b)          Delta_hat = -RMS_block(g) * u     (same output law as DirectNet)

Same interface as DirectNet (forward(x_std, g_std, k, obs) with obs = obs_time_features [n, 7 n_obs]; attributes d,
blocks, d_obs, log_mag0), so DirectDelta / run_direct / the free-delta trainer use it unchanged.
"""

from typing import Union

import torch
import torch.nn as nn

from .networks import IterationEmbedding

TINY = torch.finfo(torch.float64).tiny


class TemporalDeltaNet(nn.Module):
    def __init__(self, d: int, n_obs: int, hidden: int = 32, n_layers: int = 2, emb_dim: int = 8, kernel: int = 5,
                 log_k: bool = True, max_log_dev: float = 10.0):
        super().__init__()
        assert d in (3, 3 * n_obs), f"control dim {d} is neither strong (3) nor weak (3 n_obs = {3 * n_obs})"
        self.d, self.n_obs, self.d_obs, self.max_log_dev = d, n_obs, 7 * n_obs, max_log_dev
        self.blocks = [(0, 3)] + ([(3, d)] if d > 3 else [])
        self.emb = IterationEmbedding(emb_dim, log_input=log_k)
        self.inp = nn.Linear(13 + emb_dim, hidden)
        self.convs = nn.ModuleList([nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2) for _ in range(n_layers)])
        self.head_dir = nn.Linear(hidden, 3)
        self.head_mag = nn.ModuleList([nn.Linear(hidden, 1) for _ in self.blocks])
        for lin in [self.head_dir, *self.head_mag]:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.register_buffer("log_mag0", torch.zeros(len(self.blocks)))

    def _tokens(self, a: torch.Tensor) -> torch.Tensor:
        """[n, d] control-shaped -> [n, n_obs, 3] (strong: zeros after token 0)."""
        n = a.shape[0]
        if self.d == 3 * self.n_obs:
            return a.reshape(n, self.n_obs, 3)
        out = a.new_zeros(n, self.n_obs, 3)
        out[:, 0] = a
        return out

    def forward(self, x_std: torch.Tensor, g_std: torch.Tensor, k: Union[int, torch.Tensor], obs=None) -> torch.Tensor:
        n, T = x_std.shape[0], self.n_obs
        assert obs is not None, "TemporalDeltaNet needs the observation features (obs_input)"
        y, m, t = obs[:, :3 * T].reshape(n, T, 3), obs[:, 3 * T:6 * T].reshape(n, T, 3), obs[:, 6 * T:].reshape(n, T, 1)
        e = self.emb(k, n)[:, None, :].expand(n, T, -1)
        tok = torch.cat([self._tokens(x_std), self._tokens(g_std), y, m, t, e], -1)    # [n, T, 13 + emb]
        h = torch.nn.functional.silu(self.inp(tok)).transpose(1, 2)                     # [n, hidden, T]
        for conv in self.convs:
            h = h + torch.nn.functional.silu(conv(h))
        h = h.transpose(1, 2)                                                            # [n, T, hidden]
        v = (self._tokens(g_std) + self.head_dir(h)).reshape(n, 3 * T)[:, :self.d]
        pooled = [h[:, 0]] + ([h[:, 1:].mean(1)] if len(self.blocks) > 1 else [])
        u = torch.empty_like(v)
        for i, ((a, b), hp) in enumerate(zip(self.blocks, pooled)):
            lm = self.log_mag0[i] + self.max_log_dev * torch.tanh(self.head_mag[i](hp) / self.max_log_dev)  # [n, 1]
            vb = v[:, a:b]
            u[:, a:b] = lm.exp() * vb / vb.pow(2).mean(1, keepdim=True).sqrt().clamp_min(TINY)
        return u


class HistTemporalNet(nn.Module):
    """
    History delta network (2026-11-18, pilot: "instead of observations, history as input"): same temporal layout
    (one token per observation time, residual Conv1d along time) but the per-token features are the optimiser
    history instead of the observations:

        token_i = [ctrl_i (3), g_std_i (3), b_std_i (3), {s~_j,i (3), y~_j,i (3), flag_j}_{j=1..m}, t_i / L, emb(k)]

    b = base descent direction (-g, or the L-BFGS two-loop direction -H g of the same memory; meta_b_direct),
    s_j / y_j = the m most recent moves / gradient differences (L-BFGS memory, oldest -> newest, zero rows = no pair),
    s~ = asinh(s / RMS_block(b)), y~ = asinh(y / RMS_block(g)), flag = pair present.
    Output u (same law as DirectNet, relative to the BASE): v = b_std + W_dir h, u_b = exp(l_b) v_b / RMS(v_b);
    the caller applies Delta = RMS_block(b) * u. Zero-initialised heads: Delta = exp(l0) b (base step, fixed length).
    """

    def __init__(self, d: int, n_obs: int, m: int, hidden: int = 32, n_layers: int = 2, emb_dim: int = 8,
                 kernel: int = 5, log_k: bool = True, max_log_dev: float = 10.0):
        super().__init__()
        assert d in (3, 3 * n_obs), f"control dim {d} is neither strong (3) nor weak (3 n_obs = {3 * n_obs})"
        self.d, self.n_obs, self.m, self.d_obs, self.max_log_dev = d, n_obs, m, 0, max_log_dev
        self.blocks = [(0, 3)] + ([(3, d)] if d > 3 else [])
        self.emb = IterationEmbedding(emb_dim, log_input=log_k)
        self.inp = nn.Linear(9 + 7 * m + 1 + emb_dim, hidden)
        self.convs = nn.ModuleList([nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2) for _ in range(n_layers)])
        self.head_dir = nn.Linear(hidden, 3)
        self.head_mag = nn.ModuleList([nn.Linear(hidden, 1) for _ in self.blocks])
        for lin in [self.head_dir, *self.head_mag]:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.register_buffer("log_mag0", torch.zeros(len(self.blocks)))
        self.register_buffer("t_rel", torch.linspace(0.0, 1.0, n_obs))

    def _tokens(self, a: torch.Tensor) -> torch.Tensor:
        """[..., d] control-shaped -> [..., n_obs, 3] (strong: zeros after token 0)."""
        if self.d == 3 * self.n_obs:
            return a.reshape(*a.shape[:-1], self.n_obs, 3)
        out = a.new_zeros(*a.shape[:-1], self.n_obs, 3)
        out[..., 0, :] = a
        return out

    def forward(self, x_std, g_std, b_std, k: Union[int, torch.Tensor], hist) -> torch.Tensor:
        """hist = (S~ [n, m, d], Y~ [n, m, d], flags [n, m]) already scaled."""
        n, T, m = x_std.shape[0], self.n_obs, self.m
        S, Y, fl = hist
        st = self._tokens(S).permute(0, 2, 1, 3)                                         # [n, T, m, 3]
        yt = self._tokens(Y).permute(0, 2, 1, 3)
        ft = fl[:, None, :, None].expand(n, T, m, 1)
        h_tok = torch.cat([st, yt, ft], -1).reshape(n, T, 7 * m)
        e = self.emb(k, n)[:, None, :].expand(n, T, -1)
        t = self.t_rel[None, :, None].expand(n, T, 1).to(x_std.dtype)
        tok = torch.cat([self._tokens(x_std), self._tokens(g_std), self._tokens(b_std), h_tok, t, e], -1)
        h = torch.nn.functional.silu(self.inp(tok)).transpose(1, 2)
        for conv in self.convs:
            h = h + torch.nn.functional.silu(conv(h))
        h = h.transpose(1, 2)
        v = (self._tokens(b_std) + self.head_dir(h)).reshape(n, 3 * T)[:, :self.d]
        pooled = [h[:, 0]] + ([h[:, 1:].mean(1)] if len(self.blocks) > 1 else [])
        u = torch.empty_like(v)
        for i, ((a, b), hp) in enumerate(zip(self.blocks, pooled)):
            lm = self.log_mag0[i] + self.max_log_dev * torch.tanh(self.head_mag[i](hp) / self.max_log_dev)
            vb = v[:, a:b]
            u[:, a:b] = lm.exp() * vb / vb.pow(2).mean(1, keepdim=True).sqrt().clamp_min(TINY)
        return u
