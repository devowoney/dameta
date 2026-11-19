"""
Diagnostic (2026-11-18): can the supervised direct-delta learner imitate the strong-Wolfe L-BFGS teacher if it sees
more data, or does it lack information (L-BFGS history)? Dense-obs weak constraint, F1, sigma_q 0.1/sqrt(10).

    python src/diag_imitation.py collect <split> <start> <end>   # teacher run on cases [start, end) -> chunk file
    python src/diag_imitation.py fit <n_train_cases> <history 0|1> <lr> <epochs>   # one offline fit, prints a row

Pairs: (x^k, g^k, k, s^{k-1} = x^k - x^{k-1}, y^{k-1} = g^k - g^{k-1}) -> target x^{k+1} - x^k, active iterations.
Network = the DirectNet head (direction + per-block log magnitude) with optional history input [s_std, y_std].
Metric: median relative error |pred - target|^2 / |target|^2 on train pairs and on the 20 val cases.
"""

import copy
import glob
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_meta_b as R  # noqa: E402
from dameta.da.fourdvar import run_fourdvar  # noqa: E402
from dameta.meta import control_stats  # noqa: E402
from dameta.meta.meta_a import standardize_grad  # noqa: E402
from dameta.meta.meta_b_direct import blocks, grad_scale  # noqa: E402
from dameta.meta.networks import IterationEmbedding  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
OUT = Path("tmp/meta/dev_delta/imit")
CFG = OmegaConf.load("src/configs/meta_b.yaml")
CFG.merge_with_dotlist(["dataset=v2", "data_prefix=l63v2d", "model=f1", "constraint=weak", "pattern=full",
                        "cost.sigma_q=0.0316227766"])
KEYS = ("x", "g", "k", "t", "s", "y")


def collect(split, a, b):
    pb = R.problem(CFG, split, b)
    idx = np.arange(a, b)
    rec = []
    run_fourdvar(pb.cost(idx), pb.x0_b[idx], "lbfgs_wolfe", 1.0, 150, 1e-6,
                 on_step=lambda k, c, g, act: rec.append((c.clone(), g.clone(), act.bool().clone())))
    C, G, A = (torch.stack([r[i] for r in rec], 1) for i in range(3))
    out = {key: [] for key in KEYS}
    for i in range(C.shape[0]):
        for k in torch.nonzero(A[i]).flatten().tolist():
            if k + 1 >= C.shape[1]:
                continue
            z = torch.zeros_like(C[i, k])
            out["x"].append(C[i, k]); out["g"].append(G[i, k]); out["k"].append(float(k))  # noqa: E702
            out["t"].append(C[i, k + 1] - C[i, k])
            out["s"].append(C[i, k] - C[i, k - 1] if k > 0 else z)
            out["y"].append(G[i, k] - G[i, k - 1] if k > 0 else z)
    p = {key: torch.stack(v) if key != "k" else torch.tensor(v) for key, v in out.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(p, OUT / f"{split}_{a:04d}.pt")
    if split == "train" and a == 0:
        torch.save(control_stats(pb.x_true, pb.ccfg.n_obs, pb.ccfg.sigma_q, True), OUT / "ctrl_stats.pt")


def load(split, n_cases=None):
    parts = [torch.load(f, weights_only=False) for f in sorted(glob.glob(str(OUT / f"{split}_*.pt")))]
    p = {key: torch.cat([q[key] for q in parts]) for key in KEYS}
    if n_cases is not None:
        starts = torch.nonzero(p["k"] == 0).flatten()
        m = int(starts[n_cases]) if n_cases < len(starts) else len(p["k"])
        p = {key: v[:m] for key, v in p.items()}
    return p


class Net(torch.nn.Module):
    def __init__(self, d, hidden=64, n_layers=2, history=False):
        super().__init__()
        self.d, self.blocks, self.history = d, blocks(d), history
        self.emb = IterationEmbedding(8)
        width, layers = (4 if history else 2) * d + 8, []
        for _ in range(n_layers):
            layers += [torch.nn.Linear(width, hidden), torch.nn.SiLU()]
            width = hidden
        self.body = torch.nn.Sequential(*layers)
        self.hd, self.hm = torch.nn.Linear(width, d), torch.nn.Linear(width, len(self.blocks))
        for lin in (self.hd, self.hm):
            torch.nn.init.zeros_(lin.weight)
            torch.nn.init.zeros_(lin.bias)
        self.register_buffer("l0", torch.zeros(len(self.blocks)))

    def delta(self, p, idx, cm, cs):
        x, g, k = p["x"][idx], p["g"][idx], p["k"][idx]
        gs = standardize_grad(g, self.d)
        parts = [(x - cm) / cs, gs, self.emb(k, x.shape[0])]
        if self.history:
            parts += [standardize_grad(p["s"][idx], self.d).nan_to_num(), standardize_grad(p["y"][idx], self.d).nan_to_num()]
        h = self.body(torch.cat(parts, 1))
        v = gs + self.hd(h)
        lm = self.l0 + 10 * torch.tanh(self.hm(h) / 10)
        u = torch.empty_like(v)
        for i, (a, b) in enumerate(self.blocks):
            vb = v[:, a:b]
            u[:, a:b] = lm[:, i:i + 1].exp() * vb / vb.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-300)
        return -grad_scale(g, self.d) * u


def rel_err(net, p, idx, cm, cs):
    t = p["t"][idx]
    return (net.delta(p, idx, cm, cs) - t).pow(2).sum(1) / t.pow(2).sum(1)


def fit(n_cases, history, lr, epochs, bs=256):
    tr, va = load("train", n_cases), load("val")
    cm, cs = torch.load(OUT / "ctrl_stats.pt", weights_only=False)
    torch.manual_seed(0)
    d = tr["x"].shape[1]
    net = Net(d, history=history)
    u = -tr["t"] / grad_scale(tr["g"], d)
    with torch.no_grad():
        for i, (a, b) in enumerate(net.blocks):
            net.l0[i] = u[:, a:b].pow(2).mean(1).sqrt().log().median()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n, best, t0 = tr["x"].shape[0], (np.inf, None, -1), time.perf_counter()
    vi = torch.arange(va["x"].shape[0])
    for ep in range(epochs):
        for i in torch.randperm(n).split(bs):
            opt.zero_grad()
            rel_err(net, tr, i, cm, cs).mean().backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        with torch.no_grad():
            v = float(rel_err(net, va, vi, cm, cs).median())
        if v < best[0]:
            best = (v, copy.deepcopy(net.state_dict()), ep)
    net.load_state_dict(best[1])
    with torch.no_grad():
        trm = float(rel_err(net, tr, torch.arange(n), cm, cs).median())
        cos = torch.nn.functional.cosine_similarity(net.delta(va, vi, cm, cs), va["t"], dim=1)
    print(f"cases {n_cases:5d} | history {int(history)} | lr {lr:g} | pairs {n:6d} | best epoch {best[2]:3d}/{epochs} | "
          f"train median rel {trm:.3f} | val median rel {best[0]:.3f} | val cos(pred, teacher) {float(cos.median()):.3f} | "
          f"{time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        collect(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        fit(int(sys.argv[2]), bool(int(sys.argv[3])), float(sys.argv[4]), int(sys.argv[5]))
