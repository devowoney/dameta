"""
Time-lagged false-model backgrounds for D1 v2 (memory/branches/m5-meta-grid/blueprint.md section 3).

For every case and every lag tau_j (steps), the FALSE model is run from the true state at t0 - tau_j up to t0:
    member_j = M_F^(tau_j) ( x_true(t0 - s - tau_j) ),    tau_j in `lags` (default 10, 20, ..., 100 steps)
    s        = time_shift (steps): the members are valid at t0 - s but used as states at t0, so the first guess carries
               the model error of a short forecast AND a timing (temporal misrepresentation) error (pilot 2026-11-25;
               s = 0 reproduces the earlier background)
    x0_b     = member at lag tau_b (default 50 steps = 0.5 TU)  — ONE balanced false-model state, never the mean
    B        = sample covariance of the members (N - 1) + floor * tr(B) / 3 * I     (case-specific, flow-dependent)
The first guess and B therefore carry the false model's own dynamics and model error; the truth never enters x0_b
directly. Output schema (tmp/data/l63v2_<split>_seed<S>_bg_<name>.nc):
    x0_b (ic, var), B (ic, var, var2), members (ic, member, var), lag_steps (member), attrs: false model, tau_b, floor.
"""

from typing import Callable, Iterable

import numpy as np
import torch
import xarray as xr


@torch.no_grad()
def lagged_background(ds: xr.Dataset, step: Callable, name: str, lags: Iterable[int] = range(10, 101, 10),
                      tau_b: int = 50, floor: float = 0.01, model_desc: str = "",
                      time_shift: int = 0) -> xr.Dataset:
    lags = list(lags)
    assert tau_b in lags, "the first guess must be one of the lagged members"
    assert max(lags) + time_shift <= int(ds.lag.max()), "lag beyond the stored truth history"
    past = torch.tensor(ds.x_true_past.values)                      # [n, history + 1, 3], lag 0 = t0
    members = []
    for tau in lags:
        x = past[:, tau + time_shift]
        for _ in range(tau):
            x = step(x)
        members.append(x)
    mem = torch.stack(members, 1)                                    # [n, N, 3]
    dev = mem - mem.mean(1, keepdim=True)
    b = dev.transpose(1, 2) @ dev / (mem.shape[1] - 1)               # [n, 3, 3]
    tr = torch.diagonal(b, dim1=1, dim2=2).sum(-1)
    b = b + floor * (tr / 3)[:, None, None] * torch.eye(3, dtype=b.dtype)
    x0_b = mem[:, lags.index(tau_b)]
    return xr.Dataset(
        data_vars={"x0_b": (("ic", "var"), x0_b.numpy()), "B": (("ic", "var", "var2"), b.numpy()),
                   "members": (("ic", "member", "var"), mem.numpy()), "lag_steps": (("member",), np.array(lags))},
        coords={"ic": ds.ic.values, "var": ["x", "y", "z"], "var2": ["x", "y", "z"], "member": np.arange(len(lags))},
        attrs={"false_model": name, "false_model_desc": model_desc, "tau_b": int(tau_b), "floor": float(floor),
               "time_shift": int(time_shift),
               "split": ds.attrs["split"], "seed": int(ds.attrs["seed"]), "dataset": "D1v2-background"},
    )


def load_hybrid_background(data_dir, split: str, name: str, seed: int = 42, hybrid: dict = None):
    """
    First guess and HYBRID background covariance of false model `name` (pilot 2026-11-14):
        B_h = alpha * B_ens + (1 - alpha) * s2 * I     (alpha, s2 tuned truth-free by src/tune_hybrid_b.py)
    Returns torch tensors x0_b [n, 3], B_h [n, 3, 3], B_h^-1 [n, 3, 3] and the (alpha, s2) used.
    """
    import json
    from pathlib import Path
    data_dir = Path(data_dir)
    bg = xr.load_dataset(data_dir / f"l63v2_{split}_seed{seed}_bg_{name}.nc")
    if hybrid is None:
        hybrid = json.loads((data_dir / f"hybrid_B_seed{seed}.json").read_text())[name]
    a, s2 = float(hybrid["alpha"]), float(hybrid["s2"])
    b_ens = torch.tensor(bg.B.values)
    b_h = a * b_ens + (1 - a) * s2 * torch.eye(3, dtype=b_ens.dtype)
    return torch.tensor(bg.x0_b.values), b_h, torch.linalg.inv(b_h), {"alpha": a, "s2": s2}


def load_static_background(data_dir, split: str, name: str, seed: int = 42):
    """
    First guess of false model `name` and its STATIC, truth-free B (pilot 2026-11-26; src/estimate_b.py):
    one 3x3 B per model, B = E[d d^T] - R from the TRAIN t0 innovations d = y_0 - x0_b, same for every case.
    Returns torch tensors x0_b [n, 3], B [n, 3, 3] (expanded), B^-1 [n, 3, 3] and an info dict.
    """
    import json
    from pathlib import Path
    data_dir = Path(data_dir)
    bg = xr.load_dataset(data_dir / f"l63v2_{split}_seed{seed}_bg_{name}.nc")
    info = json.loads((data_dir / f"static_B_seed{seed}.json").read_text())[name]
    b = torch.tensor(info["B"], dtype=torch.float64)
    n = bg.sizes["ic"]
    b_n = b.expand(n, 3, 3).clone()
    return torch.tensor(bg.x0_b.values), b_n, torch.linalg.inv(b_n), {"cov": "static", "B": info["B"]}

