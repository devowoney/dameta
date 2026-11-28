"""
Quantitative model quality (pilot 2026-11-28): local error m and window error w of a model M against a reference
M_ref, on the D1 v2 test windows of the current M8 configuration (window / obs interval / sigma_o read from the data).

    m(M; M_ref) = (1/sigma_o) sqrt( (1/3) sum_v Var_{x ~ truth}[ (M^{dt_o}(x) - M_ref^{dt_o}(x))_v ] )
        x = true states at the obs times inside the window (last one excluded), pooled over cases and times;
        one obs interval = obs_every steps.
    w_i(M; M_ref) = (1/sigma_o) sqrt( (1/(3K)) sum_k |M^{t_k}(x0) - M_ref^{t_k}(x0)|^2 ),  x0 = true state at window
        start of case i, t_k = the K obs times of the window (t_0 = 0 included, error 0 there); mean +- std over cases.
    E[2 J_o(x0_true)] = p (1 + w^2) for a model run from the true x0 (J_o = 1/2 sum |y - x|^2 / sigma_o^2, p = 3K, full obs).

References per model: its teacher (emulator approximation) and L1 (total error).

Usage (project root, compute node):  python src/model_quality.py [e1 e2 e1b e2b] [--split test] [--tag x]
    A model can also be an emulator file not in the registry: <file>:<teacher>, e.g. e1_d1w4d1:f1 (screening).
Writes tmp/results/models/quality_<split>[_<tag>].{json,md}; stable_20TU = 50 free runs of 20 TU finite, |x| < 100.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dameta.dynamics.emulator import L63Emulator  # noqa: E402
from dameta.dynamics.models import build_model, canonical, load_registry  # noqa: E402

torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def advance(step, x, n):
    for _ in range(n):
        x = step(x)
    return x


@torch.no_grad()
def at_obs(step, x0, t_obs):
    """States at the obs times t_obs (steps from x0), shape (cases, K, 3)."""
    out, x, t = [], x0, 0
    for tk in t_obs:
        x = advance(step, x, tk - t)
        t = tk
        out.append(x)
    return torch.stack(out, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=["e1", "e2", "e1b", "e2b"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    reg = load_registry()
    ds = xr.load_dataset(ROOT / f"tmp/data/l63v2_{a.split}_seed42.nc")
    sigma_o = float(np.sqrt(ds.attrs["sigma_o2"]))
    every, window = int(ds.attrs["obs_every"]), int(ds.attrs["window"])
    t_obs = [int(t) for t in ds.t_obs.values if t <= window]
    xt = torch.tensor(ds.x_true.values)
    x0 = xt[:, 0]
    starts = xt[:, t_obs[:-1]].reshape(-1, 3)                     # local-error start states (inside the window)

    cache, steps = {}, {}

    def get_step(name):
        if name in steps:
            return steps[name]
        if ":" in name:
            emu = L63Emulator.load(ROOT / f"tmp/emulators/{name.split(':')[0]}.pt")
            for q in emu.parameters():
                q.requires_grad_(False)
            steps[name] = emu.step
        else:
            steps[name] = build_model(name, reg, str(ROOT / "tmp/emulators"))[0]
        return steps[name]

    def maps(name):
        if name not in cache:
            step = get_step(name)
            cache[name] = (advance(step, starts, every), at_obs(step, x0, t_obs))
        return cache[name]

    res = {"split": a.split, "n_cases": int(len(xt)), "sigma_o": sigma_o, "obs_every": every, "window": window,
           "K": len(t_obs), "n_local_samples": int(len(starts)), "models": {}}
    for name in a.models:
        if ":" in name:
            teacher = name.split(":")[1]
        else:
            name = canonical(name, reg)
            spec = reg.models[name]
            teacher = str(spec.teacher) if spec.kind == "emulator" else "l1"   # false model: reference = truth
        loc, win = maps(name)
        x, big = x0[:50], 0.0
        with torch.no_grad():
            for _ in range(2000):                                     # 20 TU free runs (max |x| over the whole run)
                x = get_step(name)(x)
                big = max(big, float(x.abs().max())) if torch.isfinite(x).all() else float("inf")
        rec = {"teacher": teacher, "stable_20TU": big < 100}
        for ref in (teacher, "l1"):
            loc_r, win_r = maps(ref)
            m = float((loc - loc_r).var(0).mean().sqrt()) / sigma_o
            w = (((win - win_r) ** 2).mean((1, 2)).sqrt() / sigma_o).numpy()
            ok = np.isfinite(w)
            rec[f"vs_{ref if ref == 'l1' else 'teacher'}"] = {
                "m": m, "w_mean": float(w[ok].mean()), "w_std": float(w[ok].std()),
                "w_median": float(np.median(w[ok])), "frac_w_gt_1": float((w[ok] > 1).mean()), "n_finite": int(ok.sum())}
        if teacher == "l1":                                           # L1 / false model: own reference = truth
            rec["vs_teacher"] = rec["vs_l1"]
        res["models"][name] = rec
        print(name, json.dumps(rec), flush=True)

    out = ROOT / f"tmp/results/models/quality_{a.split}{'_' + a.tag if a.tag else ''}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    lines = [f"Split {a.split}, {res['n_cases']} cases; window {window} steps, obs every {every} (K = {len(t_obs)} incl. "
             f"t0), sigma_o = {sigma_o:g}; m pooled over {len(starts)} one-interval samples.", "",
             "| model | teacher | stable 20 TU | m vs teacher | w vs teacher (mean ± std) | frac w>1 | m vs L1 | w vs L1 (mean ± std) "
             "| frac w>1 |", "|---|---|---|---|---|---|---|---|---|"]
    for n, r in res["models"].items():
        t, l = r["vs_teacher"], r["vs_l1"]
        lines.append(f"| {n.upper()} | {r['teacher'].upper()} | {r['stable_20TU']} | {t['m']:.3f} | {t['w_mean']:.3f} ± {t['w_std']:.3f} | "
                     f"{t['frac_w_gt_1']:.2f} | {l['m']:.3f} | {l['w_mean']:.3f} ± {l['w_std']:.3f} | "
                     f"{l['frac_w_gt_1']:.2f} |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
