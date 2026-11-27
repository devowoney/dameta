"""
Static, truth-free background error covariance B per estimation model (M8, pilot 2026-11-26).

One fixed 3x3 B per model (F1, F2, E1, E2), the same for every case / window (replaces the flow-dependent hybrid
B = alpha B_ens + (1 - alpha) s2 I, whose lagged ensemble started from the TRUE past states):
    d   = y_obs(t0) - x0_b                (innovation at t0; full-state observation stored in the data; truth not used)
    S   = E[d d^T]  over the TRAIN split   (second moment: first-guess bias included, same convention as Q)
    B_m = S - R,  R = sigma_o2 I           (innovation covariance = B + R for unbiased, independent errors)
    eigenvalues floored at 1 % of tr(B_m) / 3 to keep B positive definite (the old `floor` convention)
Diagnostic only (uses the truth, never fed back): chi2 = e^T B^-1 e, e = x0_b - x_true(t0), mean ~ 3 if calibrated.

Usage (project root):  python src/estimate_b.py [--data-dir tmp/data] [--split train] [--models f1 f2 e1 e2]
                       -> <data-dir>/static_B_seed42.json (merged: entries of other models are kept)
M8-R (2026-11-26): --models l1 (tmp/data) / f1 (tmp/data_tF1) / f2 (tmp/data_tF2) = the shared truth-model first guess.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
MODELS = ("f1", "f2", "e1", "e2")
FLOOR = 0.01


def chi2(e, b):
    return np.einsum("ni,ij,nj->n", e, np.linalg.inv(b), e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="tmp/data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    a = ap.parse_args()
    d_dir = (ROOT / a.data_dir).resolve()
    ds = xr.load_dataset(d_dir / f"l63v2_{a.split}_seed{a.seed}.nc")
    r = float(ds.attrs["sigma_o2"]) * np.eye(3)
    path = d_dir / f"static_B_seed{a.seed}.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    for m in a.models:
        bg = xr.load_dataset(d_dir / f"l63v2_{a.split}_seed{a.seed}_bg_{m}.nc")
        d = ds.y_obs.values[:, 0] - bg.x0_b.values                   # [n, 3]
        s = d.T @ d / len(d)
        b = s - r
        w, v = np.linalg.eigh(b)
        fl = FLOOR * np.trace(b) / 3
        floored = bool((w < fl).any())
        b = (v * np.maximum(w, fl)) @ v.T
        b = 0.5 * (b + b.T)
        res = {"B": b.tolist(), "eig_raw": w.tolist(), "eig": np.linalg.eigvalsh(b).tolist(), "floored": floored,
               "innov_mean": d.mean(0).tolist(), "n": int(len(d)), "split": a.split,
               "sigma_o2": float(ds.attrs["sigma_o2"]), "convention": "E[d d^T] - R, eig floor 1% tr/3"}
        for sp in ("train", "val", "test", "test_ood"):
            xt = xr.load_dataset(d_dir / f"l63v2_{sp}_seed{a.seed}.nc").x_true.values[:, 0]
            e = xr.load_dataset(d_dir / f"l63v2_{sp}_seed{a.seed}_bg_{m}.nc").x0_b.values - xt
            res[f"chi2_mean_{sp}"] = float(chi2(e, b).mean())
        out[m] = res
        print(f"{m}: std (x, y, z) {np.round(np.sqrt(np.diag(b)), 3)}  eig {np.round(res['eig'], 3)}  "
              f"floored {floored} (raw eig {np.round(w, 3)})  innov mean {np.round(d.mean(0), 3)} | chi2 mean "
              + ", ".join(f"{sp} {res[f'chi2_mean_{sp}']:.2f}" for sp in ("train", "val", "test", "test_ood"))
              + "  (calibrated ~3)")
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
