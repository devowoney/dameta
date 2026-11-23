"""
Truth-free tuning of the hybrid background covariance (pilot 2026-11-14):

    B_h = alpha * B_ens + (1 - alpha) * s2 * I        (B_ens = time-lagged false-model ensemble covariance)

Innovation consistency at t0: the full-state observation y_0 = x_true(t0) + eps (R = sigma_o2 I) gives the
innovation d = y_0 - x0_b ~ N(0, B + R). (alpha, s2) maximise the Gaussian likelihood of the VAL innovations
(grid search; truth not used). Diagnostic afterwards (uses the truth): chi2 = e^T B_h^-1 e with e = x0_b - x_true,
calibrated mean ~ 3, before and after.

Usage (project root):  python src/tune_hybrid_b.py            -> tmp/data/hybrid_B_seed42.json (+ printed table)
"""

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "tmp/data"
ALPHAS = np.round(np.arange(0.0, 1.0001, 0.05), 2)
S2 = np.exp(np.linspace(np.log(1e-3), np.log(10.0), 81))


def hybrid(b_ens, alpha, s2):
    return alpha * b_ens + (1 - alpha) * s2 * np.eye(3)


def nll(d, c):
    sign, logdet = np.linalg.slogdet(c)
    return 0.5 * (np.einsum("ni,nij,nj->n", d, np.linalg.inv(c), d) + logdet).mean()


def chi2(e, b):
    return np.einsum("ni,nij,nj->n", e, np.linalg.inv(b), e)


def main(names=("f1", "f2", "e1", "e2"), seed=42):
    out = {}
    for name in names:
        ds = xr.load_dataset(D / f"l63v2_val_seed{seed}.nc")
        bg = xr.load_dataset(D / f"l63v2_val_seed{seed}_bg_{name}.nc")
        r = float(ds.attrs["sigma_o2"]) * np.eye(3)
        d = ds.y_obs.values[:, 0] - bg.x0_b.values                  # t0 innovation (full obs), truth-free
        b_ens = bg.B.values
        best = min(((nll(d, hybrid(b_ens, a, s) + r), a, s) for a in ALPHAS for s in S2))
        _, a, s = best
        res = {"alpha": float(a), "s2": float(s), "nll": float(best[0])}
        for split in ("val", "test", "test_ood"):
            t = xr.load_dataset(D / f"l63v2_{split}_seed{seed}.nc").x_true.values[:, 0]
            bgs = xr.load_dataset(D / f"l63v2_{split}_seed{seed}_bg_{name}.nc")
            e = bgs.x0_b.values - t
            res[f"chi2_mean_{split}_ens"] = float(chi2(e, bgs.B.values).mean())
            res[f"chi2_mean_{split}_hybrid"] = float(chi2(e, hybrid(bgs.B.values, a, s)).mean())
        out[name] = res
        print(f"{name}: alpha = {a:.2f}, s2 = {s:.3f} (s = {np.sqrt(s):.3f}) | chi2 mean ensemble -> hybrid: "
              + ", ".join(f"{sp} {res[f'chi2_mean_{sp}_ens']:.1f} -> {res[f'chi2_mean_{sp}_hybrid']:.1f}"
                          for sp in ("val", "test", "test_ood")) + "  (calibrated ~3)")
    (D / f"hybrid_B_seed{seed}.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {D / f'hybrid_B_seed{seed}.json'}")


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("f1", "f2", "e1", "e2"))   # model rectangle (m7); legacy: f1 rho275
