"""
Diagnostic (2026-11-18, pilot: "is the delta learner badly conditioned?"). From the cached teacher pairs of the dev
testbed (dense-obs weak F1, sigma_q 0.1/sqrt(10); 10 train / 20 val cases), compare the teacher step in RAW control
coordinates with the control-variable-transform coordinates (cvt: v0 = L^-1 (x0 - x0_b), B = L L^T; w = eta / sigma_q).

Per teacher (strong-Wolfe L-BFGS, Adagrad), split and coordinate system, per control block (x0 | eta):
    r = log10( RMS(Delta_b) / RMS(g_b) )      teacher step in gradient units (the quantity the log-magnitude head of
                                              the delta learner must predict); percentiles, spread p95 - p5, median per k
    R^2 of r explained by log10 RMS(g_b) (gradient magnitude), by log1p(k), and by both (linear least squares)
    cos(Delta, -g) of the whole control (direction the learner must predict relative to the gradient)

    python src/diag_cvt.py          (HPC only)  -> tmp/results/dev_delta/diag_cvt.json + printed table
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_meta_b as R  # noqa: E402
from dameta.meta import control_stats  # noqa: E402
from dameta.meta.meta_b_direct import DirectConfig, build_direct  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
CFG = OmegaConf.load("src/configs/meta_b.yaml")
CFG.merge_with_dotlist(["dataset=v2", "data_prefix=l63v2d", "model=f1", "constraint=weak", "pattern=full",
                        "cost.sigma_q=0.0316227766"])
PAIRS = Path("tmp/meta/dev_delta")
OUT = Path("tmp/results/dev_delta/diag_cvt.json")
N_CASES = {"train": 10, "val": 20}
K_SHOW = [0, 1, 2, 5, 10, 20, 40, 100, 300, 900]


def r2(y, cols):
    X = np.column_stack(cols + [np.ones_like(y)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ coef
    return float(1.0 - res.var() / y.var())


def stats(dx, g, k):
    """dx, g [P, d] in one coordinate system; k [P]."""
    out = {}
    for name, (a, b) in (("x0", (0, 3)), ("eta", (3, dx.shape[1]))):
        sd = dx[:, a:b].pow(2).mean(1).sqrt()
        sg = g[:, a:b].pow(2).mean(1).sqrt()
        ok = torch.isfinite(sd) & torch.isfinite(sg) & (sd > 0) & (sg > 0)
        r, lg, kk = (sd[ok] / sg[ok]).log10().numpy(), sg[ok].log10().numpy(), k[ok].numpy()
        q = np.percentile(r, [5, 50, 95])
        per_k = {int(kv): float(np.median(r[kk == kv])) for kv in K_SHOW if (kk == kv).any()}
        out[name] = {"p5": q[0], "p50": q[1], "p95": q[2], "spread_p95_p5": q[2] - q[0],
                     "std": float(r.std()), "R2_log_g": r2(r, [lg]), "R2_log1p_k": r2(r, [np.log1p(kk)]),
                     "R2_both": r2(r, [lg, np.log1p(kk)]), "median_by_k": per_k}
    cos = torch.nn.functional.cosine_similarity(dx, -g, dim=1)
    cos = cos[torch.isfinite(cos)].numpy()
    out["cos_step_neg_grad"] = {"p10": float(np.percentile(cos, 10)), "p50": float(np.median(cos)),
                                "p90": float(np.percentile(cos, 90))}
    return out


def main():
    tr = R.problem(CFG, "train", N_CASES["train"])
    cm, cs = control_stats(tr.x_true, tr.ccfg.n_obs, tr.ccfg.sigma_q, True)
    d = 3 + 3 * (tr.ccfg.n_obs - 1)
    lrn = build_direct(d, DirectConfig(coords="cvt"), cm, cs)
    res = {}
    for teacher in ("lbfgs_wolfe", "adagrad"):
        for split in ("train", "val"):
            f = PAIRS / f"pairs_{teacher}_{split}.pt"
            p = torch.load(f, weights_only=False)
            cost = R.problem(CFG, split, N_CASES[split]).cost()
            ctx = lrn.case_ctx(cost)
            rows = {"L": ctx["L"][p["case"]], "xb": ctx["xb"][p["case"]]}
            _, gv = lrn.to_learner(p["x"], p["g"], rows)
            dv = lrn.step_to_learner(p["target"], rows)
            key = f"{teacher}/{split}"
            res[key] = {"n_pairs": int(len(p["k"])), "raw": stats(p["target"], p["g"], p["k"]),
                        "cvt": stats(dv, gv, p["k"])}
            print(f"== {key}  ({len(p['k'])} pairs)")
            for coords in ("raw", "cvt"):
                s = res[key][coords]
                for blk in ("x0", "eta"):
                    b = s[blk]
                    print(f"  {coords:3s} {blk:3s}  r=log10|D|/|g|: p5 {b['p5']:+.2f} p50 {b['p50']:+.2f} "
                          f"p95 {b['p95']:+.2f} spread {b['spread_p95_p5']:.2f} dec | R2 log|g| {b['R2_log_g']:.2f} "
                          f"log1p(k) {b['R2_log1p_k']:.2f} both {b['R2_both']:.2f} | by k "
                          + " ".join(f"{kk}:{v:+.1f}" for kk, v in b["median_by_k"].items()))
                c = s["cos_step_neg_grad"]
                print(f"  {coords:3s} cos(D, -g): p10 {c['p10']:.2f} p50 {c['p50']:.2f} p90 {c['p90']:.2f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
