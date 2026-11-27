"""
M8 command lists (branch m8-classic-4dv): classical 4D-Var on the model rectangle (F1, F2, E1, E2).

    python src/scripts/m8_lists.py smoke                 -> src/scripts/lists/m8_smoke.txt
    python src/scripts/m8_lists.py sweep  [n_jobs] [n_cases]  -> src/scripts/lists/m8_sweep_<j>.txt   (val, lr grid;
                                                            n_cases: first n val cases, 2026-11-26 pilot: 10)
    python src/scripts/m8_lists.py test   [n_jobs]       -> src/scripts/lists/m8_test_<j>.txt    (test, selected lr)
    python src/scripts/m8_lists.py binfl  [n_jobs]       -> src/scripts/lists/m8_binfl_<j>.txt   (val smoke, L-BFGS,
                                                            cost.b_infl in BINFL: choice of the weaker background)
    python src/scripts/m8_lists.py qtune  [n_jobs]       -> m8r_qtune_<j>.txt  (M8-R: chi2 tuning of the scalar Q,
                                                            weak L-BFGS, 50 TRAIN cases, full obs, q grid Q_GRID)
    python src/scripts/m8_lists.py rect_smoke [n_jobs]   -> m8r_smoke_<j>.txt  (M8-R rectangle, 10 val cases)
    python src/scripts/m8_lists.py rect   [n_jobs]       -> m8r_test_<j>.txt   (M8-R rectangle, 200 test cases)
Lines run from the project root with src/scripts/sb_parallel.sh; logs in tmp/logs/m8/.

M8-R (pilot 2026-11-26): per truth (STAGES) one SHARED first guess / static B = the truth model's lagged forecast
(background=<truth>), Q = q_m I with q_m chi2-tuned per (truth, model) (src/select_q.py -> <data>/q_chi2_seed42.json),
perfect-model reference strong only, current lr (tmp/results/m8/selected_lr.json; reference = its F model's lr).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LISTS = ROOT / "src/scripts/lists"
MODELS = ("f1", "f2", "e1", "e2")
CONSTRAINTS = ("strong", "weak")
PATTERNS = ("full", "x_only")
GRIDS = {"gd": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1], "adagrad": [0.03, 0.1, 0.3, 1.0, 3.0],
         "adam": [0.03, 0.1, 0.3, 1.0, 3.0]}
PY = "python src/run_4dvar_v2.py"
BINFL = (1.0, 4.0, 16.0)
WINDOWS = {60: "tmp/data_w60", 100: "tmp/data_w100", 160: "tmp/data"}    # feasibility of longer windows (2026-11-25);
                                                                           # 1.6 TU = tmp/data since 2026-11-26
# M8-R stages: truth tag -> (data dir, shared background = truth model, reference, imperfect models)
STAGES = {"L1": ("tmp/data", "l1", "l1", ("f1", "f2", "e1", "e2")),
          "F1": ("tmp/data_tF1", "f1", "f1", ("e1",)),
          "F2": ("tmp/data_tF2", "f2", "f2", ("e2",))}
Q_GRID = [round(10 ** (e / 4), 6) for e in range(-12, 9)]                 # 1e-3 .. 1e2, 0.25 decade (21 values)
Q_EXTRA = {"L1": [round(10 ** (e / 4), 6) for e in range(9, 13)]}          # edge extension to 1e3 (pilot 2026-11-26:
                                                                          # F2 picked q = 100 on the grid edge)
METHODS = ("gd", "adagrad", "adam", "lbfgs_wolfe")
FEAS_LR = {"gd": 0.01, "adagrad": 0.3, "adam": 0.3, "lbfgs_wolfe": 1.0}  # mid-grid lr: runs only, no tuning


def line(model, constraint, pattern, method, lr, split, out, tag, extra=""):
    log = f"tmp/logs/m8/{tag}_{model}_{constraint}_{pattern}_{method}_lr{lr:g}.log"
    return (f"{PY} model={model} constraint={constraint} pattern={pattern} method={method} lr={lr:g} split={split} "
            f"out_dir={out}/{model} {extra}".rstrip() + f" > {log} 2>&1")


def write(lines, name, n_jobs):
    LISTS.mkdir(parents=True, exist_ok=True)
    chunks = [lines[j::n_jobs] for j in range(n_jobs)]
    for j, ch in enumerate(chunks):
        (LISTS / f"{name}_{j}.txt").write_text("\n".join(ch) + "\n")
    print(f"{len(lines)} lines -> {n_jobs} x {name}_<j>.txt")


def main():
    mode = sys.argv[1]
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    n_cases = int(sys.argv[3]) if len(sys.argv) > 3 else None
    if mode == "smoke":
        lines = ["python src/tests/test_landscape.py > tmp/logs/m8/test_landscape.log 2>&1",
                 "python src/tests/test_models.py > tmp/logs/m8/test_models.log 2>&1"]
        for m in MODELS:
            for c in CONSTRAINTS:
                for meth, lr in (("gd", 0.01), ("lbfgs_wolfe", 1.0)):
                    lines.append(line(m, c, "full", meth, lr, "test", "tmp/results/m8/smoke", "smoke",
                                      "n_cases=5 max_iter=20 landscape_iter=true"))
        write(lines, "m8_smoke", 1)
    elif mode in ("sweep", "sweep_extra"):          # sweep_extra: only the edge-extension lrs (m8_grid_extra.json)
        only_extra = mode == "sweep_extra"
        extra_grid = json.loads((LISTS / "m8_grid_extra.json").read_text()) if (LISTS / "m8_grid_extra.json").exists() \
            else {}
        lines = []
        for m in MODELS:
            for c in CONSTRAINTS:
                for p in PATTERNS:
                    for meth, grid in GRIDS.items():
                        base = [] if only_extra else grid
                        for lr in base + extra_grid.get(f"{m}/{c}/{p}/{meth}", []):
                            lines.append(line(m, c, p, meth, lr, "val", "tmp/results/m8/sweep", "sweep",
                                              f"n_cases={n_cases}" if n_cases else ""))
        write(lines, "m8_sweep_extra" if only_extra else "m8_sweep", n_jobs)
    elif mode == "test":
        sel = json.loads((ROOT / "tmp/results/m8/selected_lr.json").read_text())
        lines = []
        for m in MODELS:
            for c in CONSTRAINTS:
                for p in PATTERNS:
                    for meth in list(GRIDS) + ["lbfgs_wolfe"]:
                        lr = 1.0 if meth == "lbfgs_wolfe" else float(sel[f"{m}/{c}/{p}/{meth}"])
                        lines.append(line(m, c, p, meth, lr, "test", "tmp/results/m8/test", "test",
                                          "landscape_iter=true"))
        write(lines, "m8_test", n_jobs)
    elif mode == "feas":                            # 10-case feasibility smoke, b_infl 4 (config default)
        lines = []
        for w, ddir in WINDOWS.items():
            for m in MODELS:
                for c in CONSTRAINTS:
                    for p in PATTERNS:
                        for meth, lr in FEAS_LR.items():
                            lines.append(line(m, c, p, meth, lr, "val", f"tmp/results/m8/feas/w{w}",
                                              f"feas_w{w}", f"data_dir=${{PWD}}/{ddir} n_cases=10 "
                                              "landscape_iter=true"))
        write(lines, "m8_feas", n_jobs)
    elif mode == "feasq":                           # 1.6 TU, per-model diagonal Q (cost.q=model): weak vs strong
        lines = []
        for m in MODELS:
            for c in CONSTRAINTS:
                for p in PATTERNS:
                    for meth, lr in FEAS_LR.items():
                        lines.append(line(m, c, p, meth, lr, "val", "tmp/results/m8/feasq", "feasq",
                                          "n_cases=10 landscape_iter=true hess_every=5"))
        write(lines, "m8_feasq", n_jobs)
    elif mode == "binfl":
        lines = []
        for infl in BINFL:
            for m in MODELS:
                for c in CONSTRAINTS:
                    for p in PATTERNS:
                        lines.append(line(m, c, p, "lbfgs_wolfe", 1.0, "val", f"tmp/results/m8/binfl/b{infl:g}",
                                          f"binfl{infl:g}", f"cost.b_infl={infl:g} landscape_iter=true"))
        write(lines, "m8_binfl", n_jobs)
    elif mode in ("qtune", "qtune_extra"):
        lines = []
        for tag, (ddir, bg, _, models) in STAGES.items():
            for m in models:
                for q in (Q_EXTRA.get(tag, []) if mode == "qtune_extra" else Q_GRID):
                    lines.append(line(m, "weak", "full", "lbfgs_wolfe", 1.0, "train",
                                      f"tmp/results/m8r/qtune/{tag}/q{q:g}", f"m8r_qtune_{tag}_q{q:g}",
                                      f"data_dir=${{PWD}}/{ddir} background={bg} background_cov=static cost.q=null "
                                      f"cost.sigma_q={q ** 0.5:.10g} n_cases=50"))
        write(lines, f"m8r_{mode}", n_jobs)
    elif mode in ("rect", "rect_smoke"):
        smoke = mode == "rect_smoke"
        sel = json.loads((ROOT / "tmp/results/m8/selected_lr.json").read_text())
        sub, split, cases = ("smoke", "val", "n_cases=10 ") if smoke else ("test", "test", "")
        lines = []
        for tag, (ddir, bg, ref, models) in STAGES.items():
            qj = json.loads((ROOT / ddir / "q_chi2_seed42.json").read_text())
            for m in (ref,) + models:
                lr_model = "f1" if m == "l1" else m                        # reference L1: F1's lr (no new sweep)
                for c in (("strong",) if m == ref else CONSTRAINTS):       # perfect-model reference: strong only
                    qx = f" cost.sigma_q={qj[m]['q'] ** 0.5:.10g}" if c == "weak" else ""
                    for p in PATTERNS:
                        for meth in METHODS:
                            lr = 1.0 if meth == "lbfgs_wolfe" else float(sel[f"{lr_model}/{c}/{p}/{meth}"])
                            lines.append(line(m, c, p, meth, lr, split, f"tmp/results/m8r/{sub}/{tag}",
                                              f"m8r_{sub}_{tag}",
                                              f"data_dir=${{PWD}}/{ddir} background={bg} background_cov=static "
                                              f"cost.q=null{qx} {cases}landscape_iter=true"))
        write(lines, f"m8r_{'smoke' if smoke else 'test'}", n_jobs)
    else:
        raise ValueError(mode)


if __name__ == "__main__":
    main()
