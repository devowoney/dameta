"""
M8 command lists (branch m8-classic-4dv): classical 4D-Var on the model rectangle (F1, F2, E1, E2).

    python src/scripts/m8_lists.py smoke                 -> src/scripts/lists/m8_smoke.txt
    python src/scripts/m8_lists.py sweep  [n_jobs]       -> src/scripts/lists/m8_sweep_<j>.txt   (val, lr grid)
    python src/scripts/m8_lists.py test   [n_jobs]       -> src/scripts/lists/m8_test_<j>.txt    (test, selected lr)
Lines run from the project root with src/scripts/sb_parallel.sh; logs in tmp/logs/m8/.
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
                            lines.append(line(m, c, p, meth, lr, "val", "tmp/results/m8/sweep", "sweep"))
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
    else:
        raise ValueError(mode)


if __name__ == "__main__":
    main()
