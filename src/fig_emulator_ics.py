"""
Emulator performance against its OWN teacher over many initial conditions (pilot 2026-11-27: E1 / E2 were trained on
only 10 ICs; are trajectories from other ICs worse?). Good emulators E1 (= E1g) / E2 (= E2g) and bad emulators
E1b / E2b (3 train ICs, MLP 2 x 4).

    (traj) x(t) teacher vs emulator from the same IC: the first training ICs and test ICs (never seen in training)
    (err)  |emulator - teacher|(t) from the same IC, median and 25-75 %: the training ICs vs the test ICs
    (map)  one-obs-interval error |emulator - teacher| from true states along the test trajectories, on the x-z plane,
           with the emulator's training trajectories (teacher, first n_train_ic train ICs, 600 steps) in grey

Usage (project root):  python src/fig_emulator_ics.py [--pairs f1:e1 f1:e1b f2:e2 f2:e2b] [--out-dir tmp/figs]
    -> <out-dir>/m8emu_{traj_<e>,err,map}.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dameta.dynamics.models import build_model, load_registry  # noqa: E402

torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parent.parent
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
COL = {"f1": "#5598e7", "f2": "#184f95", "e1": "#eb6834", "e2": "#a8431a", "e1b": "#d6a21e", "e2b": "#7a5a10"}
WINDOW_TU = 1.6


@torch.no_grad()
def run(step, x, n):
    out = [x]
    for _ in range(n):
        x = step(x)
        out.append(x)
    return torch.stack(out, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=["f1:e1", "f1:e1b", "f2:e2", "f2:e2b"])
    ap.add_argument("--train-data", default="/Odyssey/private/j25lee/dameta/tmp/data/l63_train_seed42.nc",
                    help="D1 v1 train set the emulators were trained on (first n_train_ic ICs)")
    ap.add_argument("--out-dir", default="tmp/figs")
    ap.add_argument("--tag", default="", help="suffix of the output file names")
    ap.add_argument("--t-traj", type=float, default=4.0)
    ap.add_argument("--t-err", type=float, default=6.0)
    ap.add_argument("--n-test-traj", type=int, default=6)
    a = ap.parse_args()
    reg = load_registry()
    te = xr.load_dataset(ROOT / "tmp/data/l63v2_test_seed42.nc")
    dt, sigma_o, obs_every = float(te.attrs["dt"]), float(np.sqrt(te.attrs["sigma_o2"])), int(te.attrs["obs_every"])
    x_test = torch.tensor(te.x_true.values)                                   # (200, 661, 3) L1 truth
    tr_all = xr.load_dataset(a.train_data)
    n_traj, n_err = int(round(a.t_traj / dt)), int(round(a.t_err / dt))

    res = {}
    for pair in a.pairs:
        t_name, e_name = pair.split(":")
        if "=" in e_name:                                   # e.g. f1:e1b=e1_d3w4d1 -> registry model e1b, other file
            e_name, f = e_name.split("=")
            reg.models[e_name].file = f
        spec = reg.models[e_name]
        ck = torch.load(ROOT / f"tmp/emulators/{spec.file}.pt", weights_only=False)
        cfg = ck["record"]["config"]
        n_tr = int(cfg.get("n_train_ic") or tr_all.sizes["ic"])
        n_tr_steps = tr_all.sizes["t"] - 1
        t_step = build_model(t_name, reg)[0]
        e_step = build_model(e_name, reg, str(ROOT / "tmp/emulators"))[0]
        x_tr0 = torch.tensor(tr_all.x_true.values[:n_tr, 0])
        tr_traj = run(t_step, x_tr0, n_tr_steps)                              # emulator training trajectories
        x_te0 = x_test[:, 0]
        r = {"n_tr": n_tr, "file": spec.file, "w": e_step.__self__.cfg.width, "d": e_step.__self__.cfg.depth,
             "train_traj": tr_traj.numpy()}
        for split, x0 in (("train", x_tr0), ("test", x_te0)):
            T, E = run(t_step, x0, n_err), run(e_step, x0, n_err)
            r[split] = {"teacher": T.numpy(), "emu": E.numpy(), "err": (E - T).norm(dim=-1).numpy()}
        starts = x_test[:, : x_test.shape[1] - obs_every: obs_every].reshape(-1, 3)
        T, E = run(t_step, starts, obs_every)[:, -1], run(e_step, starts, obs_every)[:, -1]
        r["map"] = {"x": starts.numpy(), "err": (E - T).norm(dim=-1).numpy() / sigma_o}
        long = run(e_step, x_te0[:50], int(round(20.0 / dt)))
        r["stable"] = bool(torch.isfinite(long).all() and long.abs().max() < 100)
        res[pair] = r
        print(f"{e_name} ({spec.file}, {n_tr} train ICs, MLP {r['d']}x{r['w']}): stable 20 TU (50 test ICs) "
              f"{r['stable']}", flush=True)
        for split in ("train", "test"):
            e = r[split]["err"]
            k = int(round(WINDOW_TU / dt))
            print(f"{e_name} vs {t_name} [{split}, n={e.shape[0]}]  |err| at {WINDOW_TU} TU: "
                  f"median {np.median(e[:, k]):.3f}, mean {e[:, k].mean():.3f} ± {e[:, k].std():.3f}", flush=True)
        print(f"{e_name} one-interval |err|/sigma_o on {starts.shape[0]} test states: median "
              f"{np.median(r['map']['err']):.3f}, 95 % {np.percentile(r['map']['err'], 95):.3f}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.linewidth": 0.6, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "lines.linewidth": 1.1, "mathtext.default": "regular"})
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # (traj) one figure per emulator: 2 training ICs + n test ICs, x / y / z rows
    for pair, r in res.items():
        t_name, e_name = pair.split("=")[0].split(":")
        cols = [("train", i) for i in range(min(2, r["n_tr"]))] + [("test", i) for i in range(a.n_test_traj)]
        t = np.arange(n_traj + 1) * dt
        fig, axs = plt.subplots(3, len(cols), figsize=(1.6 * len(cols), 4.6), sharex=True, sharey="row")
        for j, (split, i) in enumerate(cols):
            for v in range(3):
                ax = axs[v, j]
                ax.axvspan(0, WINDOW_TU, color=GRID, alpha=0.7, lw=0, zorder=0)
                ax.plot(t, r[split]["teacher"][i, : n_traj + 1, v], color=COL[t_name], label=t_name.upper())
                ax.plot(t, r[split]["emu"][i, : n_traj + 1, v], color=COL[e_name], ls="--",
                        label=f"{e_name.upper()} ({r['file']})")
                if j == 0:
                    ax.set_ylabel("xyz"[v])
            axs[0, j].set_title(f"{'TRAIN' if split == 'train' else 'test'} IC {i}", fontsize=7.5,
                                color=INK if split == "train" else MUTED)
            axs[2, j].set_xlabel("time [TU]")
        axs[0, 0].legend(loc="upper left", bbox_to_anchor=(0, 1.55), ncol=2, fontsize=7)
        fig.suptitle(f"{e_name.upper()} vs its teacher {t_name.upper()} from the same IC  (grey band: 1.6-TU DA "
                     f"window; training ICs = the {r['n_tr']} ICs it was trained on)", x=0.01, ha="left", y=1.04,
                     fontsize=8)
        fig.savefig(out / f"m8emu_traj_{e_name}{a.tag}.png", dpi=200)
        plt.close(fig)

    # (err) error growth, training ICs vs test ICs
    nc = 2 if len(res) > 2 else len(res)
    nr = -(-len(res) // nc)
    fig, axs = plt.subplots(nr, nc, figsize=(3.7 * nc, 2.6 * nr), sharey=True, squeeze=False,
                            gridspec_kw={"hspace": 0.45})
    t = np.arange(n_err + 1) * dt
    for ax, (pair, r) in zip(axs.ravel(), res.items()):
        t_name, e_name = pair.split("=")[0].split(":")
        ax.axvspan(0, WINDOW_TU, color=GRID, alpha=0.7, lw=0, zorder=0)
        for split, c in (("train", COL[t_name]), ("test", COL[e_name])):
            q = np.percentile(r[split]["err"], [25, 50, 75], 0)
            ax.fill_between(t, np.maximum(q[0], 1e-6), q[2], color=c, alpha=0.18, lw=0)
            ax.semilogy(t, q[1], color=c, label=f"{split} ICs (n = {r[split]['err'].shape[0]})")
        ax.axhline(sigma_o, color=MUTED, lw=0.7, ls=":")
        ax.text(a.t_err, sigma_o * 1.1, "σ_o", color=MUTED, fontsize=6.5, ha="right")
        ax.set(xlabel="time [TU]", xlim=(0, a.t_err), ylim=(1e-2, None),
               title=f"{e_name.upper()} ({r['file']}, {r['n_tr']} train ICs, MLP {r['d']}×{r['w']}) "
                     f"vs {t_name.upper()}")
        ax.title.set_fontsize(7.5)
        ax.legend(loc="lower right", fontsize=6.5)
    for ax in axs[:, 0]:
        ax.set_ylabel("|emulator − teacher|")
    fig.text(0.01, -0.04, "median and 25–75 % over ICs; same IC for emulator and teacher; grey band: 1.6-TU DA window",
             fontsize=6.5, color=MUTED)
    fig.savefig(out / f"m8emu_err{a.tag}.png", dpi=200)
    plt.close(fig)

    # (map) where on the attractor the one-interval error is large
    fig, axs = plt.subplots(nr, nc, figsize=(3.7 * nc, 3.2 * nr), sharex=True, sharey=True, squeeze=False)
    vmax = float(np.percentile(np.concatenate([r["map"]["err"] for r in res.values()]), 99))
    for ax, (pair, r) in zip(axs.ravel(), res.items()):
        t_name, e_name = pair.split("=")[0].split(":")
        for tr in r["train_traj"]:
            ax.plot(tr[:, 0], tr[:, 2], color=MUTED, lw=0.35, alpha=0.6, zorder=1)
        m = r["map"]
        o = np.argsort(m["err"])
        sc = ax.scatter(m["x"][o, 0], m["x"][o, 2], c=m["err"][o], s=3, cmap="viridis", vmin=0, vmax=vmax, lw=0,
                        zorder=2)
        ax.set(title=f"{e_name.upper()} ({r['file']}, {r['n_tr']} train ICs, MLP {r['d']}×{r['w']}) vs "
                     f"{t_name.upper()}\none-interval error: median {np.median(m['err']):.2f}, "
                     f"95 % {np.percentile(m['err'], 95):.2f} σ_o", aspect="equal")
        ax.title.set_fontsize(7.5)
        ax.grid(False)
    for ax in axs[:, 0]:
        ax.set_ylabel("z")
    for ax in axs[-1]:
        ax.set_xlabel("x")
    cb = fig.colorbar(sc, ax=axs.ravel().tolist(), shrink=0.8, pad=0.02, extend="max")
    cb.set_label("|emulator − teacher| / σ_o")
    fig.text(0.01, -0.02, f"points: start states on the {x_test.shape[0]} test truth trajectories (every {obs_every} "
             f"steps); grey: the emulator's training trajectories (its training ICs × {tr_all.sizes['t'] - 1} steps); "
             f"colour capped at the pooled 99th percentile", fontsize=6.5, color=MUTED)
    fig.savefig(out / f"m8emu_map{a.tag}.png", dpi=200)
    plt.close(fig)
    print("saved", *sorted(str(p) for p in out.glob("m8emu_*.png")))


if __name__ == "__main__":
    main()
