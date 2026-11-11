"""
Emulator training: multi-step rollout loss on the D1 training trajectories (pilot rule: same trajectories).

Data:  training ICs = D1 train x_true[:, 0]; teacher trajectories of length T = L + F (601 states) from those ICs
       (exact L63 for the clean emulator = D1 x_true itself; degraded teacher otherwise). Validation uses the D1 val
       ICs with the same teacher. Test / test_ood are never touched here.
Loss:  random windows (ic, t0) of n_roll steps; emulator rolled out from the teacher state at t0;
       MSE of standardized states averaged over the n_roll steps. Curriculum: n_roll grows linearly from 1 to
       n_roll_max over the first half of training (long rollouts early are unstable).
Optim: Adam, cosine learning-rate decay, gradient clipping. Best validation checkpoint kept.
"""

import math
import time
from dataclasses import dataclass

import torch

from .emulator import EmulatorConfig, L63Emulator, TeacherConfig, teacher_rollout


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 4000
    batch: int = 512
    n_roll_max: int = 10
    lr: float = 1e-3
    lr_min: float = 1e-5
    clip: float = 1.0
    val_every: int = 200
    val_roll: int = 50      # validation: rollout RMSE over 50 steps (0.5 TU) from val states
    seed: int = 0


def rollout_loss(emu: L63Emulator, traj: torch.Tensor, n_roll: int, idx_ic, idx_t) -> torch.Tensor:
    """Mean standardized squared error of an n_roll-step emulator rollout started on the teacher trajectory."""
    x = traj[idx_ic, idx_t]
    loss = 0.0
    for k in range(1, n_roll + 1):
        x = emu.step(x)
        loss = loss + (((x - traj[idx_ic, idx_t + k]) / emu.x_std) ** 2).mean()
    return loss / n_roll


@torch.no_grad()
def val_rmse(emu: L63Emulator, traj: torch.Tensor, n: int) -> float:
    """RMSE (physical units) of n-step emulator rollouts from every 50th state of the val trajectories."""
    starts = torch.arange(0, traj.shape[1] - n, 50)
    x0 = traj[:, starts].reshape(-1, 3)
    target = torch.stack([traj[:, s:s + n + 1] for s in starts], 1).reshape(-1, n + 1, 3)
    return float(((emu.rollout(x0, n) - target) ** 2).mean().sqrt())


def train_emulator(x0_train: torch.Tensor, x0_val: torch.Tensor, n_steps_traj: int, teacher: TeacherConfig,
                   ecfg: EmulatorConfig, tcfg: TrainConfig, log=print):
    torch.manual_seed(tcfg.seed)
    traj_tr = teacher_rollout(x0_train, n_steps_traj, teacher)     # [n_ic, T + 1, 3]
    traj_va = teacher_rollout(x0_val, n_steps_traj, teacher)
    emu = L63Emulator(ecfg)
    emu.set_normalization(traj_tr)
    opt = torch.optim.Adam(emu.parameters(), lr=tcfg.lr)
    gen = torch.Generator().manual_seed(tcfg.seed)
    best, best_state, hist, t0 = math.inf, None, [], time.perf_counter()
    for it in range(1, tcfg.steps + 1):
        frac = min(1.0, it / (0.5 * tcfg.steps))
        n_roll = max(1, round(frac * tcfg.n_roll_max))                # curriculum on the rollout length
        lr = tcfg.lr_min + 0.5 * (tcfg.lr - tcfg.lr_min) * (1 + math.cos(math.pi * it / tcfg.steps))
        for g in opt.param_groups:
            g["lr"] = lr
        i_ic = torch.randint(0, traj_tr.shape[0], (tcfg.batch,), generator=gen)
        i_t = torch.randint(0, traj_tr.shape[1] - n_roll, (tcfg.batch,), generator=gen)
        loss = rollout_loss(emu, traj_tr, n_roll, i_ic, i_t)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(emu.parameters(), tcfg.clip)
        opt.step()
        if it % tcfg.val_every == 0 or it == tcfg.steps:
            v = val_rmse(emu, traj_va, tcfg.val_roll)
            hist.append({"step": it, "loss": float(loss), "n_roll": n_roll, "lr": lr, "val_rmse": v})
            log(f"step {it:5d}  n_roll {n_roll:2d}  loss {float(loss):.3e}  val_rmse@{tcfg.val_roll} {v:.4f}  "
                f"({time.perf_counter() - t0:.0f}s)")
            if v < best and n_roll == tcfg.n_roll_max:                # keep best once the curriculum is complete
                best, best_state = v, {k: t.clone() for k, t in emu.state_dict().items()}
    if best_state is not None:
        emu.load_state_dict(best_state)
    record = {"history": hist, "best_val_rmse": best, "train_cfg": tcfg.__dict__, "wall_s": time.perf_counter() - t0}
    return emu.eval(), record
