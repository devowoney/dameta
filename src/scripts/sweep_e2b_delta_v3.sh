#!/bin/bash
# E2-B delta sweep v2, stage B (seed replicates; strong/full, M1, Adagrad 0.1, VAL).
# Usage: bash src/scripts/sweep_e2b_delta_v3.sh <worker 0-3>
# Configs (all ALIGN 2 / TRANS 2, episode_iters 100, w_perf_delta 2.5):
#   C1 meta_lr 1e-2 os10 (stage-A best), C2 meta_lr 1e-3 os10 (runner-up), C3 meta_lr 1e-2 os20 (bound extension)
#   x n_epochs {1, 3} x train_seed {0, 1, 2}  = 18 runs. Long (3-epoch) runs are dealt first for load balance.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P=/Odyssey/private/j25lee/miniforge3/envs/oceanai/bin/python
mkdir -p tmp/logs
i=0
for EP in 3 1; do for C in C1 C2 C3; do for S in 0 1 2; do
  case $C in C1) LR=1e-2; O=10;; C2) LR=1e-3; O=10;; C3) LR=1e-2; O=20;; esac
  if [ $((i % 4)) -eq "$1" ]; then
    T=_v3_${C}_mlr${LR}_os${O}_ep${EP}_s${S}
    $P src/run_meta_b.py result_group=e2b_sweep3 split=val delta.episode_iters=100 delta.w_perf_delta=2.5 \
      delta.grad_align_steps=2 delta.grad_trans_steps=2 delta.meta_lr=$LR net.output_scale=$O \
      delta.n_epochs=$EP delta.train_seed=$S tag=$T > tmp/logs/e2b_sweep3$T.log 2>&1
  fi
  i=$((i+1))
done; done; done
