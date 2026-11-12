#!/bin/bash
# E2-B delta sweep v2, stage A (gate-gap screen; strong/full, M1, Adagrad 0.1, VAL, 1 seed).
# Usage: bash src/scripts/sweep_e2b_delta_v2.sh <worker 0-3>
# Grid: ALIGN regime {a0: align 0 trans 0 (pure PERF), a2: align 2 trans 2, a6: bice default 6/2}
#       x meta_lr {1e-3, 1e-2} x output_scale {5, 10};  fixed: episode_iters 100, w_perf_delta 2.5, 1 epoch.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P=/Odyssey/private/j25lee/miniforge3/envs/oceanai/bin/python
mkdir -p tmp/logs
i=0
for A in a0 a2 a6; do
  case $A in a0) AL=0; TR=0;; a2) AL=2; TR=2;; a6) AL=6; TR=2;; esac
  for LR in 1e-3 1e-2; do for O in 5 10; do
    if [ $((i % 4)) -eq "$1" ]; then
      T=_v2_${A}_mlr${LR}_os${O}
      $P src/run_meta_b.py result_group=e2b_sweep2 split=val delta.episode_iters=100 delta.w_perf_delta=2.5 \
        delta.grad_align_steps=$AL delta.grad_trans_steps=$TR delta.meta_lr=$LR net.output_scale=$O tag=$T \
        > tmp/logs/e2b_sweep2$T.log 2>&1
    fi
    i=$((i+1))
  done; done
done
