#!/bin/bash
# E2-B delta sweep v4 (anti-saturation; strong/full, M1, Adagrad 0.1, VAL).  Usage: bash ... <worker 0-3>
# Base C1: ALIGN 2 / TRANS 2, meta_lr 1e-2, output_scale 10, episode_iters 100, w_perf_delta 2.5, 3 epochs,
# best-epoch selection on val (median J after 10 frozen iterations, first 100 val ICs).
# Grid: w_align_perf {0, 0.05, 0.1, 0.2} x train_seed {0, 1, 2} = 12 runs (w_align_perf 0 = selection effect only).
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P=/Odyssey/private/j25lee/miniforge3/envs/oceanai/bin/python
mkdir -p tmp/logs
i=0
for W in 0 0.05 0.1 0.2; do for S in 0 1 2; do
  if [ $((i % 4)) -eq "$1" ]; then
    T=_v4_C1_wap${W}_s${S}
    $P src/run_meta_b.py result_group=e2b_sweep4 split=val delta.episode_iters=100 delta.w_perf_delta=2.5 \
      delta.grad_align_steps=2 delta.grad_trans_steps=2 delta.meta_lr=1e-2 net.output_scale=10 delta.n_epochs=3 \
      delta.select_epoch=true delta.w_align_perf=$W delta.train_seed=$S tag=$T > tmp/logs/e2b_sweep4$T.log 2>&1
  fi
  i=$((i+1))
done; done
