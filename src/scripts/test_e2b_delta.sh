#!/bin/bash
# E2-B delta FINAL test (locked config, pilot 2026-11-12): frozen inference of the 3 seed learners trained in sweep v4
# (C1: ALIGN 2 / TRANS 2, meta_lr 1e-2, output_scale 10, episode_iters 100, w_perf_delta 2.5, 3 epochs, best-epoch
# selection on val, w_align_perf 0) on test and test_ood. No training, no tuning on test.
# Usage: bash src/scripts/test_e2b_delta.sh <worker 0-3>
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P=/Odyssey/private/j25lee/miniforge3/envs/oceanai/bin/python
mkdir -p tmp/logs
i=0
for SPLIT in test test_ood; do for S in 0 1 2; do
  if [ $((i % 4)) -eq "$1" ]; then
    T=_C1_s${S}
    $P src/run_meta_b.py result_group=e2b split=$SPLIT tag=$T \
      checkpoint=tmp/meta/e2b_sweep4/delta_strong_full_adagrad_lr0.1_v4_C1_wap0_s${S}.pt \
      delta.episode_iters=100 delta.w_perf_delta=2.5 delta.grad_align_steps=2 delta.grad_trans_steps=2 \
      delta.meta_lr=1e-2 net.output_scale=10 delta.n_epochs=3 delta.select_epoch=true delta.w_align_perf=0 \
      delta.train_seed=$S > tmp/logs/e2b_test${T}_${SPLIT}.log 2>&1
  fi
  i=$((i+1))
done; done
