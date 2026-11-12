#!/bin/bash
# E2-B delta sweep (strong/full, M1, Adagrad teacher lr 0.1) on VAL. Usage: bash src/scripts/sweep_e2b_delta.sh <worker 0-3>
# Worker w runs every 4th configuration of episode_iters x w_perf_delta x output_scale (12 configs), single-threaded.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P=/Odyssey/private/j25lee/miniforge3/envs/oceanai/bin/python
mkdir -p tmp/logs
i=0
for E in 100 300; do for W in 1 2.5 5; do for O in 2 5; do
  if [ $((i % 4)) -eq "$1" ]; then
    $P src/run_meta_b.py result_group=e2b_sweep split=val delta.episode_iters=$E delta.w_perf_delta=$W \
      net.output_scale=$O tag=_ep${E}_wp${W}_os${O} > tmp/logs/e2b_sweep_ep${E}_wp${W}_os${O}.log 2>&1
  fi
  i=$((i+1))
done; done; done
