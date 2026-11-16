#!/bin/bash
# Speed-retuning sweep (2026-11-16, step 2 of the meta-speed check): train one cell-1 / cell-5 variant on D1 v2 train
# ICs, frozen VAL record (selection), optional frozen test / test_ood from the checkpoint.
# Usage: bash src/scripts/speed_sweep_job.sh <model f1|emulator> <name> <teacher lr> <seed> "<overrides>" [splits]
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P=/Odyssey/private/j25lee/miniforge3/envs/oceanai/bin/python
M=$1; N=$2; LR=$3; S=$4; X=$5; SPLITS=${6:-""}
EMU=""; [ "$M" = emulator ] && EMU="emulator=rho275"
G=m5spd_$M; T=_${N}_s$S
COMMON="dataset=v2 model=$M $EMU constraint=strong pattern=full method=adagrad lr=$LR result_group=$G tag=$T $X"
VAR=$(echo "$X" | sed 's/.*variant=\([a-z_]*\).*/\1/')
CKPT=tmp/meta/$G/${VAR}_strong_full_adagrad_lr${LR}${T}.pt
mkdir -p tmp/logs
[ -f $CKPT ] || $P src/run_meta_b.py $COMMON split=val > tmp/logs/${G}${T}_val.log 2>&1 || exit 1
for SP in $SPLITS; do
  $P src/run_meta_b.py $COMMON split=$SP checkpoint=$CKPT > tmp/logs/${G}${T}_${SP}.log 2>&1
done
