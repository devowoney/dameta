#!/bin/bash
# Meta-learners on the DENSE-obs WEAK-constraint problem (2026-11-16): D1 v2-dense (obs every step, 111 obs over the
# 1.1-TU window), weak constraint with eta at every step (control 333), sigma_q = 0.1 / sqrt(10), false model F1.
# Teacher / Adagrad lr 0.03 = classical val selection for this configuration (tmp/results/dense/timing_*_q0.0316228.json).
# Train on train ICs (best epoch / selection on val) -> val record + checkpoint, then frozen test / test_ood.
# Usage: bash src/scripts/dense_meta_job.sh <cell c1|c5|c5k50|c8a> <seed> [splits] [extra overrides]
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P=/Odyssey/private/j25lee/miniforge3/envs/oceanai/bin/python
C=$1; S=$2; SPLITS=${3:-"test test_ood"}; EXTRA=$4
case $C in
  c1)    X="variant=delta delta.episode_iters=100 delta.w_perf_delta=2.5 delta.grad_align_steps=2 \
delta.grad_trans_steps=2 delta.meta_lr=1e-2 net.output_scale=10 delta.n_epochs=3 delta.select_epoch=true \
delta.train_seed=$S";;
  c1k50)    X="variant=delta delta.episode_iters=100 delta.w_perf_delta=2.5 delta.grad_align_steps=2 \
delta.grad_trans_steps=2 delta.meta_lr=1e-2 net.output_scale=10 delta.n_epochs=3 delta.select_epoch=true \
delta.train_seed=$S delta.K=50";;
  c1k100)    X="variant=delta delta.episode_iters=100 delta.w_perf_delta=2.5 delta.grad_align_steps=2 \
delta.grad_trans_steps=2 delta.meta_lr=1e-2 net.output_scale=10 delta.n_epochs=3 delta.select_epoch=true \
delta.train_seed=$S delta.K=100";;
  c5)    X="variant=unroll unroll.loss=sum unroll.K=20 unroll.seed=$S";;
  c5k50) X="variant=unroll unroll.loss=sum unroll.K=50 unroll.seed=$S";;
  c8a)   X="variant=learned_grad learned_grad.arch=adjoint learned_grad.data=sampled learned_grad.loss=jacobian \
learned_grad.traj_stride=0 learned_grad.samples_per_case=8 learned_grad.seed=$S";;
  *) echo "unknown cell $C"; exit 1;;
esac
# teacher: METHOD / LR env overrides (default Adagrad 0.03; L-BFGS teacher: METHOD=lbfgs LR=<fixed-step val lr>;
# reference strong-Wolfe L-BFGS teacher: METHOD=lbfgs_wolfe LR=1)
METHOD=${METHOD:-adagrad}; LR=${LR:-0.03}; SQ=0.0316227766
G=${GROUP:-m5dw_f1}; T=_${C}_s$S
COMMON="dataset=v2 data_prefix=l63v2d model=f1 constraint=weak pattern=full cost.sigma_q=$SQ method=$METHOD lr=$LR \
result_group=$G tag=$T $X $EXTRA"
VAR=$(echo "$X" | sed 's/.*variant=\([a-z_]*\).*/\1/')
CKPT=tmp/meta/$G/${VAR}_weak_full_${METHOD}_lr${LR}${T}.pt
mkdir -p tmp/logs
[ -f $CKPT ] || $P src/run_meta_b.py $COMMON split=val > tmp/logs/${G}${T}_val.log 2>&1 || exit 1
for SP in $SPLITS; do
  $P src/run_meta_b.py $COMMON split=$SP checkpoint=$CKPT > tmp/logs/${G}${T}_${SP}.log 2>&1
done
