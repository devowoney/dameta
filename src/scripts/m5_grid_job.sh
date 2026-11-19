#!/bin/bash
# One M5 meta-grid job on D1 v2 (strong / full): train on train ICs (best epoch / selection on val) -> val record +
# checkpoint, then frozen test and test_ood from the checkpoint.
# Usage: bash src/scripts/m5_grid_job.sh <model f1|emulator> <cell c1|...|c8|c8a> <seed> [constraint] [pattern]
# Teacher (where used) = Adagrad lr 0.3 (classical D1 v2 selection for F1 and F2, strong).
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P=/Odyssey/private/j25lee/miniforge3/envs/oceanai/bin/python
M=$1; C=$2; S=$3
case $C in
  c1)  X="variant=delta delta.output=stretch delta.episode_iters=100 delta.w_perf_delta=2.5 delta.grad_align_steps=2 \
delta.grad_trans_steps=2 delta.meta_lr=1e-2 net.output_scale=10 delta.n_epochs=3 delta.select_epoch=true \
delta.train_seed=$S";;
  c2)  X="variant=lookahead net.output_scale=10 lookahead.seed=$S";;
  c3)  X="variant=greedy greedy.seed=$S";;
  c3o) X="variant=greedy greedy.obs_input=true greedy.seed=$S";;
  c4)  X="variant=supervised supervised.seed=$S";;
  c4o) X="variant=supervised supervised.obs_input=true supervised.seed=$S";;
  c5)  X="variant=unroll unroll.seed=$S";;
  c7)  X="variant=unroll unroll.use_grad=false unroll.seed=$S";;
  c7o) X="variant=unroll unroll.use_grad=false unroll.obs_input=true unroll.seed=$S";;
  c8)  X="variant=learned_grad learned_grad.arch=mlp learned_grad.data=trajectory learned_grad.seed=$S";;
  c8a) X="variant=learned_grad learned_grad.arch=adjoint learned_grad.data=sampled learned_grad.loss=jacobian \
learned_grad.traj_stride=0 learned_grad.seed=$S";;
  *) echo "unknown cell $C"; exit 1;;
esac
CON=${4:-strong}; PAT=${5:-full}                 # optional: constraint, obs pattern (default strong / full)
LR=0.3; [ "$CON" = weak ] && LR=0.1               # classical D1 v2 Adagrad selection (F1 and F2)
EMU=""; [ "$M" = emulator ] && EMU="emulator=rho275"
G=m5_$M
T=_${C}_s$S
COMMON="dataset=v2 model=$M $EMU constraint=$CON pattern=$PAT method=adagrad lr=$LR result_group=$G tag=$T $X"
VAR=$(echo "$X" | sed 's/.*variant=\([a-z_]*\).*/\1/')
EXP=${VAR}_${CON}_${PAT}_adagrad_lr${LR}${T}
LOG=tmp/logs/${G}${T}
[ "$CON/$PAT" != "strong/full" ] && LOG=tmp/logs/${G}_${CON}_${PAT}${T}
mkdir -p tmp/logs
CKPT=tmp/meta/$G/$EXP.pt
if [ -f $CKPT ] && $P -c "import torch; torch.load('$CKPT', weights_only=False)" 2>/dev/null; then
  SPLITS="val test test_ood"
else
  $P src/run_meta_b.py $COMMON split=val > ${LOG}_val.log 2>&1 || exit 1
  SPLITS="test test_ood"
fi
for SP in $SPLITS; do
  $P src/run_meta_b.py $COMMON split=$SP checkpoint=$CKPT > ${LOG}_${SP}.log 2>&1
done
