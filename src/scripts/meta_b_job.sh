#!/bin/bash
# One Meta-B (delta, LOCKED C1 config) job: meta-train on train ICs (best epoch on val) -> val record + checkpoint,
# then frozen inference on test and test_ood from that checkpoint.
# Usage: bash src/scripts/meta_b_job.sh <constraint> <pattern> <dynamics l63|emulator> <emulator|-> <teacher lr> <seed>
#                                      [variant delta|supervised]   (default delta)
# Teacher lr = the E1/E3-selected Adagrad lr of that configuration. supervised = default supervised settings
# (configs/meta_b.yaml: MLP 3 x 128, 50 epochs, best epoch on val-IC teacher pairs), supervised.seed = <seed>.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P=/Odyssey/private/j25lee/miniforge3/envs/oceanai/bin/python
CON=$1; PAT=$2; DYN=$3; EMU=$4; LR=$5; S=$6; VAR=${7:-delta}
if [ "$DYN" = "l63" ]; then GROUP=e2b; EMUARG=""; else GROUP=e4b_$EMU; EMUARG="emulator=$EMU"; fi
if [ "$VAR" = "supervised" ]; then
  T=_sup_s$S
  LOCKED="variant=supervised supervised.seed=$S"
else
  T=_C1_s$S
  LOCKED="delta.episode_iters=100 delta.w_perf_delta=2.5 delta.grad_align_steps=2 delta.grad_trans_steps=2 \
delta.meta_lr=1e-2 net.output_scale=10 delta.n_epochs=3 delta.select_epoch=true delta.w_align_perf=0 delta.train_seed=$S"
fi
EXP=${VAR}_${CON}_${PAT}_adagrad_lr${LR}${T}
LOG=tmp/logs/${GROUP}_${EXP}
mkdir -p tmp/logs
COMMON="result_group=$GROUP constraint=$CON pattern=$PAT dynamics=$DYN $EMUARG method=adagrad lr=$LR tag=$T $LOCKED"
# Resume: if this job's checkpoint exists and loads (interrupted after training), skip training and rebuild every
# record from it (val included: an interrupted write can leave a corrupt val file). The val record then carries
# "loaded_from" instead of the training history (history stays in the original val log).
CKPT=tmp/meta/$GROUP/$EXP.pt
if [ -f $CKPT ] && $P -c "import torch; torch.load('$CKPT', weights_only=False)" 2>/dev/null; then
  SPLITS="val test test_ood"
  echo "resume: reuse $CKPT" >> ${LOG}_resume.log
else
  $P src/run_meta_b.py $COMMON split=val > ${LOG}_val.log 2>&1 || exit 1
  SPLITS="test test_ood"
fi
for SPLIT in $SPLITS; do
  $P src/run_meta_b.py $COMMON split=$SPLIT checkpoint=tmp/meta/$GROUP/$EXP.pt > ${LOG}_${SPLIT}.log 2>&1
done
