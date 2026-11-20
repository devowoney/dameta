#!/bin/bash
# ICE tables, final design (pilot 2026-11-20): Table 1 OSSE (obs from F1, meta TRAINED here), Table 2 OSE (real-system
# obs, OSSE learners FROZEN). CPU only (no --gres), 12 GB per run, <= 12 batch jobs (src/scripts/sb_submit.py).
# Login node: squeue / sbatch only; stops at the first failed check. Log: tmp/logs/ice/chain_v2.log
# Usage (project root): nohup setsid bash src/scripts/ice_chain_v2.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/../.."
LOG=tmp/logs/ice/chain_v2.log; L=tmp/logs/ice
log() { echo "[$(date '+%F %T')] $*" >> $LOG; }
wait_name() { while squeue -h -u "$USER" -o %j | grep -q "^$1"; do sleep 30; done; }
submit() { python3 src/scripts/sb_submit.py "$@" >> $LOG 2>&1; }
cells() {   # cells() <mode> <cells> <models> <seeds> -> command lines
  for s in $4; do for c in $2; do for m in $3; do for k in strong weak; do
    echo "bash src/scripts/ice_meta_v2.sh $1 $c $m $k $s"; done; done; done; done; }
count() { ls tmp/results/$1/ 2>/dev/null | grep -c "$2"; }
fail_logs() { grep -l Traceback $L/$1 2>/dev/null | wc -l; }

log "v2 chain start"
# 1. OSSE train split (meta training data) + classical OSSE rows of F2 (F1 done: tmp/results/ice_osse_f1)
echo "python src/make_osse_obs.py --splits train > $L/make_osse_train.log 2>&1" > $L/make_osse_train.txt
submit $L/make_osse_train.txt ice_v2_data --jobs 1 --cpus 1
submit src/scripts/lists/ice_osse.txt ice_v2_classical --jobs 4 --cpus 4
wait_name ice_v2_data
if [ ! -f tmp/data/l63v2osse_train_seed42.nc ]; then log "STOP: OSSE train split missing"; exit 1; fi
log "OSSE train split: $(tail -1 $L/make_osse_train.log)"

# 2. cell 1 seed 0 (writes the OSSE teacher-pair caches used by cells 1 / 2; quick check of the wrapper)
cells osse c1 "f1 emulator" 0 > $L/v2_c1s0.txt
submit $L/v2_c1s0.txt ice_v2_c1s0 --jobs 4 --cpus 1
wait_name ice_v2_c1s0
N=$(count ice_osse_meta_f1 _c1_)$'+'$(count ice_osse_meta_emulator _c1_); TB=$(fail_logs "ice_osse_meta_*_c1_*s0_*.log")
log "cell 1 seed 0: records $N (expect 4+4), $TB logs with a traceback"
if [ "$TB" != 0 ]; then log "STOP: cell 1 seed 0 failed"; exit 1; fi

# 3. every other OSSE training: cell 1 seeds 1-2, cells 2 / 3 / 4 seeds 0-2 (44 runs over 12 jobs)
{ cells osse c1 "f1 emulator" "1 2"; cells osse "c2 c3 c4" "f1 emulator" "0 1 2"; } > $L/v2_train.txt
wait_name ice_v2_classical
submit $L/v2_train.txt ice_v2_train --jobs 12 --cpus 4
wait_name ice_v2_train
for m in f1 emulator; do log "OSSE $m: $(count ice_osse_meta_$m '\.nc') / 48 meta records, classical $(count ice_osse_$m '\.nc') / 16"; done
log "OSSE training logs with a traceback: $(fail_logs 'ice_osse_meta_*.log')"

# 4. OSE: OSSE-trained learners frozen on the real-system observations (48 runs x test / test_ood)
cells ose "c1 c2 c3 c4" "f1 emulator" "0 1 2" > $L/v2_ose.txt
submit $L/v2_ose.txt ice_v2_ose --jobs 12 --cpus 4
wait_name ice_v2_ose
for m in f1 emulator; do log "OSE $m: $(count ice_ose_meta_$m '\.nc') / 48 meta records"; done

# 5. tables
submit $L/report.txt ice_v2_report --jobs 1 --cpus 1
wait_name ice_v2_report
log "tables -> tmp/results/ice/"
log "v2 chain end"
