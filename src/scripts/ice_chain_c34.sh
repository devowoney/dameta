#!/bin/bash
# Overnight driver of the ICE-table rows 7-8 (cells 3 and 4), pilot request 2026-11-19: start only after EVERY cell-2
# job (training rerun + OSSE inference) has finished, then gate -> smoke -> full (3 seeds) -> OSSE (frozen) -> report.
# Runs on the login node but only calls squeue / sbatch (no computation); stops at the first failed check.
# Usage (project root): nohup bash src/scripts/ice_chain_c34.sh "<cell-2 job ids>" > /dev/null 2>&1 &
# Log: tmp/logs/ice/chain_c34.log
cd "$(dirname "$0")/../.."
ROOT=$PWD; LOG=tmp/logs/ice/chain_c34.log; WAIT_IDS=$1
SB="--gres=gpu:a40:1 --nodelist=sl-mee-br-205"                    # single-job stages: node 205 (a40)
log() { echo "[$(date '+%F %T')] $*" >> $LOG; }
wait_ids() { while [ -n "$(squeue -h -j "$1" 2>/dev/null)" ]; do sleep 60; done; }
wait_name() { while squeue -h -u "$USER" -o %j | grep -q "^$1"; do sleep 60; done; }
slots() { echo $(( 8 - $(squeue -h -u "$USER" | wc -l) )); }        # pilot rule: at most 8 of my jobs

log "chain start; waiting for cell-2 jobs $WAIT_IDS"
wait_ids "$WAIT_IDS"
log "cell-2 jobs finished"

# 1. gate: all ICE tests (classical, cells 2, 3, 4)
J=$(sbatch --parsable --job-name=ice_gate_c34 $SB --cpus-per-task=1 --mem=8G src/scripts/sb_parallel.sh \
    tmp/logs/ice/gate2.txt "$ROOT" 1); log "gate job $J"
wait_ids "$J"
RES=$(grep "tests passed" tmp/logs/ice/gate_test_ice.log | tail -1); log "gate: $RES"
read -r OK TOT <<< "$(echo "$RES" | sed 's#\([0-9]*\)/\([0-9]*\).*#\1 \2#')"
if [ -z "$OK" ] || [ "$OK" != "$TOT" ]; then log "STOP: gate failed"; exit 1; fi

# 2. smoke: cells 3 / 4 x F1 / F2 x strong / weak (10 train, 20 val, 20 cases)
: > tmp/logs/ice/smoke_c34.txt
for c in c3 c4; do for m in f1 emulator; do for k in strong weak; do
  echo "SMOKE=1 bash src/scripts/ice_meta_cells.sh $c $m $k 0" >> tmp/logs/ice/smoke_c34.txt; done; done; done
J=$(sbatch --parsable --job-name=ice_smoke_c34 $SB --cpus-per-task=8 --mem=32G src/scripts/sb_parallel.sh \
    tmp/logs/ice/smoke_c34.txt "$ROOT" 8); log "smoke job $J"
wait_ids "$J"
N=$(ls tmp/results/ice_smoke_*/ 2>/dev/null | grep -c "_c[34]_"); TB=$(grep -l Traceback tmp/logs/ice/ice_smoke_*c[34]_*.log | wc -l)
log "smoke: $N / 16 records, $TB logs with a traceback"
if [ "$N" != 16 ] || [ "$TB" != 0 ]; then log "STOP: smoke failed"; exit 1; fi

# 3. full: 2 cells x 2 models x 2 constraints x 3 seeds = 24 runs (no shared cache: all seeds in parallel)
: > tmp/logs/ice/c34_all.txt
for s in 0 1 2; do for c in c3 c4; do for m in f1 emulator; do for k in strong weak; do
  echo "bash src/scripts/ice_meta_cells.sh $c $m $k $s" >> tmp/logs/ice/c34_all.txt; done; done; done; done
NJ=$(slots); [ "$NJ" -gt 6 ] && NJ=6; [ "$NJ" -lt 1 ] && NJ=1
python3 src/scripts/sb_submit.py tmp/logs/ice/c34_all.txt ice_c34 --jobs "$NJ" --cpus 4 >> $LOG 2>&1
wait_name ice_c34
N=$(ls tmp/results/ice_meta_*/ | grep -c "_c[34]_"); log "full: $N / 48 records (24 runs x test / test_ood)"

# 4. OSSE: frozen F1 learners of cells 3 / 4 on the F1-generated observations
: > tmp/logs/ice/osse_meta_c34.txt
for c in c3 c4; do for s in 0 1 2; do for k in strong weak; do
  echo "OSSE=1 bash src/scripts/ice_meta_cells.sh $c f1 $k $s" >> tmp/logs/ice/osse_meta_c34.txt; done; done; done
J=$(sbatch --parsable --job-name=ice_osse_c34 $SB --cpus-per-task=12 --mem=32G src/scripts/sb_parallel.sh \
    tmp/logs/ice/osse_meta_c34.txt "$ROOT" 12); log "OSSE job $J"
wait_ids "$J"
N=$(ls tmp/results/ice_osse_meta_f1/ | grep -c "_c[34]_"); log "OSSE: $N / 24 records"

# 5. table
J=$(sbatch --parsable --job-name=ice_report $SB --cpus-per-task=1 --mem=8G src/scripts/sb_parallel.sh \
    tmp/logs/ice/report.txt "$ROOT" 1); wait_ids "$J"
log "report done -> tmp/results/ice/ice_table_{test,test_ood}.md"
log "chain end"
