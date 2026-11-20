#!/bin/bash
# Fast driver of the ICE-table rows 7-8 (cells 3 and 4), pilot request 2026-11-20: use the empty h100 / h200 nodes
# sl-mee-br-211 (h100) and sl-mee-br-215 (h200). Replaces ice_chain_c34.sh after its first gate:
#   wait gate job (given) -> check -> smoke cell 3 only (cell 4 smoke passed, 8 / 8) -> full 24 runs as TWO jobs
#   (12 runs each, one thread per run, 211 and 215) -> OSSE 12 runs (211) -> report (211).
# Runs on the login node, only squeue / sbatch; stops at the first failed check.
# Usage (project root): nohup setsid bash src/scripts/ice_chain_c34_fast.sh <gate job id> > /dev/null 2>&1 &
# Log: tmp/logs/ice/chain_c34.log (appended)
cd "$(dirname "$0")/../.."
ROOT=$PWD; LOG=tmp/logs/ice/chain_c34.log; GATE=$1
A="--nodelist=sl-mee-br-211 --gres=gpu:h100:1"; B="--nodelist=sl-mee-br-215 --gres=gpu:h200:1"
log() { echo "[$(date '+%F %T')] $*" >> $LOG; }
wait_ids() { while [ -n "$(squeue -h -j "$1" 2>/dev/null)" ]; do sleep 30; done; }

log "fast chain start (nodes 211 h100 / 215 h200); waiting for gate $GATE"
wait_ids "$GATE"
RES=$(grep "tests passed" tmp/logs/ice/gate_test_ice.log | tail -1); log "gate: $RES"
read -r OK TOT <<< "$(echo "$RES" | sed 's#\([0-9]*\)/\([0-9]*\).*#\1 \2#')"
if [ -z "$OK" ] || [ "$OK" != "$TOT" ]; then log "STOP: gate failed"; exit 1; fi

# smoke: cell 3 only (config key unroll.base added after the first smoke)
: > tmp/logs/ice/smoke_c3.txt
for m in f1 emulator; do for k in strong weak; do
  echo "SMOKE=1 bash src/scripts/ice_meta_cells.sh c3 $m $k 0" >> tmp/logs/ice/smoke_c3.txt; done; done
J=$(sbatch --parsable --job-name=ice_smoke_c3 $A --cpus-per-task=4 --mem=16G src/scripts/sb_parallel.sh \
    tmp/logs/ice/smoke_c3.txt "$ROOT" 4); log "smoke c3 job $J (211)"
wait_ids "$J"
N=$(ls tmp/results/ice_smoke_*/ 2>/dev/null | grep -c "_c3_"); TB=$(grep -l Traceback tmp/logs/ice/ice_smoke_*c3_*.log | wc -l)
log "smoke c3: $N / 8 records, $TB logs with a traceback"
if [ "$N" != 8 ] || [ "$TB" != 0 ]; then log "STOP: smoke c3 failed"; exit 1; fi

# full: 24 runs, dealt round-robin over two jobs (211 / 215), 12 threads each
: > tmp/logs/ice/c34_all.txt
for s in 0 1 2; do for c in c3 c4; do for m in f1 emulator; do for k in strong weak; do
  echo "bash src/scripts/ice_meta_cells.sh $c $m $k $s" >> tmp/logs/ice/c34_all.txt; done; done; done; done
awk 'NR % 2 == 1' tmp/logs/ice/c34_all.txt > tmp/logs/ice/c34_a.txt
awk 'NR % 2 == 0' tmp/logs/ice/c34_all.txt > tmp/logs/ice/c34_b.txt
JA=$(sbatch --parsable --job-name=ice_c34_a $A --cpus-per-task=12 --mem=64G src/scripts/sb_parallel.sh \
     tmp/logs/ice/c34_a.txt "$ROOT" 12)
JB=$(sbatch --parsable --job-name=ice_c34_b $B --cpus-per-task=12 --mem=64G src/scripts/sb_parallel.sh \
     tmp/logs/ice/c34_b.txt "$ROOT" 12)
log "full jobs $JA (211) / $JB (215), 12 runs each"
wait_ids "$JA,$JB"
N=$(ls tmp/results/ice_meta_*/ | grep -c "_c[34]_"); log "full: $N / 48 records (24 runs x test / test_ood)"

# OSSE: frozen F1 learners of cells 3 / 4 on the F1-generated observations
: > tmp/logs/ice/osse_meta_c34.txt
for c in c3 c4; do for s in 0 1 2; do for k in strong weak; do
  echo "OSSE=1 bash src/scripts/ice_meta_cells.sh $c f1 $k $s" >> tmp/logs/ice/osse_meta_c34.txt; done; done; done
J=$(sbatch --parsable --job-name=ice_osse_c34 $A --cpus-per-task=12 --mem=32G src/scripts/sb_parallel.sh \
    tmp/logs/ice/osse_meta_c34.txt "$ROOT" 12); log "OSSE job $J (211)"
wait_ids "$J"
N=$(ls tmp/results/ice_osse_meta_f1/ | grep -c "_c[34]_"); log "OSSE: $N / 24 records"

J=$(sbatch --parsable --job-name=ice_report $A --cpus-per-task=1 --mem=8G src/scripts/sb_parallel.sh \
    tmp/logs/ice/report.txt "$ROOT" 1); wait_ids "$J"
log "report done -> tmp/results/ice/ice_table_{test,test_ood}.md"
log "fast chain end"
