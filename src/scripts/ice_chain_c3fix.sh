#!/bin/bash
# Cell 3 rerun with the backtrack safeguard (2026-11-20) + OSSE of cells 3 / 4 + ICE table. CPU only (no --gres),
# >= 12 GB per run, <= 12 batch jobs, one run per job on different nodes (src/scripts/sb_submit.py).
# Login node: squeue / sbatch only; stops at the first failed check. Log: tmp/logs/ice/chain_c34.log (appended)
# Usage (project root): nohup setsid bash src/scripts/ice_chain_c3fix.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/../.."
LOG=tmp/logs/ice/chain_c34.log
log() { echo "[$(date '+%F %T')] $*" >> $LOG; }
wait_name() { while squeue -h -u "$USER" -o %j | grep -q "^$1"; do sleep 30; done; }
SG='"test test_ood" "unroll.safeguard=backtrack"'

log "cell-3 fix chain start (backtrack safeguard + NaN guard)"
python3 src/scripts/sb_submit.py tmp/logs/ice/gate2.txt ice_gate_c3 --jobs 1 --cpus 1 >> $LOG 2>&1
wait_name ice_gate_c3
RES=$(grep "tests passed" tmp/logs/ice/gate_test_ice.log | tail -1); log "gate: $RES"
read -r OK TOT <<< "$(echo "$RES" | sed 's#\([0-9]*\)/\([0-9]*\).*#\1 \2#')"
if [ -z "$OK" ] || [ "$OK" != "$TOT" ]; then log "STOP: gate failed"; exit 1; fi

: > tmp/logs/ice/c3_all.txt
for s in 0 1 2; do for m in f1 emulator; do for k in strong weak; do
  echo "bash src/scripts/ice_meta_cells.sh c3 $m $k $s $SG" >> tmp/logs/ice/c3_all.txt; done; done; done
python3 src/scripts/sb_submit.py tmp/logs/ice/c3_all.txt ice_c3 --jobs 12 --cpus 1 >> $LOG 2>&1
wait_name ice_c3
N=$(ls tmp/results/ice_meta_*/*_c3_*.nc 2>/dev/null | wc -l); TB=$(grep -l Traceback tmp/logs/ice/ice_meta_*_c3_*.log | wc -l)
log "cell 3: $N / 24 records, $TB logs with a traceback"

: > tmp/logs/ice/osse_meta_c34.txt
for c in c3 c4; do for s in 0 1 2; do for k in strong weak; do
  echo "OSSE=1 bash src/scripts/ice_meta_cells.sh $c f1 $k $s $SG" >> tmp/logs/ice/osse_meta_c34.txt; done; done; done
python3 src/scripts/sb_submit.py tmp/logs/ice/osse_meta_c34.txt ice_osse_c34 --jobs 6 --cpus 2 >> $LOG 2>&1
wait_name ice_osse_c34
N=$(ls tmp/results/ice_osse_meta_f1/ | grep -c "_c[34]_"); log "OSSE cells 3 / 4: $N / 24 records"

python3 src/scripts/sb_submit.py tmp/logs/ice/report.txt ice_report --jobs 1 --cpus 1 >> $LOG 2>&1
wait_name ice_report
log "report done -> tmp/results/ice/ice_table_{test,test_ood}.md"
log "cell-3 fix chain end"
