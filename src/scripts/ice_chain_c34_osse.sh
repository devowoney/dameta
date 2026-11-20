#!/bin/bash
# After the full cells 3 / 4 runs (CPU only, 6 jobs on 6 nodes, pilot 2026-11-20): OSSE frozen inference of the F1
# learners (12 runs, 3 jobs on 3 nodes) -> ICE table. No GPU request anywhere. Login node: squeue / sbatch only.
# Usage (project root): nohup setsid bash src/scripts/ice_chain_c34_osse.sh "<full job ids, comma-separated>" &
# Log: tmp/logs/ice/chain_c34.log (appended)
cd "$(dirname "$0")/../.."
ROOT=$PWD; LOG=tmp/logs/ice/chain_c34.log
log() { echo "[$(date '+%F %T')] $*" >> $LOG; }
wait_ids() { while [ -n "$(squeue -h -j "$1" 2>/dev/null)" ]; do sleep 30; done; }
wait_name() { while squeue -h -u "$USER" -o %j | grep -q "^$1"; do sleep 30; done; }

log "CPU-only chain: waiting for full jobs $1"
wait_ids "$1"
N=$(ls tmp/results/ice_meta_*/ | grep -c "_c[34]_"); log "full: $N / 48 records (24 runs x test / test_ood)"

: > tmp/logs/ice/osse_meta_c34.txt
for c in c3 c4; do for s in 0 1 2; do for k in strong weak; do
  echo "OSSE=1 bash src/scripts/ice_meta_cells.sh $c f1 $k $s" >> tmp/logs/ice/osse_meta_c34.txt; done; done; done
python3 src/scripts/sb_submit.py tmp/logs/ice/osse_meta_c34.txt ice_osse_c34 --jobs 3 --cpus 4 >> $LOG 2>&1
wait_name ice_osse_c34
N=$(ls tmp/results/ice_osse_meta_f1/ | grep -c "_c[34]_"); log "OSSE: $N / 24 records"

python3 src/scripts/sb_submit.py tmp/logs/ice/report.txt ice_report --jobs 1 --cpus 1 >> $LOG 2>&1
wait_name ice_report
log "report done -> tmp/results/ice/ice_table_{test,test_ood}.md"
log "CPU-only chain end"
