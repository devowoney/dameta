#!/bin/bash
# Runs dense_meta_job.sh jobs from a list, N_PAR at a time. List lines: cell|seed|splits|extra overrides
# Usage: bash src/scripts/dense_meta_run.sh <list file> [n_par]
cd "$(dirname "$0")/../.."
LIST=$1; NP=${2:-3}
while IFS='|' read -r c s sp x; do
  [ -z "$c" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge "$NP" ]; do sleep 20; done
  bash src/scripts/dense_meta_job.sh "$c" "$s" "$sp" "$x" &
done < "$LIST"
wait
echo ALL DONE
