#!/bin/bash
# Runs a list of speed_sweep_job.sh jobs, N_PAR at a time.
# List file lines: model|name|teacher lr|seed|overrides|splits
# Usage: bash src/scripts/speed_sweep_run.sh <list file> [n_par]
cd "$(dirname "$0")/../.."
LIST=$1; NP=${2:-3}
while IFS='|' read -r m n lr s x sp; do
  [ -z "$m" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge "$NP" ]; do sleep 20; done
  bash src/scripts/speed_sweep_job.sh "$m" "$n" "$lr" "$s" "$x" "$sp" &
done < "$LIST"
wait
echo ALL DONE
