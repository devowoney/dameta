#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --account=odyssey
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --job-name=dameta
#SBATCH --output=/Odyssey/private/j25lee/.bin/log/%x_%j.log
# ----------------------------------------------------------------------------------------------------------------------
# dameta: run the lines of a command-list file in parallel, ONE thread each (L63 tensors are tiny: per-op overhead,
# not arithmetic, dominates, so threads only add contention). Lines run from the project root given as $2.
# CPU ONLY (pilot 2026-11-20): never request a GPU (--gres) — the runs do not use it and it blocks other users.
# Usage:  sbatch --nodelist=sl-mee-br-<N> [--job-name=x] [--cpus-per-task=N] src/scripts/sb_parallel.sh <cmd_list_file>
#                <project_root> [n_par]
#   n_par default = SLURM_CPUS_PER_TASK. Each line's stdout/stderr should be redirected inside the line itself.
# ----------------------------------------------------------------------------------------------------------------------
set -e
LIST=$1; ROOT=$2; NPAR=${3:-${SLURM_CPUS_PER_TASK:-8}}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH=/Odyssey/private/j25lee/miniforge3/envs/oceanai/bin:$PATH
cd "$ROOT"
echo "job $SLURM_JOB_ID on $(hostname): $(grep -cv '^\s*\(#\|$\)' "$LIST") commands from $LIST, $NPAR in parallel"
grep -v '^\s*\(#\|$\)' "$LIST" | xargs -d '\n' -P "$NPAR" -I{} bash -c '{}'   # -d: no quote processing (keeps quoted args)
echo "job $SLURM_JOB_ID done $(date)"
