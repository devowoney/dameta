#!/bin/bash
# Development testbed for the delta module (pilot 2026-11-18): ONE seed, 10 train cases, 20 val cases, val split only.
# Problem: dense-obs weak constraint, F1, sigma_q 0.1/sqrt(10) (same as the m5dw* meta runs).
# Usage: bash src/scripts/dev_delta.sh <name> <variant delta_exact|delta> <teacher lbfgs_wolfe|adagrad> [overrides]
#   delta_exact = supervised delta (imitate the teacher update)   delta = unsupervised delta (meta_b.py)
# Teacher lr / budget: strong-Wolfe L-BFGS lr 1, max_iter 150; Adagrad lr 0.03 (classical val selection), 1000.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P=/Odyssey/private/j25lee/miniforge3/envs/oceanai/bin/python
NAME=$1; VAR=$2; T=$3; EXTRA=$4
case $T in
  lbfgs_wolfe) TM="method=lbfgs_wolfe lr=1 max_iter=150";;
  adagrad)     TM="method=adagrad lr=0.03 max_iter=1000";;
  *) echo "unknown teacher $T"; exit 1;;
esac
G=dev_delta
NTR=${NTR:-10}                                   # env NTR: train cases (default 10; other values get their own pair cache)
PC=tmp/meta/$G/pairs_${T}; [ "$NTR" != 10 ] && PC=${PC}_n${NTR}_${NAME}
PC=${PC_OVERRIDE:-$PC}                            # env PC_OVERRIDE: reuse an existing teacher-pair cache prefix
mkdir -p tmp/logs
$P src/run_meta_b.py dataset=v2 data_prefix=l63v2d model=f1 constraint=weak pattern=full cost.sigma_q=0.0316227766 \
  variant=$VAR train.n_train=$NTR train.n_val_pairs_ic=20 n_cases=${NVAL:-20} split=val \
  delta_exact.pairs_cache=$PC delta.pairs_cache=$PC result_group=$G tag=_$NAME hydra.run.dir=tmp/hydra/dev_delta/$NAME $TM $EXTRA > tmp/logs/${G}_$NAME.log 2>&1
tail -1 tmp/logs/${G}_$NAME.log | sed 's/.*INFO\] - //; s/ -> .*//'
