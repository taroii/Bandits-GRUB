#!/usr/bin/env bash
# Task 0 -- launch the GRUB bias-convention sweeps.
#
# One single-threaded process per (instance, bias).  Each runner is
# checkpointed per cell, so re-running this script resumes rather than
# restarting.  Logs land in experiments/outputs/logs/.
#
#   bash experiments/launch_grub_bias.sh [SEEDS]
set -u

SEEDS="${1:-5}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="$REPO/experiments/outputs/logs"
mkdir -p "$LOGDIR"

# Keep every process single-threaded: 8 concurrent BLAS pools would
# oversubscribe the machine, and that is what crashed the shared server
# previously (see experiments/utils/runners.py).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$REPO"

for BIAS in legacy published sqrt none; do
  for INST in chain movielens; do
    LOG="$LOGDIR/grub_bias_${INST}_${BIAS}.log"
    nohup python experiments/grub_bias_sweep.py \
        --instance "$INST" --bias "$BIAS" --seeds "$SEEDS" \
        >> "$LOG" 2>&1 &
    echo "launched $INST/$BIAS  pid=$!  log=$LOG"
    sleep 1
  done
done

echo
echo "All launched. Watch with:"
echo "  tail -f $LOGDIR/grub_bias_*.log"
wait
