#!/usr/bin/env bash
# Resource-guarded launcher for long experiments.
#
# sklearn's HistGradientBoosting* parallelises through OpenMP and by
# default uses every core, and numpy/scipy do the same through their BLAS
# backend. Several "single" experiment processes therefore expand to many
# times the core count and can starve the rest of the machine. Nothing in
# the Python code requests that, so it is limited at launch.
#
# Guards applied:
#   1. thread ceiling  -- every parallel backend pinned to $JOB_THREADS
#   2. memory ceiling  -- hard address-space cap; the job dies, not the OS
#   3. nice + ionice   -- the desktop always outranks a batch experiment
#   4. pre-flight halt -- refuses to start when free memory is already low
#
# usage: scripts/safe_run.sh [-t THREADS] [-m MEM_GB] -- <command...>

set -euo pipefail

JOB_THREADS="${JOB_THREADS:-4}"
JOB_MEM_GB="${JOB_MEM_GB:-6}"
MIN_FREE_GB="${MIN_FREE_GB:-2}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -t|--threads) JOB_THREADS="$2"; shift 2 ;;
    -m|--mem-gb)  JOB_MEM_GB="$2";  shift 2 ;;
    --)           shift; break ;;
    *)            break ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "usage: scripts/safe_run.sh [-t THREADS] [-m MEM_GB] -- <command...>" >&2
  exit 2
fi

CORES="$(nproc)"
AVAIL_GB="$(awk '/MemAvailable/ {printf "%.1f", $2/1048576}' /proc/meminfo)"

# Pre-flight: starting a heavy job on an already-starved machine is how
# the desktop dies. Refuse rather than "try anyway".
if (( $(echo "$AVAIL_GB < $MIN_FREE_GB" | bc -l) )); then
  echo "REFUSING TO START: only ${AVAIL_GB}GB available, need >= ${MIN_FREE_GB}GB" >&2
  echo "  free up memory or lower MIN_FREE_GB if you accept the risk" >&2
  exit 1
fi

# Every parallel backend that could otherwise expand to $CORES threads.
# OMP_NUM_THREADS alone is NOT enough -- MKL, OpenBLAS, NumExpr and
# Accelerate each read their own variable, and sklearn reads OMP.
export OMP_NUM_THREADS="$JOB_THREADS"
export OPENBLAS_NUM_THREADS="$JOB_THREADS"
export MKL_NUM_THREADS="$JOB_THREADS"
export NUMEXPR_NUM_THREADS="$JOB_THREADS"
export VECLIB_MAXIMUM_THREADS="$JOB_THREADS"
export JOBLIB_START_METHOD=loky

echo "[safe_run] threads=$JOB_THREADS/$CORES  mem_cap=${JOB_MEM_GB}GB  available=${AVAIL_GB}GB"
echo "[safe_run] cmd: $*"

# ulimit -v caps ADDRESS SPACE, so an unbounded allocation kills this job
# with MemoryError instead of pushing the machine into swap-thrash and
# taking the session with it. nice/ionice keep the desktop responsive.
(
  ulimit -v $((JOB_MEM_GB * 1024 * 1024)) || true
  exec nice -n 15 ionice -c3 "$@"
)
