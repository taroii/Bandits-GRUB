#!/usr/bin/env bash
#
# Rebuttal-round experiment sweep, sized for an overnight run on a many-core
# box.  Everything here is checkpointed per cell: re-running this script
# resumes from where it stopped rather than restarting.
#
#   bash run_server.sh              # full sweep, default seed counts
#   bash run_server.sh --pilot      # 5 seeds everywhere, much shorter
#   JOBS=32 bash run_server.sh      # cap concurrency (default: cores - 2)
#
# Logs land in experiments/outputs/logs/<name>.log; results in
# experiments/outputs/*.npz.  Nothing overwrites the committed paper data
# (main_2_results.npz, movielens_1_results.npz, fb_1_results.npz,
# fb_ablation_results.npz, kernel_1_results.npz, mis_2_results.npz).
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
LOG="$REPO/experiments/outputs/logs"
mkdir -p "$LOG"

PILOT=0
[ "${1:-}" = "--pilot" ] && PILOT=1

# One BLAS thread per process; we get parallelism from running many processes,
# and oversubscribed BLAS pools are what destabilised earlier shared-server
# runs (see the note in experiments/utils/runners.py).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CORES=$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu)
JOBS="${JOBS:-$(( CORES > 3 ? CORES - 2 : 1 ))}"
echo "[run_server] $CORES cores detected, running up to $JOBS concurrent jobs"

if [ "$PILOT" = "1" ]; then
  SEEDS_MAIN=5;  SEEDS_APPX=5;  MAXSTEPS=2000000
else
  SEEDS_MAIN=20; SEEDS_APPX=10; MAXSTEPS=10000000
fi
echo "[run_server] seeds: main=$SEEDS_MAIN appendix=$SEEDS_APPX  max_steps=$MAXSTEPS"

# ---------------------------------------------------------------------------
# Simple job queue: `queue <name> <command...>`, then `drain`.
# ---------------------------------------------------------------------------
declare -a NAMES=() CMDS=()
queue() { NAMES+=("$1"); shift; CMDS+=("$*"); }

drain() {
  local i=0 running=0
  for i in "${!NAMES[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done
    echo "  -> ${NAMES[$i]}"
    # shellcheck disable=SC2086
    nohup bash -c "${CMDS[$i]}" > "$LOG/${NAMES[$i]}.log" 2>&1 &
    sleep 0.5
  done
  echo "[run_server] all launched; waiting..."
  wait
  echo "[run_server] all finished"
}

# ---------------------------------------------------------------------------
# Task 0 -- GRUB bias conventions (8 jobs)
# ---------------------------------------------------------------------------
for BIAS in legacy published sqrt none; do
  for INST in chain movielens; do
    queue "grub_bias_${INST}_${BIAS}" \
      "python -u experiments/grub_bias_sweep.py --instance $INST --bias $BIAS \
         --seeds $SEEDS_MAIN --max-steps $MAXSTEPS"
  done
done

# ---------------------------------------------------------------------------
# Task 2 -- smoothness x graph feedback (2 jobs)
# ---------------------------------------------------------------------------
queue "fb_smooth_er_smooth" \
  "python -u experiments/fb_smooth.py --instance er_smooth --n 20 \
     --ps 0.1 0.2 0.4 0.6 0.8 1.0 --rhos 1 10 100 1000 \
     --seeds $SEEDS_MAIN --max-steps 3000000"
queue "fb_smooth_er_uniform" \
  "python -u experiments/fb_smooth.py --instance er_uniform --n 20 \
     --ps 0.1 0.2 0.4 0.6 0.8 1.0 --rhos 1 10 100 1000 \
     --seeds $SEEDS_MAIN --max-steps 3000000"

# ---------------------------------------------------------------------------
# Task 3 -- eps over/under-specification, both rho_diag policies (2 jobs)
# ---------------------------------------------------------------------------
queue "eps_sensitivity_fixed" \
  "python -u experiments/eps_sensitivity.py --seeds $SEEDS_MAIN \
     --max-steps $MAXSTEPS --rho-diag-policy fixed --adaptive --residual probe"
queue "eps_sensitivity_scaled" \
  "python -u experiments/eps_sensitivity.py --seeds $SEEDS_APPX \
     --max-steps $MAXSTEPS --rho-diag-policy scaled"

# ---------------------------------------------------------------------------
# Task 4 -- scale extensions (3 jobs)
# ---------------------------------------------------------------------------
queue "fb_scale_p0.2" \
  "python -u experiments/fb_scale.py --ns 20 50 100 200 --p 0.2 \
     --seeds $SEEDS_MAIN --max-steps $MAXSTEPS"
queue "fb_scale_p0.6" \
  "python -u experiments/fb_scale.py --ns 20 50 100 200 --p 0.6 \
     --seeds $SEEDS_MAIN --max-steps $MAXSTEPS"
# MovieLens K=50. GRUB is excluded by default -- see REBUTTAL_FINDINGS.md §11
# for the extrapolation that motivates it. Add it back by appending GRUB to
# --algos if you want the cap confirmed empirically.
queue "movielens_K50" \
  "python -u experiments/movielens_1.py --K 50 --rhos 1 3 10 30 100 300 1000 \
     --seeds $SEEDS_APPX --algos TS-Explore 'Basic TS' KL-LUCB \
     --max-steps $MAXSTEPS"

# ---------------------------------------------------------------------------
# Task 6 -- TaS-FG baseline (1 job)
# ---------------------------------------------------------------------------
queue "tas_fg" \
  "python -u experiments/tas_fg_run.py --n 20 --ps 0.1 0.2 0.4 0.6 0.8 1.0 \
     --seeds $SEEDS_MAIN --max-steps $MAXSTEPS"

# ---------------------------------------------------------------------------
# Task 1 -- instrumented traces (4 jobs; these write ~60 MB to
# experiments/outputs/traces/, which is gitignored)
# ---------------------------------------------------------------------------
for K in 20 50 200; do
  queue "trace_chain_K$K" \
    "python -u experiments/trace_runs.py --instance chain --K $K --rho 100 \
       --seeds $SEEDS_APPX --every 200 --algos TS-Explore 'Basic TS' KL-LUCB"
done
# rho_var(eps_true) on the K=31 SBM; see REBUTTAL_FINDINGS.md §2 and §8 for why
# rho=100 is not informative for the influence-factor comparison.
queue "trace_sbm31_rhovar" \
  "python -u experiments/trace_runs.py --instance sbm31 --rho 10960 \
     --seeds $SEEDS_APPX --every 100 --algos TS-Explore 'Basic TS' KL-LUCB"

echo "[run_server] queued ${#NAMES[@]} jobs"
drain

# ---------------------------------------------------------------------------
# Zero-compute tables and all figures (fast, run serially at the end)
# ---------------------------------------------------------------------------
echo "[run_server] graph parameter tables + figures"
python -u experiments/graph_params.py --n 20 > "$LOG/graph_params.log" 2>&1
python -u experiments/grub_audit.py --Ks 10 20 50 100 200 \
  > "$LOG/grub_audit.log" 2>&1
python -u experiments/grub_bias_plot.py   > "$LOG/grub_bias_plot.log" 2>&1
python -u experiments/fb_smooth_plot.py   > "$LOG/fb_smooth_plot.log" 2>&1
python -u experiments/trace_analysis.py   > "$LOG/trace_analysis.log" 2>&1

echo
echo "[run_server] done. Results:  experiments/outputs/*.npz"
echo "             Figures:       experiments/outputs/{grub_bias,fb_smooth}.pdf"
echo "             Tables/logs:   $LOG/"
