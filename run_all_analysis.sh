#!/bin/bash
# Run full analysis for all 8 systems (2 replicas × 4 systems)
# Usage: bash run_all_analysis.sh [--quick]
set -euo pipefail
RDIR="$(cd "$(dirname "$0")" && pwd)"
ANA="$RDIR/analysis/psma_md_analysis.py"
PYTHON=/home/stalin/miniforge3/bin/python3
export SASA_STRIDE=50   # 1 SASA per ns (200 total after skip)
export HBOND_STRIDE=10  # H-bond check every 200 ps
export SKIP_NS=100       # discard first 100 ns as equilibration

mkdir -p "$RDIR/out"
LOG="$RDIR/out/run_all.log"
echo "[$(date)] Starting analysis" | tee "$LOG"

run_system() {
    local SYS=$1 REP=$2
    local SDIR="$RDIR/systems/$SYS"
    local TRAJ="$SDIR/rep${REP}/md.xtc"
    local TOP="$SDIR/solute.gro"
    local OUT="$RDIR/out/${SYS}_rep${REP}"
    local FLAGS=""

    [ "$SYS" = "naag"    ] && FLAGS="--naag"
    [ "$SYS" = "zno541"  ] && FLAGS="--zno --site 541"
    [ "$SYS" = "tern541" ] && FLAGS="--naag --zno --site 541"

    echo "[$(date)] START $SYS rep$REP" | tee -a "$LOG"
    $PYTHON "$ANA" \
        --top "$TOP" --traj "$TRAJ" --out "$OUT" \
        --skip-ns $SKIP_NS --sasa-stride $SASA_STRIDE --hbond-stride $HBOND_STRIDE \
        $FLAGS 2>&1 | tee -a "${OUT}.run.log"
    echo "[$(date)] DONE  $SYS rep$REP" | tee -a "$LOG"
}

export -f run_system
export RDIR ANA PYTHON SKIP_NS SASA_STRIDE HBOND_STRIDE LOG

# Run 2 in parallel to keep RAM under control (~2 GB per job)
(run_system apo    1 &
 run_system naag   1 &
 wait)
echo "[$(date)] Batch 1 done" | tee -a "$LOG"

(run_system apo    2 &
 run_system naag   2 &
 wait)
echo "[$(date)] Batch 2 done" | tee -a "$LOG"

(run_system zno541  1 &
 run_system tern541 1 &
 wait)
echo "[$(date)] Batch 3 done" | tee -a "$LOG"

(run_system zno541  2 &
 run_system tern541 2 &
 wait)
echo "[$(date)] Batch 4 done" | tee -a "$LOG"

echo "[$(date)] ALL ANALYSIS COMPLETE" | tee -a "$LOG"
