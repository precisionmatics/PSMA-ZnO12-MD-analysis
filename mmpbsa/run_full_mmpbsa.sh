#!/bin/bash
# Run full 251-frame MM-PBSA for zno541 rep1
# After gmx_MMPBSA fails at parsing (BOND overflow), mdout files are preserved
# and we parse them with parse_mmpbsa_mdout.py

set -euo pipefail

WDIR="/home/stalin/Desktop/Zinc/work/results/mmpbsa/zno541_rep1"
SDIR="/home/stalin/Desktop/Zinc/work/results/systems/zno541"
MMPBSA_ENV="/home/stalin/miniforge3/envs/mmpbsa_env"
GMX_BIN="/home/stalin/miniforge3/envs/gmx/bin"
PYTHON="/home/stalin/miniforge3/bin/python3"

export PATH="$MMPBSA_ENV/bin:$GMX_BIN:$PATH"
export CUDA_VISIBLE_DEVICES=""
export OMPI_MCA_mpi_cuda_support=0
export OMPI_MCA_btl="self,vader,tcp"
export OMPI_MCA_btl_vader_single_copy_mechanism="none"

LOG="$WDIR/full_run.log"
echo "[$(date)] Starting full 251-frame MM-PBSA" | tee "$LOG"

cd "$WDIR"

# Run gmx_MMPBSA — expect it to fail at parsing due to BOND overflow,
# but keep_files=2 preserves all intermediate mdout files
gmx_MMPBSA -O \
    -i mmpbsa.in \
    -cs "$SDIR/rep1/solute_md.tpr" \
    -ct md_mmpbsa_dry.xtc \
    -ci "$SDIR/mmpbsa_dry_index.ndx" \
    -cg 19 13 \
    -cp "$SDIR/topol_solute.top" \
    -o mmpbsa_full.dat \
    -eo mmpbsa_full_energies.csv \
    >> "$LOG" 2>&1 && echo "SUCCESS" >> "$LOG" || echo "gmx_MMPBSA exited (expected if BOND overflow)" >> "$LOG"

echo "[$(date)] gmx_MMPBSA phase done, running parser" | tee -a "$LOG"

# Parse mdout files directly (handles BOND=***** via NaN, which cancels in STP)
$PYTHON /home/stalin/Desktop/Zinc/work/results/mmpbsa/parse_mmpbsa_mdout.py "$WDIR" 2>&1 | tee -a "$LOG"

echo "[$(date)] Done. Results in $WDIR/mmpbsa_result.json" | tee -a "$LOG"
