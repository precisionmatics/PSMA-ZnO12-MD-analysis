#!/bin/bash
# Make a centred, whole-molecule, protein-fitted SOLUTE trajectory for analysis / MM-PBSA.
#     ./server/postprocess.sh <case> <replica>
# creates systems/<case>/rep<R>/{solute.tpr, solute.gro, md_fit.xtc}
set -euo pipefail
CASE=$1; REP=$2; GMX=${GMX:-gmx}
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd); cd "$ROOT/systems/$CASE"; R=rep$REP
echo "Solute" | $GMX convert-tpr -s $R/md.tpr -n index.ndx -o $R/solute.tpr
# 1) make molecules whole & centre the protein in the box
printf "Protein\nSystem\n" | $GMX trjconv -s $R/solute.tpr -f $R/md.xtc -o $R/md_c.xtc -pbc mol -ur compact -center
# 2) remove protein rotation/translation (least-squares fit on backbone)
printf "Backbone\nSystem\n" | $GMX trjconv -s $R/solute.tpr -f $R/md_c.xtc -o $R/md_fit.xtc -fit rot+trans
$GMX editconf -f $R/solute.tpr -o $R/solute.gro      # topology/atom names for MDAnalysis
rm -f $R/md_c.xtc
echo "done: $R/md_fit.xtc  $R/solute.gro  $R/solute.tpr"
