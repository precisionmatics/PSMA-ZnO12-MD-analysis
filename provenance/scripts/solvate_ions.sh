#!/bin/bash
# usage: solvate_ions.sh CASE   (run from 04_build; needs gmx on PATH)
set -e; C=$1; mkdir -p systems; cd systems/$C
ln -sf ../../charmm36-jul2021.ff . ; for f in protein.itp naag.itp zno.itp zno_types.itp posre.itp posre_naag.itp; do ln -sf ../../$f .; done
echo 0 | gmx editconf -f dry.gro -o box.gro -princ -bt triclinic -d 1.0 -c > editconf.log 2>&1
cp topol_core.top topol.top
gmx solvate -cp box.gro -cs spc216.gro -p topol.top -o solv.gro > solvate.log 2>&1
NTOT=$(sed -n 2p solv.gro); NDRY=$(sed -n 2p dry.gro)
python3 - <<PY
n0=$NDRY; n=$NTOT
open("added_sol.ndx","w").write("[ SOL ]\n"+"\n".join(" ".join(str(i) for i in range(s,min(s+15,n+1))) for s in range(n0+1,n+1,15))+"\n")
PY
gmx grompp -f ../../mdp/ions.mdp -c solv.gro -p topol.top -o ions.tpr -maxwarn 2 > grompp_ions.log 2>&1
echo SOL | gmx genion -s ions.tpr -n added_sol.ndx -o solv_ions.gro -p topol.top -pname SOD -nname CLA -neutral -conc 0.15 > genion.log 2>&1
cp topol.top topol_core_ions.top; cat topol_core_ions.top > topol.top; echo >> topol.top; cat intermol.txt >> topol.top
echo "$C: dry $NDRY atoms -> solvated $NTOT atoms; box: $(tail -1 solv_ions.gro)"
grep -E "^(SOL|SOD|CLA)" topol.top | tr '\n' ' '; echo
