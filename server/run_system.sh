#!/bin/bash
# Run EM -> NVT -> NPT -> production for ONE system / replica.   Run from the package root:
#     ./server/run_system.sh <case> <replica 1|2|3> [production_ns=300]
# <case> = apo | naag | zno541 | tern541 | zno410 | tern410   (folder systems/<case>)
# Restartable: if interrupted, run the same command again (continues from md.cpt; finished stages are skipped).
# Env: GMX=gmx (binary)  NT="-ntmpi 1 -ntomp 16 -nb gpu -pme gpu -bonded gpu"  QUICK=1 (tiny test run, ~minutes)
set -euo pipefail
CASE=$1; REP=$2; NS=${3:-300}
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd); MDP=$ROOT/mdp; GMX=${GMX:-gmx}; NT=${NT:-""}
NT_EM=$(echo "$NT" | sed -E 's/-pme gpu//; s/-bonded gpu//')
cd "$ROOT/systems/$CASE"; RUN=$PWD/rep$REP; mkdir -p "$RUN"
SEED=$((1000*REP+17)); NSTEPS=$(python3 -c "print(int($NS*1000/0.002))")
EMS=50000; NVTS=100000; NPTS=500000
if [ "${QUICK:-0}" = "1" ]; then EMS=300; NVTS=500; NPTS=500; NSTEPS=1000; fi
if [ ! -f $RUN/em.gro ]; then
  sed "s/nsteps = 50000/nsteps = $EMS/" $MDP/em.mdp > $RUN/em.mdp
  $GMX grompp -f $RUN/em.mdp -c solv_ions.gro -r solv_ions.gro -p topol.top -n index.ndx -o $RUN/em.tpr -maxwarn 1
  $GMX mdrun -deffnm $RUN/em $NT_EM
fi
if [ ! -f $RUN/nvt.gro ]; then
  sed "s/gen_seed = -1/gen_seed = $SEED/; s/nsteps = 100000/nsteps = $NVTS/" $MDP/nvt.mdp > $RUN/nvt.mdp
  $GMX grompp -f $RUN/nvt.mdp -c $RUN/em.gro -r $RUN/em.gro -p topol.top -n index.ndx -o $RUN/nvt.tpr -maxwarn 1
  $GMX mdrun -deffnm $RUN/nvt $NT
fi
if [ ! -f $RUN/npt.gro ]; then
  sed "s/nsteps = 500000/nsteps = $NPTS/" $MDP/npt.mdp > $RUN/npt.mdp
  $GMX grompp -f $RUN/npt.mdp -c $RUN/nvt.gro -r $RUN/nvt.gro -t $RUN/nvt.cpt -p topol.top -n index.ndx -o $RUN/npt.tpr -maxwarn 1
  $GMX mdrun -deffnm $RUN/npt $NT
fi
sed "s/nsteps = 250000000.*/nsteps = $NSTEPS/" $MDP/md.mdp > $RUN/md.mdp
[ "${QUICK:-0}" = "1" ] && sed -i "s/nstxout-compressed = 10000.*/nstxout-compressed = 25/; s/nstenergy = 5000.*/nstenergy = 100/; s/nstlog = 50000/nstlog = 500/" $RUN/md.mdp
if [ ! -f $RUN/md.tpr ]; then
  $GMX grompp -f $RUN/md.mdp -c $RUN/npt.gro -t $RUN/npt.cpt -p topol.top -n index.ndx -o $RUN/md.tpr -maxwarn 1
fi
CPI=""; [ -f $RUN/md.cpt ] && CPI="-cpi $RUN/md.cpt -append"
$GMX mdrun -deffnm $RUN/md $CPI $NT
echo "DONE $CASE rep$REP"
