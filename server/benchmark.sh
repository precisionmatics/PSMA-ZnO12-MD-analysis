#!/bin/bash
# 2-minute speed test on YOUR server. Run once, send me the printed "ns/day" line.
#   usage: ./benchmark.sh <case folder with solv_ions.gro topol.top index.ndx>   e.g.  ./benchmark.sh tern541
set -e; GMX=${GMX:-gmx}; C=$1; ROOT=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd); MDP=$ROOT/mdp; STEPS=${STEPS:-20000}; NT=${NT:-""}; cd "$ROOT/systems/$C"; mkdir -p bench
$GMX grompp -f $MDP/em.mdp -c solv_ions.gro -r solv_ions.gro -p topol.top -n index.ndx -o bench/em.tpr -maxwarn 1 >/dev/null 2>&1
$GMX mdrun -deffnm bench/em -nsteps 500 >/dev/null 2>&1
sed "s/nsteps.*/nsteps = $STEPS/; s/gen_seed.*/gen_seed = 1/" $MDP/nvt.mdp > bench/b.mdp
$GMX grompp -f bench/b.mdp -c bench/em.gro -r bench/em.gro -p topol.top -n index.ndx -o bench/b.tpr -maxwarn 1 >/dev/null 2>&1
$GMX mdrun -deffnm bench/b -nsteps $STEPS -resethway -noconfout $NT >/dev/null 2>&1
grep -E "Performance|Finished" bench/b.log; echo "GPU/CPU info:"; grep -E "GPU|Hardware|Running on" bench/b.log | head -4
