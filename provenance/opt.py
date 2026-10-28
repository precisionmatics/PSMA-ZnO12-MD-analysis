import numpy as np, time, sys
from pyscf import gto, dft, lib
from pyscf.geomopt.geometric_solver import optimize
lib.num_threads(8)
mol=gto.M(atom="cage_init.xyz",basis="6-31g*",charge=0,spin=0,verbose=4,max_memory=6000,output="opt_scf.log")
mf=dft.RKS(mol).density_fit(); mf.xc="b3lyp"; mf.grids.level=3; mf.conv_tol=1e-8; mf.max_cycle=200
t=time.time(); e=mf.kernel(); print("first SCF E=%.8f  t=%.0fs  converged=%s"%(e,time.time()-t,mf.converged),flush=True)
mol_eq=optimize(mf,maxsteps=100,convergence_set="GAU")
mol_eq.tofile("cage_opt.xyz"); print("OPT DONE",flush=True)
