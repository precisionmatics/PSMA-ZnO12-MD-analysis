"""Force-matching reference data: DFT (B3LYP/6-31G*) energies+forces of randomly distorted (ZnO)12 cages."""
import numpy as np, sys, time
from pyscf import gto, dft, lib
lib.num_threads(8)
n=int(sys.argv[1]); sig=float(sys.argv[2]) if len(sys.argv)>2 else 0.05
L=open("cage_opt.xyz").read().splitlines()[2:26]; el=[l.split()[0] for l in L]; X0=np.array([[float(v) for v in l.split()[1:4]] for l in L])
rng=np.random.default_rng(42); out={"X":[],"E":[],"F":[]}
mf=None
for k in range(n):
    X=X0+rng.normal(scale=sig,size=X0.shape)
    mol=gto.M(atom=[(e,tuple(x)) for e,x in zip(el,X)],basis="6-31g*",verbose=0,max_memory=6000)
    mf=dft.RKS(mol).density_fit(); mf.xc="b3lyp"; mf.grids.level=3; mf.conv_tol=1e-9; mf.max_cycle=100
    t=time.time(); e=mf.kernel(); g=mf.nuc_grad_method().kernel()
    out["X"].append(X); out["E"].append(e); out["F"].append(-g)       # forces in Ha/bohr
    np.savez("fm_data.npz",X=np.array(out["X"]),E=np.array(out["E"]),F=np.array(out["F"]),X0=X0,el=el)
    print("sample",k,"E=%.6f conv=%s  %.0fs"%(e,mf.converged,time.time()-t),flush=True)
