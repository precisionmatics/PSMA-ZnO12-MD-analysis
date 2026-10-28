"""ESP-fit (Merz-Kollman layers) atomic charges for the DFT-optimised (ZnO)12 cage; also per-atom (unconstrained) and symmetry-constrained fits."""
import numpy as np, sys
from pyscf import gto, dft, lib
lib.num_threads(8)
mol=gto.M(atom="cage_opt.xyz",basis="6-31g*",verbose=3,max_memory=6000,output="esp_scf.log")
mf=dft.RKS(mol).density_fit(); mf.xc="b3lyp"; mf.grids.level=3; mf.conv_tol=1e-9; mf.kernel(); print("SCF",mf.e_tot,mf.converged,flush=True)
dm=mf.make_rdm1(); X=mol.atom_coords(unit="ANG"); Z=[mol.atom_symbol(i) for i in range(mol.natm)]
rad={"Zn":1.39,"O":1.52}; rng=np.random.default_rng(0)
pts=[]
for scale in (1.4,1.6,1.8,2.0):
    for i in range(mol.natm):
        r=rad[Z[i]]*scale; n=int(4*np.pi*r*r*2.0)       # ~2 pts/A^2
        v=rng.normal(size=(n,3)); v/=np.linalg.norm(v,axis=1)[:,None]; p=X[i]+r*v
        d=np.linalg.norm(p[:,None]-X[None],axis=2)
        ok=np.all(d>=np.array([rad[z]*scale-1e-6 for z in Z])[None],axis=1); pts.append(p[ok])
pts=np.vstack(pts); print("ESP points",len(pts),flush=True)
B=pts/0.52917721092      # bohr
V=np.zeros(len(B))
for s in range(0,len(B),150):
    ch=B[s:s+150]
    with mol.with_rinv_origin((0,0,0)): pass
    ints=mol.intor("int1e_grids",grids=ch)           # (ngrid,nao,nao)  <mu|1/|r-R||nu>
    V[s:s+150]=-np.einsum("gij,ij->g",ints,dm)
nucl=np.zeros(len(B))
for i in range(mol.natm):
    nucl+=mol.atom_charge(i)/np.linalg.norm(B-mol.atom_coord(i),axis=1)
V+=nucl
np.savez("esp_data.npz",pts=pts,V=V,X=X,Z=Z)
A=1.0/np.linalg.norm(B[:,None]-mol.atom_coords()[None],axis=2)     # (ng,natm) in 1/bohr
n=mol.natm
def fit(Amat,constraints):
    # minimise |A q - V|^2 with constraints C q = d (Lagrange)
    C,dd=constraints; M=np.block([[Amat.T@Amat,C.T],[C,np.zeros((len(dd),len(dd)))]]); b=np.concatenate([Amat.T@V,dd]); sol=np.linalg.solve(M,b); return sol[:A.shape[1]] if Amat is A else sol[:Amat.shape[1]]
tot=(np.ones((1,n)),np.array([0.0]))
q_free=fit(A,tot)
iz=[i for i in range(n) if Z[i]=="Zn"]; io=[i for i in range(n) if Z[i]=="O"]
Asym=np.stack([A[:,iz].sum(1),A[:,io].sum(1)],axis=1)
sol=np.linalg.solve(np.block([[Asym.T@Asym,np.array([[12],[12]])],[np.array([[12,12]]),np.zeros((1,1))]]),np.concatenate([Asym.T@V,[0.0]]))
qz,qo=sol[0],sol[1]
rms=lambda q:np.sqrt(np.mean((A@q-V)**2))
print("symmetry-constrained: q(Zn)=%.4f q(O)=%.4f  RMS ESP error %.5f a.u."%(qz,qo,rms(np.array([qz if z=="Zn" else qo for z in Z]))))
print("unconstrained per-atom: Zn %.3f..%.3f (mean %.3f)  O %.3f..%.3f (mean %.3f) RMS %.5f"%(q_free[iz].min(),q_free[iz].max(),q_free[iz].mean(),q_free[io].min(),q_free[io].max(),q_free[io].mean(),rms(q_free)))
np.save("charges_sym.npy",np.array([qz if z=="Zn" else qo for z in Z])); np.save("charges_free.npy",q_free)
