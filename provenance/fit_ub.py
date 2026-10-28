"""Force-matching of an all-bonded (Urey-Bradley) internal force field for the (ZnO)12 cage: all intra-cage non-bonded pairs excluded.
E = sum 1/2 kb (r-r0)^2 + sum 1/2 k_th (th-th0)^2 + sum 1/2 k_ub (r13-r13_0)^2, equilibrium values = DFT minimum. Linear in k."""
import numpy as np, itertools, networkx as nx, json, math
HA2KJ=2625.4996; BOHR=0.052917721; FkJ=HA2KJ/BOHR
D=np.load("fm_data.npz",allow_pickle=True); X0=D["X0"]/10.0; el=list(D["el"]); Xs=D["X"]/10.0; Es=D["E"]*HA2KJ; Fs=D["F"]*FkJ
n=24; d0=np.linalg.norm(X0[:,None]-X0[None],axis=2)
G=nx.Graph(); [G.add_edge(i,j) for i in range(n) for j in range(i+1,n) if el[i]!=el[j] and d0[i,j]<0.23]
sq=set()
for c in nx.simple_cycles(G,length_bound=4):
    if len(c)==4: sq|={frozenset((c[i],c[(i+1)%4])) for i in range(4)}
bonds=[tuple(b) for b in G.edges]; bcls=[0 if frozenset(b) in sq else 1 for b in bonds]
angles=[];acls=[]
for c in range(n):
    for a,b in itertools.combinations(list(G[c]),2):
        angles.append((a,c,b)); acls.append((0 if el[c]=="Zn" else 1)*2+(0 if (frozenset((a,c)) in sq and frozenset((c,b)) in sq) else 1))
NB,NA=2,4; NUB=4
def terms(X):
    """return list of 'unit' generalised-coordinate gradients: for each class a (nclass,72) matrix G s.t. dE/dx = sum_k k * (q-q0) dq/dx"""
    Fb=np.zeros((NB,n,3)); Fa=np.zeros((NA,n,3)); Fu=np.zeros((NUB,n,3)); E=np.zeros(NB+NA+NUB)
    for (i,j),c in zip(bonds,bcls):
        r=X[i]-X[j]; rr=np.linalg.norm(r); r0=np.linalg.norm(X0[i]-X0[j]); dq=rr-r0; E[c]+=0.5*dq*dq; f=-dq*r/rr; Fb[c,i]+=f; Fb[c,j]-=f
    for (a,c,b),k in zip(angles,acls):
        def th(Y): v1=Y[a]-Y[c]; v2=Y[b]-Y[c]; return math.acos(np.clip(v1@v2/np.linalg.norm(v1)/np.linalg.norm(v2),-1,1))
        v1=X[a]-X[c]; v2=X[b]-X[c]; l1=np.linalg.norm(v1); l2=np.linalg.norm(v2); cs=np.clip(v1@v2/(l1*l2),-1,1); t=math.acos(cs); dq=t-th(X0); E[NB+k]+=0.5*dq*dq
        st=math.sqrt(max(1-cs*cs,1e-12)); ga=-(1/st)*(v2/(l1*l2)-cs*v1/l1**2); gb=-(1/st)*(v1/(l1*l2)-cs*v2/l2**2)
        Fa_=-dq*ga; Fb_=-dq*gb; Fa[k,a]+=Fa_; Fa[k,b]+=Fb_; Fa[k,c]-=Fa_+Fb_
        # UB 1-3 distance a..b ; class = angle class
        r=X[a]-X[b]; rr=np.linalg.norm(r); r0=np.linalg.norm(X0[a]-X0[b]); dq=rr-r0; E[NB+NA+k]+=0.5*dq*dq; f=-dq*r/rr; Fu[k,a]+=f; Fu[k,b]-=f
    return np.concatenate([Fb,Fa,Fu]).reshape(NB+NA+NUB,-1),E
cols=[];Evec=[];Fvec=[]
for X,E,F in zip(Xs,Es,Fs):
    T,e=terms(X); cols.append(T.T); Evec.append(e); Fvec.append(F.ravel())
A=np.vstack(cols); b=np.concatenate(Fvec)
def fit(A,b,use):
    from scipy.optimize import nnls
    k,res=nnls(A[:,use],b); full=np.zeros(A.shape[1]); full[use]=k; return full
names=["kb_tetra-hex","kb_hex-hex","kth_Zn_tetra","kth_Zn_hex","kth_O_tetra","kth_O_hex","kub_Zn_tetra","kub_Zn_hex","kub_O_tetra","kub_O_hex"]
def r2(k,rows=None):
    pr=A@k; return 1-np.sum((pr-b)**2)/np.sum(b**2), np.sqrt(np.mean((pr-b)**2))
for label,use in (("bonds+angles only",list(range(6))),("bonds+angles+Urey-Bradley",list(range(10)))):
    k=fit(A,b,use); R2,rm=r2(k)
    # leave-one-out
    cv=[]
    for h in range(len(Xs)):
        m=np.ones(len(b),bool); m[h*72:(h+1)*72]=False; kk=fit(A[m],b[m],use); pr=A[~m]@kk; cv.append(1-np.sum((pr-b[~m])**2)/np.sum(b[~m]**2))
    print("== %s:  R2=%.3f  force RMSD=%.1f kJ/mol/nm (DFT RMS %.1f) | LOO R2 mean %.3f (min %.3f)"%(label,R2,rm,np.sqrt(np.mean(b**2)),np.mean(cv),np.min(cv)))
    if len(use)==10: kbest=k
print("force constants (kJ/mol/nm2 for bond & UB, kJ/mol/rad2 for angle):")
for nm,v in zip(names,kbest): print("  %-14s %10.1f"%(nm,v))
# energy check with best
Etot=np.array([e@kbest for e in Evec]); off=np.mean(Es-Etot); print("energy RMSD vs DFT: %.2f kJ/mol (DFT spread %.1f)"%(np.sqrt(np.mean((Etot+off-Es)**2)),Es.max()-Es.min()))
json.dump(dict(zip(names,kbest.tolist())),open("ff_ub_params.json","w"),indent=1)

# ---------- diagnostic: add intramolecular Coulomb (q=+-0.844) per topological shell as free-scale columns
KE=138.935458; q=np.array([0.8441 if e=="Zn" else -0.8441 for e in el])
sp=dict(nx.all_pairs_shortest_path_length(G))
def coul_cols(X):
    out=np.zeros((3,n,3))
    for i in range(n):
        for j in range(i+1,n):
            s=sp[i][j]
            if s<2: continue
            sh=0 if s==2 else (1 if s==3 else 2)
            r=X[i]-X[j]; rr=np.linalg.norm(r); f=KE*q[i]*q[j]/rr**2*r/rr; out[sh,i]+=f; out[sh,j]-=f
    return out.reshape(3,-1).T
Cc=np.vstack([coul_cols(X) for X in Xs]); A2=np.hstack([A,Cc])
from scipy.optimize import lsq_linear
lo=np.concatenate([np.zeros(10),-2*np.ones(3)]); hi=np.concatenate([np.full(10,1e6),2*np.ones(3)])
sol=lsq_linear(A2,b,bounds=(lo,hi)); pr=A2@sol.x
print("\nDIAGNOSTIC (UB + free-scale Coulomb per shell): R2=%.3f  force RMSD=%.1f"%(1-np.sum((pr-b)**2)/np.sum(b**2),np.sqrt(np.mean((pr-b)**2))))
print("  Coulomb scale factors  1-3: %.3f   1-4: %.3f   1-5+: %.3f   (1.0 = full point-charge Coulomb, 0 = none)"%tuple(sol.x[10:]))
print("  UB-model constants:",np.round(sol.x[:10],0))
