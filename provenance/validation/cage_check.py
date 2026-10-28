import numpy as np, MDAnalysis as mda, networkx as nx, sys
from MDAnalysis.analysis import align
u=mda.Universe("npt.gro","md.xtc"); cage=u.select_atoms("resname ZNO"); el=[a.name[0] if a.name.startswith("O") else "Zn" for a in cage]
ref=np.array([[float(v) for v in l.split()[1:4]] for l in open("../../01_zno12/zno_ref_final.xyz").read().splitlines()[2:26]])
d0=np.linalg.norm(ref[:,None]-ref[None],axis=2); G=nx.Graph(); [G.add_edge(i,j) for i in range(24) for j in range(i+1,24) if el[i]!=el[j] and d0[i,j]<2.3]
sq=set()
for c in nx.simple_cycles(G,length_bound=4):
    if len(c)==4: sq|={frozenset((c[i],c[(i+1)%4])) for i in range(4)}
bt=[(i,j) for i,j in G.edges if frozenset((i,j)) in sq]; bh=[(i,j) for i,j in G.edges if frozenset((i,j)) not in sq]
T=[];B_t=[];B_h=[];RG=[];RMS=[];MAXB=[];MINB=[];DIST=[]
for ts in u.trajectory:
    x=cage.positions.copy(); bx=ts.dimensions[:3]; dd=x-x[0]; dd-=bx*np.round(dd/bx); x=x[0]+dd; 
    B_t.append(np.mean([np.linalg.norm(x[i]-x[j]) for i,j in bt])); B_h.append(np.mean([np.linalg.norm(x[i]-x[j]) for i,j in bh]))
    allb=[np.linalg.norm(x[i]-x[j]) for i,j in G.edges]; MAXB.append(max(allb)); MINB.append(min(allb))
    RG.append(np.sqrt(np.mean(np.sum((x-x.mean(0))**2,axis=1)))); RMS.append(align.rotation_matrix(x-x.mean(0),ref-ref.mean(0))[1]); T.append(ts.time/1000)
RG0=np.sqrt(np.mean(np.sum((ref-ref.mean(0))**2,axis=1)))
T=np.array(T); s=T>=0.2
print("frames %d, time %.2f-%.2f ns"%(len(T),T[0],T[-1]))
print("DFT reference : tetragon-hexagon bond 1.9333 A | hexagon-hexagon bond 1.8533 A | Rg %.3f A"%RG0)
print("MD (after 0.2 ns, mean +- sd): tetragon-hexagon %.4f +- %.4f | hexagon-hexagon %.4f +- %.4f | Rg %.3f +- %.3f A"%(np.mean(np.array(B_t)[s]),np.std(np.array(B_t)[s]),np.mean(np.array(B_h)[s]),np.std(np.array(B_h)[s]),np.mean(np.array(RG)[s]),np.std(np.array(RG)[s])))
print("longest Zn-O bond ever %.3f A, shortest %.3f A (cage intact if all bonds stay ~1.7-2.2)"%(max(MAXB),min(MINB)))
print("internal RMSD to DFT cage: mean %.3f A, max %.3f A (thermal motion only if << 1 A)"%(np.mean(np.array(RMS)[s]),np.max(RMS)))
