#!/usr/bin/env python3
"""Assemble a PSMA system (dry): protein + [NAAG] + [(ZnO)12] + 2 Zn2+ + Ca2+ + crystal waters. Writes <case>/dry.gro, <case>/topol.top
usage: assemble.py CASE  (apo | naag | zno541 | zno410 | tern541 | tern410)"""
import sys, os, numpy as np, pickle, shutil
sys.path.insert(0,os.path.dirname(__file__))
from crystal_parts import *
case=sys.argv[1]; outd="systems/"+case; os.makedirs(outd,exist_ok=True)
use_naag=case in("naag","tern541","tern410"); site={"zno541":541,"tern541":541,"zno410":410,"tern410":410}.get(case)
KB=50000.0   # kJ mol-1 nm-2 metal-ligand harmonic bonds
# ---------- read pieces
def read_gro(fn):
    L=open(fn).read().splitlines(); n=int(L[1]); at=[]
    for l in L[2:2+n]: at.append((int(l[:5]),l[5:10].strip(),l[10:15].strip(),np.array([float(l[20:28]),float(l[28:36]),float(l[36:44])])))
    return at
prot=read_gro("protein/protein.gro"); naag=read_gro("naag/naag.gro")
nprot=len(prot)
# protein top: (resnr,atomname)->nr
pmap={}; sec=None
for l in open("protein/protein.top"):
    s=l.strip()
    if s.startswith("["): sec=s; continue
    if sec=="[ atoms ]" and s and not s.startswith(";"):
        p=s.split(); pmap[(int(p[2]),p[4])]=int(p[0])
assert len(pmap)==nprot
_,zn,ca,wat=crystal_parts()
water_hs=pickle.load(open("crystal_waters_H.pkl","rb"))
# ---------- ZnO placement (docked pose fitted by Kabsch to cage reference geometry)
def read_xyz(fn): 
    L=open(fn).read().splitlines()[2:]; return [l.split()[0] for l in L if l.strip()],np.array([[float(x) for x in l.split()[1:4]] for l in L if l.strip()])/10.0
def kabsch(P,Q):
    Pc=P.mean(0);Qc=Q.mean(0);H=(P-Pc).T@(Q-Qc);U,S,Vt=np.linalg.svd(H);d=np.sign(np.linalg.det(Vt.T@U.T));D=np.diag([1,1,d]);R=Vt.T@D@U.T
    return R,Pc,Qc
zno_atoms=None
if site:
    el,ref=read_xyz("zno_ref.xyz")             # parameterised (DFT) geometry, atom order = cage_init order
    dock=[]; 
    for l in open(f"../03_docking/zno_W{site}_s1.pdbqt"):
        if l.startswith(("ATOM","HETATM")): dock.append([float(l[30:38]),float(l[38:46]),float(l[46:54])])
        if l.startswith("ENDMDL"): break
    dock=np.array(dock)/10.0; assert len(dock)==24
    R,Pc,Qc=kabsch(ref,dock); placed=(ref-Pc)@R.T+Qc
    print("fit RMSD ref->docked pose: %.4f nm"%np.sqrt(np.mean(np.sum((placed-dock)**2,axis=1))))
    zno_atoms=[("ZNO",f"{e}{i+1}",placed[i]) for i,e in enumerate(el)]
# ---------- build gro atom list
G=[]  # (resname, atomname, xyz)
for (_,rn,an,x) in prot: G.append((rn,an,x))
if use_naag:
    for (_,rn,an,x) in naag: G.append((rn,an,x))
if zno_atoms: G+= zno_atoms
iZN=len(G)+1
for z in zn: G.append(("ZN2","ZN",z["xyz"]))
iCA=len(G)+1
for c in ca: G.append(("CAL","CAL",c["xyz"]))
iW=len(G)+1
for (o,h1,h2) in water_hs: G+=[("SOL","OW",o),("SOL","HW1",h1),("SOL","HW2",h2)]
# gro residue numbers
lines=[]; resn=0; last=None; prev_key=None; 
# residue numbering: protein/naag keep their own; others sequential
def key_of(i):return i
with open(f"{outd}/dry.gro","w") as f:
    f.write(f"PSMA {case}\n{len(G)}\n")
    nres=0; k=0; 
    # assign residue ids
    ids=[]; rid=0; prev=None
    # protein residues from protein.gro
    for (_,rn,an,x),(a0) in zip(prot,prot): ids.append(None)
    out=[]
    ai=0
    for idx,(rn,an,x) in enumerate(G):
        if idx<nprot: rno=prot[idx][0]
        elif use_naag and idx<nprot+len(naag): rno=900+naag[idx-nprot][0]
        else: rno=None
        out.append([rn,an,x,rno])
    # fill rno for rest
    cur=1000; lastkey=None
    for idx,o in enumerate(out):
        if o[3] is None:
            if o[0]=="SOL":
                if (idx-(iW-1))%3==0: cur+=1
                o[3]=cur
            elif o[0]=="ZNO": o[3]=990
            else: cur+=1; o[3]=cur
    for idx,(rn,an,x,rno) in enumerate(out):
        f.write("%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n"%(rno%100000,rn,an[:5],(idx+1)%100000,x[0],x[1],x[2]))
    f.write("%10.5f%10.5f%10.5f\n"%(0,0,0))
# ---------- metal-ligand intermolecular harmonic bonds (crystal distances)
def pa(res,name): return pmap[(res,name)]
def xyz_of(res,name): 
    i=pa(res,name); return prot[i-1][3]
zn1,zn2=iZN,iZN+1
# bridging water O (closest crystal water to both Zn) and Ca-bound water
wO=[w["xyz"] for w in wat]
dbr=[np.linalg.norm(w-zn[0]["xyz"])+np.linalg.norm(w-zn[1]["xyz"]) for w in wO]; ibr=int(np.argmin(dbr))
dcw=[np.linalg.norm(w-ca[0]["xyz"]) for w in wO]; icw=int(np.argmin(dcw))
pairs=[(zn1,("ASP",387,"OD2")),(zn1,("HIS",553,"NE2")),(zn1,("GLU",425,"OE2")),(zn1,("GLU",425,"OE1")),
       (zn2,("ASP",387,"OD1")),(zn2,("ASP",453,"OD2")),(zn2,("HIS",377,"NE2")),
       (iCA,("THR",269,"O")),(iCA,("THR",269,"OG1")),(iCA,("TYR",272,"O")),(iCA,("GLU",433,"OE1")),(iCA,("GLU",433,"OE2")),(iCA,("GLU",436,"OE2"))]
metal_xyz={zn1:zn[0]["xyz"],zn2:zn[1]["xyz"],iCA:ca[0]["xyz"]}
ib=[]
for m,(rn,r,an) in pairs:
    j=pa(r,an); d=np.linalg.norm(metal_xyz[m]-prot[j-1][3]); ib.append((m,j,d))
for m,w in ((zn1,ibr),(zn2,ibr),(iCA,icw)):
    j=iW+3*w; d=np.linalg.norm(metal_xyz[m]-wO[w]); ib.append((m,j,d))
with open(f"{outd}/topol_core.top","w") as t:
    t.write('; PSMA system %s\n#include "charmm36-jul2021.ff/forcefield.itp"\n'%case)
    if site: t.write('#include "zno_types.itp"\n')
    t.write('#include "protein.itp"\n')
    if use_naag: t.write('#include "naag.itp"\n')
    if site: t.write('#include "zno.itp"\n')
    t.write('#include "charmm36-jul2021.ff/tip3p.itp"\n#include "charmm36-jul2021.ff/ions.itp"\n\n[ system ]\nPSMA %s\n\n[ molecules ]\nProtein_chain_A 1\n'%case)
    if use_naag: t.write("NAAG 1\n")
    if site: t.write("ZNO 1\n")
    t.write("ZN2 2\nCAL 1\nSOL %d\n\n"%len(water_hs))
    pass
with open(f"{outd}/intermol.txt","w") as t:
    t.write("[ intermolecular_interactions ]\n[ bonds ]\n; ai aj funct b0(nm) kb(kJ/mol/nm2)   metal-ligand coordination bonds from crystal geometry\n")
    for m,j,d in ib: t.write("%d %d 6 %.4f %.1f\n"%(m,j,d,KB))
print(case,"atoms",len(G),"| Zn idx",zn1,zn2,"Ca",iCA,"first crystal water O",iW,"| bridging water #",ibr+1,"| coordination bonds",len(ib))
print("   bond lengths (nm):",[round(d,3) for _,_,d in ib])
