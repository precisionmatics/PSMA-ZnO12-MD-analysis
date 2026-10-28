#!/usr/bin/env python3
"""
Per-replica analysis of PSMA / NAAG / (ZnO)12 trajectories (solute-only, fitted/centred, e.g. 20 ps per frame).
  python psma_md_analysis.py --top solute.gro --traj md_fit.xtc --out res/tern541_rep1 --site 541 --naag --zno --dt-ps 20 --skip-ns 100
Writes  <out>_timeseries.npz  <out>_summary.json  (all later figures/tables are made from these by summarize.py).
Selections assume the builder's naming: protein residues, NAAG = resids 901-903, ZNO, ZN2 (2x), CAL.
"""
import argparse, json, numpy as np, MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.transformations import NoJump
ap=argparse.ArgumentParser()
ap.add_argument("--top",required=True); ap.add_argument("--traj",required=True); ap.add_argument("--out",required=True)
ap.add_argument("--site",type=int,default=0,help="Trp residue number the cluster was docked to (0 = none)")
ap.add_argument("--naag",action="store_true"); ap.add_argument("--zno",action="store_true")
ap.add_argument("--dt-ps",type=float,default=20.0); ap.add_argument("--skip-ns",type=float,default=100.0,help="equilibration to discard for statistics")
ap.add_argument("--sasa-stride",type=int,default=10)
ap.add_argument("--stop-frame",type=int,default=None); ap.add_argument("--hbond-stride",type=int,default=5)
ap.add_argument("--nprot",type=int,default=10946,help="number of PSMA atoms (they come first in the solute)")
ap.add_argument("--naag-sel",default="bynum 10947:10980"); ap.add_argument("--zno-sel",default="resname ZNO")
ap.add_argument("--zn-sel",default="resname ZN2"); ap.add_argument("--ca-sel",default="resname CAL")
a=ap.parse_args()
u=mda.Universe(a.top,a.traj)
# Fix PBC: prevent atoms from jumping across box boundaries between frames
u.trajectory.add_transformations(NoJump())
n=len(u.trajectory) if a.stop_frame is None else a.stop_frame
t_ns=np.arange(n)*a.dt_ps/1000.0; skip=int(a.skip_ns*1000/a.dt_ps)
P=f"protein and bynum 1:{a.nprot}"
prot=u.select_atoms(P); ca=u.select_atoms(P+" and name CA"); bb=u.select_atoms(P+" and backbone")
naag=u.select_atoms(a.naag_sel) if a.naag else None; zno=u.select_atoms(a.zno_sel) if a.zno else None
zn=u.select_atoms(a.zn_sel); cal=u.select_atoms(a.ca_sel)
heavy=lambda ag: ag.select_atoms("not name H*")
indole=u.select_atoms(f"({P}) and resid {a.site} and name CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2") if a.site else None
prot_heavy=heavy(prot)
# ---- reference frame 0 for alignment
u.trajectory[0]; ref_ca=ca.positions.copy(); ref_all=u.atoms.positions.copy()
S={k:[] for k in ["rmsd_ca","rmsd_bb","rmsd_complex","rmsd_naag","rmsd_zno","rg_prot","zno_rg","zno_rmsd_internal","d_zno_prot_min","d_zno_indole_min","d_zno_indole_com","d_naag_prot_min","zn_zn","ncoord_zn1","ncoord_zn2","naag_zn_min","zno_bond_mean","zno_bond_max","zno_bond_min"]}
zn_lig={}  # distances from each Zn to coordinating atoms
coord_sel=u.select_atoms(f"({P} and (name OD1 OD2 OE1 OE2 NE2 ND1 OG1 OG OH O)) or (bynum 10947:10980 and name O*) or (resname SOL and name OW)")
from MDAnalysis.lib.distances import distance_array, capped_distance
if a.zno: zref=zno.positions.copy()
zero_idx=np.arange(len(zno)) if zno is not None else np.array([])
fit_pos=[]
for i,ts in enumerate(u.trajectory[:n]):
    # fit protein CA to frame 0 (rotation+translation of whole system)
    R,_=align.rotation_matrix(ca.positions-ca.center_of_mass(),ref_ca-ref_ca.mean(0))
    cm=ca.center_of_mass(); u.atoms.translate(-cm); u.atoms.rotate(R); u.atoms.translate(ref_ca.mean(0))
    d=ca.positions-ref_ca; S["rmsd_ca"].append(np.sqrt((d**2).sum(1).mean())/10)
    d=bb.positions-ref_all[bb.indices]; S["rmsd_bb"].append(np.sqrt((d**2).sum(1).mean())/10)
    heavy_idx=u.select_atoms("not name H* and not resname SOL").indices if i==0 and not hasattr(u,"_hi") else getattr(u,"_hi")
    u._hi=heavy_idx; d=u.atoms.positions[heavy_idx]-ref_all[heavy_idx]; S["rmsd_complex"].append(np.sqrt((d**2).sum(1).mean())/10)
    S["rg_prot"].append(prot.radius_of_gyration()/10)
    if a.naag:
        ng=heavy(naag); S["rmsd_naag"].append(np.sqrt(((ng.positions-ref_all[ng.indices])**2).sum(1).mean())/10)
        S["d_naag_prot_min"].append(distance_array(ng.positions,prot_heavy.positions).min()/10)
        S["naag_zn_min"].append(distance_array(ng.select_atoms("name O*").positions,zn.positions).min()/10)
    if a.zno:
        S["rmsd_zno"].append(np.sqrt(((zno.positions-ref_all[zno.indices])**2).sum(1).mean())/10)
        S["zno_rg"].append(zno.radius_of_gyration()/10)
        S["zno_rmsd_internal"].append(align.rotation_matrix(zno.positions-zno.centroid(),zref-zref.mean(0))[1]/10/np.sqrt(len(zno))*1.0)
        S["d_zno_prot_min"].append(distance_array(zno.positions,prot_heavy.positions).min()/10)
        if a.site:
            S["d_zno_indole_min"].append(distance_array(zno.positions,indole.positions).min()/10)
            S["d_zno_indole_com"].append(np.linalg.norm(zno.center_of_geometry()-indole.center_of_geometry())/10)
        # Zn-O bonds of the cage (distance < 2.3 A)
        dd=distance_array(zno.select_atoms("name Z*").positions,zno.select_atoms("name O*").positions); b=dd[dd<2.3]/10
        S["zno_bond_mean"].append(b.mean() if len(b) else np.nan); S["zno_bond_max"].append(b.max() if len(b) else np.nan); S["zno_bond_min"].append(b.min() if len(b) else np.nan)
    if len(zn)>=2:
        S["zn_zn"].append(np.linalg.norm(zn.positions[0]-zn.positions[1])/10)
        dz=distance_array(zn.positions,coord_sel.positions); S["ncoord_zn1"].append(int((dz[0]<2.6).sum())); S["ncoord_zn2"].append(int((dz[1]<2.6).sum()))
ts_arr={k:np.array(v) for k,v in S.items() if len(v)}
# ---- RMSF (per-residue CA) over the production window with 4-block errors
u.trajectory[0]; coords=[]
for ts in u.trajectory[skip:n]: 
    R,_=align.rotation_matrix(ca.positions-ca.center_of_mass(),ref_ca-ref_ca.mean(0)); c=ca.positions-ca.center_of_mass(); coords.append(c@R.T)
C=np.array(coords); 
def rmsf(C): avg=C.mean(0); return np.sqrt(((C-avg)**2).sum(2).mean(0))/10
rm=rmsf(C); blocks=np.array_split(np.arange(len(C)),4); rb=np.array([rmsf(C[b]) for b in blocks])
# ---- dynamic cross-correlation matrix (Calpha) from the same fitted coordinates (R2: covariance of fluctuations)
dC=C-C.mean(0); cov=np.einsum("tia,tja->ij",dC,dC)/len(dC); dccm=cov/np.sqrt(np.outer(np.diag(cov),np.diag(cov)))
ts_arr["dccm_ca"]=dccm.astype(np.float32); ts_arr["cov_ca_nm2"]=(cov/100.0).astype(np.float32)
# ---- per-frame SASA (Shrake-Rupley via freesasa, solute atoms only incl. NAAG/ZnO/metals) for every Trp + total protein
import freesasa
RAD={"C":1.70,"N":1.55,"O":1.52,"S":1.80,"ZN":1.39,"CA":1.80,"Z":1.39}
solute=u.select_atoms("not name H* and not resname SOL")
rad=np.array([RAD.get("ZN" if nm.upper().startswith("ZN") and rn in("ZN2","ZNO") else ("CA" if rn=="CAL" else nm[0].upper()),1.7) for nm,rn in zip(solute.names,solute.resnames)])
is_prot=np.isin(solute.indices,prot.indices)
trp=u.select_atoms(f"({P}) and resname TRP and not name H*")
trp_res=sorted(set(trp.resids)); idx_by_trp={r:np.where(np.isin(solute.indices,trp.select_atoms(f"resid {r}").indices))[0] for r in trp_res}
sasa_t=[];sasa_trp={r:[] for r in trp_res}; sframes=list(range(skip,n,a.sasa_stride))
for i in sframes:
    u.trajectory[i]; res=freesasa.calcCoord(solute.positions.ravel().tolist(),rad.tolist()); ar=np.array([res.atomArea(k) for k in range(len(rad))])
    sasa_t.append(ar[is_prot].sum()/100.0)
    for r in trp_res: sasa_trp[r].append(ar[idx_by_trp[r]].sum())
ts_arr["sasa_prot_nm2"]=np.array(sasa_t); ts_arr["sasa_frames"]=np.array(sframes)
for r in trp_res: ts_arr[f"sasa_trp{r}_A2"]=np.array(sasa_trp[r])
ts_arr["rmsf_ca"]=rm; ts_arr["rmsf_ca_blockstd"]=rb.std(0,ddof=1); ts_arr["rmsf_resids"]=ca.resids
ts_arr["t_ns"]=t_ns
np.savez(a.out+"_timeseries.npz",**ts_arr)
# ---- statistics with block averaging / autocorrelation-based effective sample size
def stats(x):
    x=np.asarray(x,float); x=x[~np.isnan(x)]
    if len(x)<10: return {}
    m=x.mean(); xc=x-m; var=xc.var()
    # integrated autocorrelation time via FFT, Sokal windowing
    f=np.fft.rfft(xc,n=2*len(xc)); ac=np.fft.irfft(f*np.conj(f))[:len(xc)]; ac/=ac[0] if ac[0]>0 else 1
    tau=1.0
    for W in range(1,len(ac)):
        tau=1+2*ac[1:W+1].sum()
        if W>=5*tau: break
    tau=max(tau,1.0); neff=len(x)/tau
    # block-averaging SEM (5 blocks, 10 blocks -> report max)
    sems=[]
    for nb in (5,10,20):
        bm=np.array([b.mean() for b in np.array_split(x,nb)]); sems.append(bm.std(ddof=1)/np.sqrt(nb))
    q=np.percentile(x,[25,50,75])
    return dict(n=int(len(x)),mean=float(m),sd=float(x.std(ddof=1)),median=float(q[1]),p25=float(q[0]),p75=float(q[2]),min=float(x.min()),max=float(x.max()),
                tau_frames=float(tau),n_eff=float(neff),sem_naive=float(x.std(ddof=1)/np.sqrt(len(x))),sem_corr=float(x.std(ddof=1)/np.sqrt(neff)),sem_block=float(max(sems)))
summ={"meta":dict(n_frames=int(n),dt_ps=a.dt_ps,skip_ns=a.skip_ns,site=a.site)}
for k,v in ts_arr.items():
    if k.startswith(("rmsf","dccm","cov_","sasa_frames")) or k in("t_ns",): continue
    summ[k]=stats(v if k.startswith("sasa") else v[skip:])
# ---- binding events for the cluster (hysteresis: bound if d<0.45 nm, unbound if >0.8 nm)
def events(d,lo=0.45,hi=0.80,dt_ns=a.dt_ps/1000):
    st=d[0]<lo; segs=[]; start=0
    for i,x in enumerate(d):
        if st and x>hi: segs.append(("bound",start,i)); st=False; start=i
        elif (not st) and x<lo: segs.append(("unbound",start,i)); st=True; start=i
    segs.append(("bound" if st else "unbound",start,len(d)))
    b=[(e-s)*dt_ns for t,s,e in segs if t=="bound"]; ub=[(e-s)*dt_ns for t,s,e in segs if t=="unbound"]
    return dict(n_dissociations=int(sum(1 for t,s,e in segs if t=="unbound" and s>0)),n_rebinding=int(sum(1 for t,s,e in segs if t=="bound" and s>0)),
                bound_fraction=float((np.array(d)<lo).mean()),bound_residence_ns=b,unbound_ns=ub,median_bound_ns=float(np.median(b)) if b else 0.0)
if a.zno:
    summ["cluster_events"]=events(ts_arr["d_zno_prot_min"][skip:])
    for thr in (0.2,0.3,0.5,1.0,1.5,2.0):
        if a.site: summ.setdefault("fraction_within_nm_of_indole_com",{})[str(thr)]=float((ts_arr["d_zno_indole_com"][skip:]<thr).mean())
    if a.site:
        h,e=np.histogram(ts_arr["d_zno_indole_com"][skip:],bins=np.arange(0,6.01,0.05),density=True); summ["P_r_indole_com"]=dict(r_nm=((e[1:]+e[:-1])/2).tolist(),P=h.tolist())
json.dump(summ,open(a.out+"_summary.json","w"),indent=1)
print("done",a.out,"frames",n)
