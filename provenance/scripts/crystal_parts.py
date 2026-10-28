"""Build crystal-water (with H), Zn2+, Ca2+ pieces in the protein frame (nm) and the metal-coordination restraint list."""
import numpy as np
P="../02_protein/psma_wt_heavy_renum.pdb"
def read_pdb_atoms(fn):
    out=[]
    for l in open(fn):
        if l.startswith(("ATOM","HETATM")):
            out.append(dict(rec=l[:6].strip(),name=l[12:16].strip(),res=l[17:20].strip(),chain=l[21],resid=int(l[22:26]),xyz=np.array([float(l[30:38]),float(l[38:46]),float(l[46:54])])/10.0))
    return out
def crystal_parts():
    A=read_pdb_atoms(P)
    prot=[a for a in A if a["rec"]=="ATOM"]
    zn=[a for a in A if a["res"]=="ZN"]; ca=[a for a in A if a["res"]=="CA" and a["rec"]=="HETATM"]
    wat=[a for a in A if a["res"]=="HOH"]
    return prot,zn,ca,wat
def orient_waters(wat,heavy,metals):
    """return list of (O,H1,H2) coordinates (nm). H-bond-aware; metal-bound waters point H away from metal."""
    rng=np.random.default_rng(1)
    ho=0.09572; ang=np.radians(104.52)/2
    heavy=np.array(heavy); metals=np.array(metals); owat=np.array([w["xyz"] for w in wat])
    res=[]
    for i,w in enumerate(wat):
        o=w["xyz"]
        dm=np.linalg.norm(metals-o,axis=1) if len(metals) else np.array([9.])
        near_m=metals[dm<0.27]
        d=np.linalg.norm(heavy-o,axis=1); cand=[(dd,h) for dd,h in zip(d,heavy) if 0.24<dd<0.34]
        cand.sort(key=lambda t:t[0])
        if len(near_m):
            b=o-near_m.mean(0); b/=np.linalg.norm(b)
            tgt=cand[0][1]-o if cand else None
        else:
            tgt=cand[0][1]-o if cand else None
            b=None
        if b is None:
            # bisector away from the 'average acceptor/donor' direction
            if len(cand)>=2:
                u1=(cand[0][1]-o); u2=(cand[1][1]-o); u1/=np.linalg.norm(u1); u2/=np.linalg.norm(u2)
                bis=u1+u2
                if np.linalg.norm(bis)>0.3:
                    b=bis/np.linalg.norm(bis); pl=u1-(u1@b)*b; pl/=np.linalg.norm(pl)
                    h1=o+ho*(np.cos(ang)*b+np.sin(ang)*pl); h2=o+ho*(np.cos(ang)*b-np.sin(ang)*pl)
                    res.append((o,h1,h2)); continue
            b=rng.normal(size=3); b/=np.linalg.norm(b)
        # build H's around bisector b, in a plane containing b and (tgt if available else random)
        ref=(tgt/np.linalg.norm(tgt)) if tgt is not None else rng.normal(size=3)
        perp=ref-(ref@b)*b
        if np.linalg.norm(perp)<1e-6: perp=np.cross(b,[1,0,0])
        perp/=np.linalg.norm(perp)
        h1=o+ho*(np.cos(ang)*b+np.sin(ang)*perp); h2=o+ho*(np.cos(ang)*b-np.sin(ang)*perp)
        res.append((o,h1,h2))
    return res

def refine_waters(wat_hs,heavy,acceptors,metals,n_try=400,seed=3):
    """For non-metal-bound waters pick, from random orientations plus the initial one, the one with fewest clashes / most H-bonds."""
    rng=np.random.default_rng(seed); ho=0.09572; ang=np.radians(104.52)/2
    heavy=np.array(heavy); acc=np.array(acceptors); metals=np.array(metals)
    out=[]
    for o,h1,h2 in wat_hs:
        if len(metals) and np.min(np.linalg.norm(metals-o,axis=1))<0.27: out.append((o,h1,h2)); continue
        nb=heavy[np.linalg.norm(heavy-o,axis=1)<0.6]; na=acc[np.linalg.norm(acc-o,axis=1)<0.45]
        def score(h):
            s=0.0
            for x in h:
                if len(nb):
                    d=np.linalg.norm(nb-x,axis=1); s+=np.sum(np.clip(0.19-d,0,None))*200
                if len(na):
                    d=np.linalg.norm(na-x,axis=1); s-=np.sum((d>0.16)&(d<0.23))*1.0
            return s
        best=(score((h1,h2)),(h1,h2))
        for _ in range(n_try):
            b=rng.normal(size=3); b/=np.linalg.norm(b); r=rng.normal(size=3); pl=r-(r@b)*b; pl/=np.linalg.norm(pl)
            a=o+ho*(np.cos(ang)*b+np.sin(ang)*pl); c=o+ho*(np.cos(ang)*b-np.sin(ang)*pl)
            sc=score((a,c))
            if sc<best[0]: best=(sc,(a,c))
        out.append((o,)+tuple(best[1]))
    return out
