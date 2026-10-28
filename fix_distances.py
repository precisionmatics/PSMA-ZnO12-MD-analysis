#!/usr/bin/env python3
"""Recompute intermolecular distances WITHOUT NoJump (use PBC minimum-image instead).
Also recompute RMSF properly by centering on protein COM at each frame before NoJump."""
import numpy as np, json
from pathlib import Path
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array

RDIR = Path("/home/stalin/Desktop/Zinc/work/results")
OUT  = RDIR / "out"
SKIP_NS = 100.0; DT_PS = 20.0
skip = int(SKIP_NS * 1000 / DT_PS)

CONFIGS = {
    "apo":    dict(naag=False, zno=False, site=0),
    "naag":   dict(naag=True,  zno=False, site=0),
    "zno541": dict(naag=False, zno=True,  site=541),
    "tern541":dict(naag=True,  zno=True,  site=541),
}

def load(sys, rep):
    top  = str(RDIR/f"systems/{sys}/solute.gro")
    traj = str(RDIR/f"systems/{sys}/rep{rep}/md.xtc")
    return mda.Universe(top, traj)   # NO NoJump — use raw coordinates

for sys, cfg in CONFIGS.items():
    for rep in (1, 2):
        print(f"\n=== {sys} rep{rep} ===")
        u = load(sys, rep)
        n = len(u.trajectory)
        nprot = 10946

        P = f"protein and bynum 1:{nprot}"
        prot     = u.select_atoms(P)
        ca       = u.select_atoms(P + " and name CA")
        prot_hvy = u.select_atoms(f"({P}) and not name H*")
        zno      = u.select_atoms("resname ZNO")  if cfg["zno"]  else None
        naag     = u.select_atoms(f"bynum 10947:10980") if cfg["naag"] else None
        indole   = u.select_atoms(f"({P}) and resid {cfg['site']} and name CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2") if cfg["site"] else None
        zn       = u.select_atoms("resname ZN2")

        d_zno_prot = []; d_zno_ind = []; d_zno_ind_com = []
        d_naag_prot = []; naag_zn_min = []
        ncoord1 = []; ncoord2 = []; zn_zn = []
        coord_sel = u.select_atoms(
            f"({P} and (name OD1 OD2 OE1 OE2 NE2 ND1 OG1 OG OH O)) "
            f"or (bynum 10947:10980 and name O*)")

        for i, ts in enumerate(u.trajectory):
            box = ts.dimensions  # [a,b,c,α,β,γ]

            if cfg["zno"] and len(zno) > 0:
                d = distance_array(zno.positions, prot_hvy.positions, box=box)
                d_zno_prot.append(d.min() / 10)  # nm
                if indole is not None and len(indole) > 0:
                    d2 = distance_array(zno.positions, indole.positions, box=box)
                    d_zno_ind.append(d2.min() / 10)
                    # CoM-to-CoM distance with minimum image
                    dv = zno.center_of_geometry() - indole.center_of_geometry()
                    # apply minimum image to displacement vector
                    # for orthogonal box: box = [a,b,c,90,90,90]
                    a,b,c = box[0], box[1], box[2]
                    dv[0] -= a * np.round(dv[0]/a)
                    dv[1] -= b * np.round(dv[1]/b)
                    dv[2] -= c * np.round(dv[2]/c)
                    d_zno_ind_com.append(np.linalg.norm(dv) / 10)

            if cfg["naag"] and len(naag) > 0:
                n_hvy = naag.select_atoms("not name H*")
                d = distance_array(n_hvy.positions, prot_hvy.positions, box=box)
                d_naag_prot.append(d.min() / 10)
                if len(zn) > 0:
                    d3 = distance_array(naag.select_atoms("name O*").positions,
                                        zn.positions, box=box)
                    naag_zn_min.append(d3.min() / 10)

            if len(zn) >= 2:
                dv = zn.positions[0] - zn.positions[1]
                a,b,c = box[0],box[1],box[2]
                dv[0] -= a*np.round(dv[0]/a); dv[1] -= b*np.round(dv[1]/b); dv[2] -= c*np.round(dv[2]/c)
                zn_zn.append(np.linalg.norm(dv)/10)
                dz = distance_array(zn.positions, coord_sel.positions, box=box)
                ncoord1.append(int((dz[0]<2.6).sum()))
                ncoord2.append(int((dz[1]<2.6).sum()))

        # Save corrected time series back into the existing npz
        npz_path = OUT / f"{sys}_rep{rep}_timeseries.npz"
        d_old = dict(np.load(npz_path, allow_pickle=True))
        if d_zno_prot:    d_old["d_zno_prot_min"]   = np.array(d_zno_prot)
        if d_zno_ind:     d_old["d_zno_indole_min"]  = np.array(d_zno_ind)
        if d_zno_ind_com: d_old["d_zno_indole_com"]  = np.array(d_zno_ind_com)
        if d_naag_prot:   d_old["d_naag_prot_min"]   = np.array(d_naag_prot)
        if naag_zn_min:   d_old["naag_zn_min"]        = np.array(naag_zn_min)
        if ncoord1:       d_old["ncoord_zn1"]          = np.array(ncoord1)
        if ncoord2:       d_old["ncoord_zn2"]          = np.array(ncoord2)
        if zn_zn:         d_old["zn_zn"]               = np.array(zn_zn)
        np.savez(npz_path, **d_old)

        # Quick report
        prod_slice = slice(skip, None)
        if d_zno_prot:
            arr = np.array(d_zno_prot)[prod_slice]
            print(f"  d_zno_prot_min : {arr.mean():.3f} ± {arr.std():.3f} nm  (min {arr.min():.3f})")
        if d_zno_ind_com:
            arr = np.array(d_zno_ind_com)[prod_slice]
            print(f"  d_zno_ind_com  : {arr.mean():.3f} ± {arr.std():.3f} nm")
            # rebuild P(r) and cluster events
            bins = np.arange(0, 6.01, 0.05)
            h, e = np.histogram(arr, bins=bins, density=True)
            d_old["P_r_indole_com_fixed"] = np.column_stack([(e[:-1]+e[1:])/2, h])
            bound = (arr < 0.45).mean()
            print(f"  bound_fraction (<0.45nm): {bound:.3f}")
        if d_naag_prot:
            arr = np.array(d_naag_prot)[prod_slice]
            print(f"  d_naag_prot    : {arr.mean():.3f} ± {arr.std():.3f} nm")
        if zn_zn:
            arr = np.array(zn_zn)[prod_slice]
            print(f"  Zn-Zn dist     : {arr.mean():.4f} ± {arr.std():.4f} nm")
        if ncoord1:
            print(f"  ncoord_zn1     : {np.array(ncoord1)[prod_slice].mean():.2f}")
            print(f"  ncoord_zn2     : {np.array(ncoord2)[prod_slice].mean():.2f}")

        np.savez(npz_path, **d_old)
        print(f"  npz updated: {npz_path.name}")

print("\nDone.")
