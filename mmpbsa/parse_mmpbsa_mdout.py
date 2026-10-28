#!/usr/bin/env python3
"""
Parse gmx_MMPBSA sander mdout files directly, bypassing BOND=***** overflow
caused by CHARMM36 Urey-Bradley / CMAP terms.

In STP (single-trajectory protocol) BOND/UB/IMP/CMAP/ANGLE/DIHED all cancel
exactly (same atoms in COM = REC + LIG), so ΔG_bind depends only on:
  ΔEvdW   = Δ(VDWAALS + 1-4 VDW)
  ΔEelec  = Δ(EEL + 1-4 EEL)
  ΔEPB    = ΔG_polar_solv
  ΔECAVITY + ΔEDISPER = ΔG_nonpolar_solv

Usage: python parse_mmpbsa_mdout.py <workdir>
  workdir contains _GMXMMPBSA_{complex,receptor,ligand}_pb.mdout.0
"""
import numpy as np, sys, os, json, re

def parse_mdout(path):
    """Extract per-frame energy terms from a sander mdout file.
    Returns dict of {term: [val, val, ...]} across all frames."""
    terms = {
        "VDWAALS": [], "EEL": [], "EPB": [],
        "1-4 VDW": [], "1-4 EEL": [],
        "ECAVITY": [], "EDISPER": [],
    }

    # patterns: "KEY = VALUE" where VALUE may be ***** (overflow)
    pat = re.compile(r'(\S[\w\s\-]+?)\s*=\s*([*\d\s\.\+\-eE]+?)(?=\s{2,}|\n)')

    in_frame = False
    frame_data = {}

    with open(path) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        # Frame boundary: NSTEP line
        if "NSTEP" in line and "ENERGY" in line:
            if frame_data:
                for k in terms:
                    if k in frame_data:
                        terms[k].append(frame_data[k])
            frame_data = {}
            in_frame = True
            continue

        if not in_frame:
            continue

        # Parse key=value pairs on the line
        for m in pat.finditer(line):
            key = m.group(1).strip()
            val_str = m.group(2).strip()
            if '*' in val_str:
                val = np.nan
            else:
                try:
                    val = float(val_str)
                except ValueError:
                    continue
            # normalise key
            if key in terms:
                frame_data[key] = val

        # End of frame: blank or separator
        if line.strip() == '' and frame_data:
            pass  # keep accumulating

    # flush last frame
    if frame_data:
        for k in terms:
            if k in frame_data:
                terms[k].append(frame_data[k])

    return {k: np.array(v) for k, v in terms.items() if v}


def compute_dg(workdir):
    com = parse_mdout(os.path.join(workdir, "_GMXMMPBSA_complex_pb.mdout.0"))
    rec = parse_mdout(os.path.join(workdir, "_GMXMMPBSA_receptor_pb.mdout.0"))
    lig = parse_mdout(os.path.join(workdir, "_GMXMMPBSA_ligand_pb.mdout.0"))

    n_com = len(com["VDWAALS"])
    n_rec = len(rec["VDWAALS"])
    n_lig = len(lig["VDWAALS"])
    print(f"Frames parsed: complex={n_com}, receptor={n_rec}, ligand={n_lig}")
    n = min(n_com, n_rec, n_lig)

    def get(d, k, n):
        return d.get(k, np.zeros(n))[:n]

    # ΔE_vdw = Δ(VDWAALS + 1-4 VDW)
    vdw_c = get(com,"VDWAALS",n) + get(com,"1-4 VDW",n)
    vdw_r = get(rec,"VDWAALS",n) + get(rec,"1-4 VDW",n)
    vdw_l = get(lig,"VDWAALS",n) + get(lig,"1-4 VDW",n)
    d_vdw = vdw_c - vdw_r - vdw_l

    # ΔE_elec = Δ(EEL + 1-4 EEL)
    eel_c = get(com,"EEL",n) + get(com,"1-4 EEL",n)
    eel_r = get(rec,"EEL",n) + get(rec,"1-4 EEL",n)
    eel_l = get(lig,"EEL",n) + get(lig,"1-4 EEL",n)
    d_eel = eel_c - eel_r - eel_l

    # ΔEPB (polar solvation)
    d_pb  = get(com,"EPB",n) - get(rec,"EPB",n) - get(lig,"EPB",n)

    # ΔE_nonpolar = Δ(ECAVITY + EDISPER)
    np_c = get(com,"ECAVITY",n) + get(com,"EDISPER",n)
    np_r = get(rec,"ECAVITY",n) + get(rec,"EDISPER",n)
    np_l = get(lig,"ECAVITY",n) + get(lig,"EDISPER",n)
    d_np  = np_c - np_r - np_l

    dg = d_vdw + d_eel + d_pb + d_np  # kcal/mol, TΔS=0 for STP

    print(f"\n{'Term':<16} {'Mean':>12} {'SD':>10} (kcal/mol)")
    print("-" * 42)
    for name, arr in [("ΔEvdW", d_vdw), ("ΔEelec", d_eel), ("ΔEPB", d_pb), ("ΔEnpol", d_np), ("ΔG_bind", dg)]:
        m = float(np.nanmean(arr))
        s = float(np.nanstd(arr, ddof=1) if len(arr) > 1 else 0.0)
        print(f"  {name:<14} {m:>12.2f} {s:>10.2f}")

    # also in kJ/mol (multiply kcal × 4.184)
    dg_kj = np.nanmean(dg) * 4.184
    dg_sd_kj = np.nanstd(dg, ddof=1) * 4.184 if len(dg) > 1 else 0.0
    print(f"\n  ΔG_bind = {np.nanmean(dg):.1f} ± {np.nanstd(dg,ddof=1):.1f} kcal/mol")
    print(f"           = {dg_kj:.1f} ± {dg_sd_kj:.1f} kJ/mol")

    return {
        "n_frames": int(n),
        "dG_bind_kcal_mean": float(np.nanmean(dg)),
        "dG_bind_kcal_sd":   float(np.nanstd(dg, ddof=1) if len(dg) > 1 else 0),
        "dG_bind_kJ_mean":   float(dg_kj),
        "dG_bind_kJ_sd":     float(dg_sd_kj),
        "dEvdW_kcal_mean":   float(np.nanmean(d_vdw)),
        "dEelec_kcal_mean":  float(np.nanmean(d_eel)),
        "dEPB_kcal_mean":    float(np.nanmean(d_pb)),
        "dEnpol_kcal_mean":  float(np.nanmean(d_np)),
        "per_frame_dg":      dg.tolist(),
    }


if __name__ == "__main__":
    workdir = sys.argv[1] if len(sys.argv) > 1 else "."
    print(f"Parsing mdout files in: {workdir}")
    result = compute_dg(workdir)
    out_json = os.path.join(workdir, "mmpbsa_result.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved to: {out_json}")
