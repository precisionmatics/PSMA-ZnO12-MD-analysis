#!/usr/bin/env python3
"""Extract energy, density, temperature, and conserved quantity from GROMACS .edr files.
Writes out/<sys>_rep<N>_energy.npz for each system."""
import subprocess, os, numpy as np, json

RDIR = os.path.dirname(os.path.abspath(__file__))
OUTDIR = f"{RDIR}/out"
os.makedirs(OUTDIR, exist_ok=True)

SYSTEMS = {
    "apo":    (1, 2),
    "naag":   (1, 2),
    "zno541": (1, 2),
    "tern541":(1, 2),
}

# GROMACS energy terms to extract (numeric indices passed to gmx energy)
# 1=Bond, 9=Potential, 10=Kinetic, 11=Total, 12=Conserved, 14=Temperature, 17=Pressure, 25=Density
TERMS = {
    "potential_kJ":   12,
    "kinetic_kJ":     13,
    "total_kJ":       14,
    "conserved_kJ":   15,
    "temperature_K":  16,
    "pressure_bar":   17,
    "density_kgm3":   23,
}

GMX = "/home/stalin/miniforge3/envs/gmx/bin/gmx"  # path verified working

def run_gmx_energy(edr, term_id, outxvg):
    """Run gmx energy and return time (ns) and value arrays."""
    proc = subprocess.run(
        [GMX, "energy", "-f", edr, "-o", outxvg, "-quiet"],
        input=f"{term_id}\n0\n",
        capture_output=True, text=True
    )
    if not os.path.exists(outxvg):
        return None, None
    t, v = [], []
    with open(outxvg) as f:
        for line in f:
            if line.startswith(('#', '@')):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    t.append(float(parts[0]))  # ps
                    v.append(float(parts[1]))
                except ValueError:
                    continue
    os.remove(outxvg)
    return np.array(t)/1000.0, np.array(v)  # convert ps -> ns

for sys, reps in SYSTEMS.items():
    for rep in reps:
        edr = f"{RDIR}/systems/{sys}/rep{rep}/md.edr"
        if not os.path.exists(edr):
            print(f"MISSING: {edr}")
            continue
        outbase = f"{OUTDIR}/{sys}_rep{rep}_energy"
        if os.path.exists(outbase + ".npz"):
            print(f"EXISTS: {outbase}.npz")
            continue
        print(f"Extracting {sys} rep{rep}...")
        data = {}
        for name, tid in TERMS.items():
            xvg = f"/tmp/gmx_energy_{sys}_{rep}_{name}.xvg"
            t_ns, v = run_gmx_energy(edr, tid, xvg)
            if t_ns is not None:
                data[f"t_ns_{name}"] = t_ns
                data[name] = v
                print(f"  {name}: {len(v)} points, mean={np.nanmean(v):.3g}")
        if data:
            np.savez(outbase + ".npz", **data)
            print(f"  -> {outbase}.npz")

print("Energy extraction complete")
