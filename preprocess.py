#!/usr/bin/env python3
"""Extract Solute GRO files from full solvated GRO + NDX for each system."""
import os, sys

RDIR = os.path.dirname(os.path.abspath(__file__)) + "/systems"
SYSTEMS = ["apo", "naag", "zno541", "tern541"]

def parse_ndx(ndx_file):
    groups = {}
    current = None
    with open(ndx_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith('['):
                current = line.strip('[ ]')
                groups[current] = []
            elif current and line:
                groups[current].extend(int(x) for x in line.split())
    return groups

def parse_gro(gro_file):
    with open(gro_file) as f:
        lines = f.readlines()
    title = lines[0]
    natoms = int(lines[1].strip())
    atom_lines = lines[2:2+natoms]
    box = lines[2+natoms]
    return title, natoms, atom_lines, box

def write_gro(outfile, title, atom_lines, box):
    with open(outfile, 'w') as f:
        f.write(title.rstrip('\n').rstrip() + ' (Solute)\n')
        f.write(f'{len(atom_lines):5d}\n')
        for line in atom_lines:
            f.write(line)
        f.write(box)

for sys in SYSTEMS:
    sdir = f"{RDIR}/{sys}"
    gro = f"{sdir}/solv_ions.gro"
    ndx = f"{sdir}/index.ndx"
    out = f"{sdir}/solute.gro"
    if os.path.exists(out):
        print(f"{sys}: solute.gro already exists, skipping")
        continue
    groups = parse_ndx(ndx)
    solute_ids = set(groups.get("Solute", []))  # 1-based atom numbers
    title, natoms, atom_lines, box = parse_gro(gro)
    selected = [atom_lines[i-1] for i in sorted(solute_ids) if i <= natoms]
    write_gro(out, title, selected, box)
    print(f"{sys}: wrote {len(selected)} atoms -> {out}")

print("Done preprocessing")
