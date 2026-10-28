# PSMA–(ZnO)₁₂ MD Analysis Scripts

Analysis and simulation scripts for:  
**"Molecular Dynamics Simulation of PSMA–(ZnO)₁₂ Nanocluster Binding"**

## Repository Structure

| Path | Description |
|------|-------------|
| `analysis/psma_md_analysis.py` | Main MD trajectory analysis (RMSD, RMSF, Rg, SASA, distances, FRET, coordination) |
| `generate_figures.py` | Publication-quality figure generation (Figs 1–11) |
| `gen_missing_figures.py` | Supplementary figure generation (NAAG distance, MM-PBSA) |
| `extract_edr.py` | GROMACS EDR energy file parser |
| `preprocess.py` | Trajectory preprocessing and alignment |
| `fix_distances.py` | Distance metric corrections |
| `run_all_analysis.sh` | Master script to run full analysis pipeline |
| `mmpbsa/` | MM-PBSA binding free energy calculation and parsing |
| `provenance/` | Force-field parameterisation, docking, system assembly |
| `server/` | HPC job submission and postprocessing scripts |

## Requirements

- GROMACS 2024.4
- Python 3.10+: numpy, MDAnalysis, matplotlib, scipy
- CHARMM36-Jul2021 force field
- (ZnO)₁₂ RESP charges (±0.8441 e, provided in provenance/)

## Systems Simulated

Four systems × 2 replicas × 300 ns (CHARMM36m/TIP3P, NPT):
- **Apo** — PSMA alone
- **NAAG** — PSMA + substrate NAAG
- **ZnO** — PSMA + (ZnO)₁₂ nanocluster
- **Ternary** — PSMA + NAAG + (ZnO)₁₂
