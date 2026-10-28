import numpy as np, subprocess, sys, os
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from meeko import MoleculePreparation, PDBQTWriterLegacy, PDBQTMolecule, RDKitMolCreate
smi="CC(=O)N[C@@H](CC(=O)[O-])C(=O)N[C@@H](CCC(=O)[O-])C(=O)[O-]"   # NAAG, net -3 (pH 7.4)
m=Chem.AddHs(Chem.MolFromSmiles(smi)); print("NAAG",CalcMolFormula(m),"charge",Chem.GetFormalCharge(m))
# crystal reference (3BXM chain I, heavy atoms) mapped onto template
pdb=[l for l in open("../00_structure/3BXM.pdb") if l.startswith(("ATOM","HETATM")) and l[21]=="I" and l[17:20]!="HOH"]
ref=Chem.MolFromPDBBlock("".join(pdb),removeHs=True,proximityBonding=True)
tmpl=Chem.MolFromSmiles("CC(=O)N[C@@H](CC(=O)[O-])C(=O)N[C@@H](CCC(=O)[O-])C(=O)[O-]")
try:
    ref=AllChem.AssignBondOrdersFromTemplate(tmpl,ref)
except Exception as e:
    tmpl=Chem.MolFromSmiles("CC(=O)N[C@@H](CC(=O)O)C(=O)N[C@@H](CCC(=O)O)C(=O)O"); ref=AllChem.AssignBondOrdersFromTemplate(tmpl,ref)
print("reference heavy atoms",ref.GetNumAtoms())
Chem.MolToMolFile(ref,"naag_crystal_ref.mol")
# ligand pdbqt from random conformer
AllChem.EmbedMolecule(m,randomSeed=7); AllChem.MMFFOptimizeMolecule(m)
prep=MoleculePreparation(); setups=prep.prepare(m)
pq,ok,err=PDBQTWriterLegacy.write_string(setups[0]); open("naag.pdbqt","w").write(pq); print("pdbqt ok",ok)
Chem.MolToMolFile(m,"naag_start.mol")
