#!/usr/bin/env python3
import sys, warnings
sys.path.insert(0, "/flash/project_465002574/UAAG2_main/src")
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=PDBConstructionWarning)

PDBS = [
    "/scratch/project_465002574/PDB/PDB_processed/1EW9.pdb",
    "/scratch/project_465002574/PDB/PDB_processed/7KTD.pdb",
]

for pdb_path in PDBS:
    name = pdb_path.split("/")[-1]
    mol = Chem.MolFromPDBFile(pdb_path, removeHs=True, sanitize=False)
    if mol:
        Chem.SanitizeMol(mol)
    structure = PDBParser(QUIET=True).get_structure("x", pdb_path)

    rdkit_atoms = list(mol.GetAtoms()) if mol else []
    pdb_atoms = []
    for chain in structure[0]:
        for res in chain:
            for atom in res:
                elem = (atom.element or "").upper().strip()
                if elem not in ("H", "D"):
                    pdb_atoms.append((atom, res.get_resname(), chain.id, res.id))

    print(f"\n{name}: RDKit={len(rdkit_atoms)}, BioPython={len(pdb_atoms)}")

    if len(rdkit_atoms) == len(pdb_atoms):
        print("  counts match")
        continue

    rdkit_elems = [a.GetSymbol() for a in rdkit_atoms]
    biopython_elems = [a[0].element.capitalize() for a in pdb_atoms]

    # Find first divergence in the aligned prefix
    first_div = None
    for i, (re, be) in enumerate(zip(rdkit_elems, biopython_elems)):
        if re != be:
            first_div = i
            a = pdb_atoms[i]
            print(f"  First element mismatch at idx {i}: RDKit={re}, PDB={be}  "
                  f"resname={a[1]} chain={a[2]} resid={a[3]} atomname={a[0].get_name()}")
            break

    if first_div is None:
        # Lists are same up to the shorter length; extra atoms are at the end
        extra_start = min(len(rdkit_atoms), len(pdb_atoms))
        if len(pdb_atoms) > len(rdkit_atoms):
            print(f"  BioPython has {len(pdb_atoms) - len(rdkit_atoms)} extra atom(s) at the end:")
            for a in pdb_atoms[extra_start:extra_start+10]:
                print(f"    elem={a[0].element} name={a[0].get_name().strip()} "
                      f"resname={a[1]} chain={a[2]} resid={a[3]}")
        else:
            print(f"  RDKit has {len(rdkit_atoms) - len(pdb_atoms)} extra atom(s):")
            for a in rdkit_atoms[extra_start:extra_start+10]:
                info = a.GetMonomerInfo()
                print(f"    sym={a.GetSymbol()} name={info.GetName().strip() if info else '?'}")

    # Also check for alternate conformations in the PDB
    alt_count = 0
    for chain in structure[0]:
        for res in chain:
            for atom in res:
                if hasattr(atom, 'altloc') and atom.altloc not in (' ', 'A', ''):
                    alt_count += 1
    if alt_count:
        print(f"  Found {alt_count} atoms with alternate conformations (altloc != ' '/'A')")
