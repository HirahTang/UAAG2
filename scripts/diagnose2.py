#!/usr/bin/env python3
"""Check whether skipping non-primary altloc atoms fixes the count mismatch."""
import sys, os, warnings, tempfile
sys.path.insert(0, "/flash/project_465002574/UAAG2_main/src")
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=PDBConstructionWarning)

from uaag2.datasets.pdb_dataset import _needs_repair, _repair

PDBS = [
    "/scratch/project_465002574/PDB/PDB_processed/1EW9.pdb",
    "/scratch/project_465002574/PDB/PDB_processed/7KTD.pdb",
]

for pdb_path in PDBS:
    name = pdb_path.split("/")[-1]

    # Replicate _read_mol_and_structure logic
    parse_path = _repair(pdb_path) if _needs_repair(pdb_path) else pdb_path
    mol = Chem.MolFromPDBFile(parse_path, removeHs=True, sanitize=False)
    if mol is None and parse_path == pdb_path:
        parse_path = _repair(pdb_path)
        mol = Chem.MolFromPDBFile(parse_path, removeHs=True, sanitize=False)
    if mol:
        Chem.SanitizeMol(mol)
    structure = PDBParser(QUIET=True).get_structure("x", parse_path)

    rdkit_n = len(list(mol.GetAtoms())) if mol else 0

    # Count WITHOUT altloc filter (current behaviour)
    pdb_all = []
    for chain in structure[0]:
        for res in chain:
            for atom in res:
                if (atom.element or "").upper().strip() not in ("H", "D"):
                    pdb_all.append(atom)

    # Count WITH altloc filter (proposed fix: skip non-primary alternates)
    pdb_primary = []
    for chain in structure[0]:
        for res in chain:
            for atom in res:
                if (atom.element or "").upper().strip() in ("H", "D"):
                    continue
                altloc = getattr(atom, "altloc", " ")
                if altloc not in (" ", "", "A"):
                    continue
                pdb_primary.append(atom)

    print(f"{name}: RDKit={rdkit_n}  BioPython_all={len(pdb_all)}  BioPython_primary={len(pdb_primary)}")
    print(f"  altloc-filtered match RDKit? {rdkit_n == len(pdb_primary)}")

    # Show the extra atoms that get filtered out
    extra = [a for a in pdb_all if getattr(a, "altloc", " ") not in (" ", "", "A")]
    for a in extra:
        print(f"  skipped altloc={a.altloc!r} elem={a.element} name={a.get_name().strip()}")

    if parse_path != pdb_path and os.path.exists(parse_path):
        os.remove(parse_path)
