#!/usr/bin/env python3
"""
Diagnose all PDBBind complexes and report exact counts per failure reason.

Usage:
    python diagnose_pdbbind.py \
        --in-dir /scratch/project_465002574/PDB/PDBBind/pbpp-2020 \
        --out-log /scratch/project_465002574/logs/pdbbind_diagnosis.tsv \
        [--n-workers 14]
"""
import argparse
import os
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, "/flash/project_465002574/UAAG2_main/src")

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

from uaag2.datasets.pdb_dataset import (
    ATOM_ENCODER, HYBRIDIZATION_ENCODER,
    _read_mol_and_structure, _extract_atom_bond_features,
)

# ---------------------------------------------------------------------------
# Helpers (duplicated from build script to keep this self-contained)
# ---------------------------------------------------------------------------

def _strip_h_from_pdb(pdb_path: str) -> str:
    lines = []
    with open(pdb_path, "r", errors="ignore") as fh:
        for line in fh:
            if line.startswith(("ATOM  ", "HETATM")):
                padded = line.rstrip("\n").ljust(80)
                elem = padded[76:78].strip().upper()
                if not elem:
                    alpha = "".join(c for c in padded[12:16].strip() if c.isalpha())
                    elem = alpha[:1].upper() if alpha else ""
                if elem in ("H", "D"):
                    continue
            lines.append(line)
    fd, tmp = tempfile.mkstemp(prefix="diag_noh_", suffix=".pdb")
    os.close(fd)
    with open(tmp, "w") as fh:
        fh.writelines(lines)
    return tmp


def _diagnose_one(args):
    complex_dir, complex_id = args
    try:
        return _diagnose_one_inner(complex_dir, complex_id)
    except Exception as exc:
        return complex_id, "UNEXPECTED_ERROR", str(exc)[:120]


def _diagnose_one_inner(complex_dir, complex_id):
    pocket_pdb = os.path.join(complex_dir, f"{complex_id}_pocket.pdb")
    ligand_sdf = os.path.join(complex_dir, f"{complex_id}_ligand.sdf")

    # ---- File existence ----
    missing = []
    if not os.path.isfile(pocket_pdb):
        missing.append("pocket.pdb")
    if not os.path.isfile(ligand_sdf):
        missing.append("ligand.sdf")
    if missing:
        return complex_id, "MISSING_FILES", f"missing: {','.join(missing)}"

    # ---- Ligand parse ----
    mol_lig = Chem.MolFromMolFile(ligand_sdf, removeHs=False, sanitize=False)
    if mol_lig is None:
        return complex_id, "LIG_PARSE_FAIL", "RDKit MolFromMolFile returned None"

    san_err = Chem.SanitizeMol(mol_lig, catchErrors=True)
    if san_err != Chem.SanitizeFlags.SANITIZE_NONE:
        # Try without kekulization
        mol_lig2 = Chem.MolFromMolFile(ligand_sdf, removeHs=False, sanitize=False)
        san_err2 = Chem.SanitizeMol(
            mol_lig2,
            Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
            catchErrors=True,
        )
        if san_err2 != Chem.SanitizeFlags.SANITIZE_NONE:
            return complex_id, "LIG_SAN_FAIL", f"SanitizeFlags error code {int(san_err)}"
        mol_lig = mol_lig2

    try:
        mol_lig = Chem.RemoveAllHs(mol_lig, sanitize=False)
    except Exception as exc:
        return complex_id, "LIG_REMOVE_H_FAIL", str(exc)[:120]
    if mol_lig is None:
        return complex_id, "LIG_REMOVE_H_FAIL", "RemoveAllHs returned None"

    Chem.AssignStereochemistry(mol_lig)

    # ---- Ligand feature check ----
    lig_atoms = []
    for atom in mol_lig.GetAtoms():
        lig_atoms.append({
            "Atoms": atom.GetSymbol(),
            "Hybridization": str(atom.GetHybridization()),
        })

    if len(lig_atoms) < 4:
        return complex_id, "LIG_TOO_FEW_ATOMS", f"n={len(lig_atoms)}"

    bad_lig = [
        (a["Atoms"], a["Hybridization"])
        for a in lig_atoms
        if a["Atoms"] not in ATOM_ENCODER or a["Hybridization"] not in HYBRIDIZATION_ENCODER
    ]
    if bad_lig:
        # Separate element vs hybridisation failures
        bad_elem = sorted({a for a, h in bad_lig if a not in ATOM_ENCODER})
        bad_hyb  = sorted({h for a, h in bad_lig if a in ATOM_ENCODER and h not in HYBRIDIZATION_ENCODER})
        parts = []
        if bad_elem:
            parts.append(f"unsupported_element={bad_elem}")
        if bad_hyb:
            parts.append(f"unsupported_hyb={bad_hyb}")
        tag = "LIG_UNSUPPORTED_ELEMENT" if bad_elem else "LIG_UNSUPPORTED_HYB"
        return complex_id, tag, "; ".join(parts)

    # ---- Pocket parse ----
    tmp_pocket = _strip_h_from_pdb(pocket_pdb)
    try:
        mol_pkt, struct_pkt = _read_mol_and_structure(tmp_pocket)
        pkt_af, _ = _extract_atom_bond_features(mol_pkt, struct_pkt)
    except Exception as exc:
        return complex_id, "PKT_PARSE_FAIL", str(exc)[:120]
    finally:
        try:
            os.remove(tmp_pocket)
        except OSError:
            pass

    # Filter pocket atoms
    pkt_atoms = [
        a for a in pkt_af.values()
        if a["Atoms"] in ATOM_ENCODER and a["Hybridization"] in HYBRIDIZATION_ENCODER
    ]
    if not pkt_atoms:
        return complex_id, "PKT_ALL_FILTERED", "no supported pocket atoms after filtering"

    # ---- Size check ----
    n_total = len(lig_atoms) + len(pkt_atoms)
    if n_total > 400:
        return complex_id, "TOO_LARGE", f"n_lig={len(lig_atoms)} n_pkt={len(pkt_atoms)} total={n_total}"

    return complex_id, "OK", f"n_lig={len(lig_atoms)} n_pkt={len(pkt_atoms)}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir",    required=True)
    ap.add_argument("--out-log",   required=True)
    ap.add_argument("--n-workers", type=int, default=14)
    args = ap.parse_args()

    complex_ids = sorted(
        d for d in os.listdir(args.in_dir)
        if os.path.isdir(os.path.join(args.in_dir, d)) and not d.startswith(".")
    )
    print(f"Diagnosing {len(complex_ids)} complexes with {args.n_workers} workers …", flush=True)

    work = [(os.path.join(args.in_dir, cid), cid) for cid in complex_ids]

    counts: dict = {}
    detail_rows = []

    with Pool(processes=args.n_workers) as pool:
        for cid, tag, detail in pool.imap_unordered(_diagnose_one, work, chunksize=20):
            counts[tag] = counts.get(tag, 0) + 1
            detail_rows.append((cid, tag, detail))

    # Write per-complex TSV
    detail_rows.sort(key=lambda r: (r[1], r[0]))
    os.makedirs(os.path.dirname(args.out_log), exist_ok=True)
    with open(args.out_log, "w") as fh:
        fh.write("complex_id\tstatus\tdetail\n")
        for row in detail_rows:
            fh.write("\t".join(row) + "\n")

    # Print summary
    total = len(complex_ids)
    print(f"\n{'='*60}")
    print(f"PDBBind diagnosis — {total} complexes")
    print(f"{'='*60}")
    for tag in ["OK", "TOO_LARGE", "LIG_UNSUPPORTED_ELEMENT", "LIG_UNSUPPORTED_HYB",
                "LIG_SAN_FAIL", "LIG_TOO_FEW_ATOMS", "LIG_PARSE_FAIL",
                "LIG_REMOVE_H_FAIL", "PKT_PARSE_FAIL", "PKT_ALL_FILTERED", "MISSING_FILES"]:
        n = counts.get(tag, 0)
        if n:
            print(f"  {tag:<30} {n:5d}  ({100*n/total:.1f}%)")
    other = {k: v for k, v in counts.items() if k not in ["OK","TOO_LARGE","LIG_UNSUPPORTED_ELEMENT",
             "LIG_UNSUPPORTED_HYB","LIG_SAN_FAIL","LIG_TOO_FEW_ATOMS","LIG_PARSE_FAIL",
             "LIG_REMOVE_H_FAIL","PKT_PARSE_FAIL","PKT_ALL_FILTERED","MISSING_FILES"]}
    for tag, n in sorted(other.items()):
        print(f"  {tag:<30} {n:5d}  ({100*n/total:.1f}%)")
    print(f"{'='*60}")
    print(f"  {'TOTAL':<30} {total:5d}")
    print(f"\nPer-complex log written to: {args.out_log}", flush=True)

    # For LIG_UNSUPPORTED_ELEMENT: show element breakdown
    bad_elems: dict = {}
    for cid, tag, detail in detail_rows:
        if tag == "LIG_UNSUPPORTED_ELEMENT":
            # parse "unsupported_element=['I']"
            import re
            m = re.search(r"unsupported_element=\[([^\]]+)\]", detail)
            if m:
                for e in m.group(1).replace("'","").split(", "):
                    bad_elems[e] = bad_elems.get(e, 0) + 1
    if bad_elems:
        print("\nUnsupported ligand elements:")
        for e, n in sorted(bad_elems.items(), key=lambda x: -x[1]):
            print(f"  {e:<6} {n}")

    # For TOO_LARGE: show total-atom distribution
    large_totals = []
    for cid, tag, detail in detail_rows:
        if tag == "TOO_LARGE":
            import re
            m = re.search(r"total=(\d+)", detail)
            if m:
                large_totals.append(int(m.group(1)))
    if large_totals:
        large_totals.sort()
        n = len(large_totals)
        print(f"\nTOO_LARGE total-atom distribution (n={n}):")
        for pct, label in [(25,'p25'), (50,'p50'), (75,'p75'), (90,'p90'), (95,'p95'), (100,'max')]:
            idx = min(int(pct/100*n), n-1)
            print(f"  {label}: {large_totals[idx]}")
        brackets = [(401,500),(501,600),(601,800),(801,1000),(1001,9999)]
        for lo, hi in brackets:
            c = sum(1 for x in large_totals if lo <= x <= hi)
            if c:
                print(f"  {lo}-{hi}: {c}")


if __name__ == "__main__":
    main()
