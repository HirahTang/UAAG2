#!/usr/bin/env python3
"""
Process mmCIF files to clean PDB output:
  - Remove hydrogen / deuterium atoms
  - Remove water molecules (HOH, WAT, DOD, ...)
  - Split multi-model structures into separate PDB files
  - Handles structures with >99999 atoms (serial-number wrap)

Usage (called by SLURM array job):
  python process_cif.py --in-dir DIR --out-dir DIR
                        --task-id INT --n-tasks INT [--n-workers INT]
"""
import argparse
import os
import sys
import warnings
from pathlib import Path
from multiprocessing import Pool

from Bio.PDB import MMCIFParser, PDBIO, Select
from Bio.PDB.PDBExceptions import PDBConstructionWarning, PDBIOException

warnings.filterwarnings("ignore", category=PDBConstructionWarning)

# ---------------------------------------------------------------------------
WATER_RESNAMES = {"HOH", "WAT", "H2O", "DOD", "MOH", "HHO"}


class StructureFilter(Select):
    """Drop water residues and H/D atoms at write time.

    Checks resname directly — avoids identity/hash issues that arise when
    set_structure(model) deep-copies the model hierarchy.
    """
    def accept_residue(self, res):
        return res.resname.strip().upper() not in WATER_RESNAMES

    def accept_atom(self, atom):
        return (atom.element or "").upper().strip() not in ("H", "D")


_FILTER = StructureFilter()


def _renumber_atoms(entity):
    """Renumber atoms 1-based, wrapping mod 99999 for very large structures."""
    for i, atom in enumerate(entity.get_atoms(), start=1):
        atom.serial_number = ((i - 1) % 99999) + 1


# ---------------------------------------------------------------------------

def process_cif(args):
    cif_path, out_dir = args
    pdb_id = Path(cif_path).stem.upper()
    parser = MMCIFParser(QUIET=True)

    try:
        structure = parser.get_structure(pdb_id, cif_path)
    except Exception as exc:
        return f"PARSE_ERR\t{pdb_id}\t{exc}"

    models   = list(structure.get_models())
    n_models = len(models)
    io       = PDBIO()
    written  = 0
    errors   = []

    for model in models:
        # Check there are surviving heavy atoms after filtering
        surviving = sum(
            1
            for chain in model
            for res   in chain
            if res.resname.strip().upper() not in WATER_RESNAMES
            for atom  in res.get_atoms()
            if (atom.element or "").upper().strip() not in ("H", "D")
        )
        if surviving == 0:
            continue

        out_name = (
            f"{pdb_id}.pdb"
            if n_models == 1
            else f"{pdb_id}_model{model.id}.pdb"
        )
        out_path = os.path.join(out_dir, out_name)

        io.set_structure(model)
        try:
            io.save(out_path, _FILTER)
        except (PDBIOException, ValueError) as exc:
            if "serial number" in str(exc).lower() or "100000" in str(exc):
                # Renumber and retry once
                _renumber_atoms(model)
                io.set_structure(model)
                try:
                    io.save(out_path, _FILTER)
                except Exception as exc2:
                    errors.append(f"WRITE_ERR\t{pdb_id}\t{exc2}")
                    continue
            else:
                errors.append(f"WRITE_ERR\t{pdb_id}\t{exc}")
                continue
        except Exception as exc:
            errors.append(f"WRITE_ERR\t{pdb_id}\t{exc}")
            continue

        written += 1

    if errors:
        return "\n".join(errors)
    return f"OK\t{pdb_id}\t{n_models} model(s)\t{written} written"


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir",    required=True)
    ap.add_argument("--out-dir",   required=True)
    ap.add_argument("--task-id",   type=int, default=0)
    ap.add_argument("--n-tasks",   type=int, default=1)
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--log-dir",   default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    all_cifs = sorted(str(p) for p in Path(args.in_dir).glob("*.cif"))
    my_cifs  = all_cifs[args.task_id :: args.n_tasks]

    log_path = (
        os.path.join(args.log_dir, f"task_{args.task_id:04d}.log")
        if args.log_dir else None
    )
    log_fh = open(log_path, "w") if log_path else sys.stdout

    print(f"[task {args.task_id}/{args.n_tasks}] "
          f"{len(my_cifs)} files, {args.n_workers} workers", flush=True)

    ok = err = 0
    with Pool(processes=args.n_workers) as pool:
        for result in pool.imap_unordered(
            process_cif, [(c, args.out_dir) for c in my_cifs], chunksize=20
        ):
            for line in result.split("\n"):
                if line.startswith("OK"):
                    ok += 1
                else:
                    err += 1
                    print(line, file=log_fh, flush=True)

    summary = (f"[task {args.task_id}] done: {ok} OK, {err} errors "
               f"out of {len(my_cifs)} files")
    print(summary, flush=True)
    if log_fh is not sys.stdout:
        print(summary, file=log_fh)
        log_fh.close()


if __name__ == "__main__":
    main()
