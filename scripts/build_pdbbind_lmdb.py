#!/usr/bin/env python3
"""
Build PDBBind.lmdb from the PDBBind 2020 dataset.

Each protein-ligand complex becomes one entry: a raw (pre-_post_process) pickled
PyG Data object with the same field layout as UAAG2DatasetPDB._build_graph output.
Ligand atoms have is_ligand=1; pocket atoms have is_ligand=0.  is_backbone=0 for
all atoms (no backbone concept for small molecules).

Usage (run on LUMI login/compute node):
    python build_pdbbind_lmdb.py \
        --in-dir  /scratch/project_465002574/PDB/PDBBind/pbpp-2020 \
        --out-path /scratch/project_465002574/PDB/PDBBind.lmdb \
        [--n-workers 8] [--edge-radius 8.0] [--log-path build_pdbbind.log]
"""
import argparse
import os
import pickle
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import lmdb
import numpy as np
import torch
from rdkit import Chem, RDLogger
from scipy.spatial.distance import cdist
from torch_geometric.data import Data

# ---------------------------------------------------------------------------
# Locate pdb_dataset.py — try common install locations
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent

def _add_pdb_dataset_to_path():
    candidates = [
        _SCRIPT_DIR.parent / "src",                                           # scripts/../src
        _SCRIPT_DIR.parent / "src" / "uaag2" / "datasets",
        Path("/flash/project_465002574/UAAG2_main/src"),
    ]
    for c in candidates:
        if (c / "uaag2" / "datasets" / "pdb_dataset.py").exists():
            sys.path.insert(0, str(c))
            return
        if (c / "pdb_dataset.py").exists():
            sys.path.insert(0, str(c))
            return
    # Last resort: same directory as this script
    sys.path.insert(0, str(_SCRIPT_DIR))

_add_pdb_dataset_to_path()

try:
    from uaag2.datasets.pdb_dataset import (
        ATOM_ENCODER, HYBRIDIZATION_ENCODER, BOND_ENCODER,
        _read_mol_and_structure, _extract_atom_bond_features,
        _has_supported_features,
    )
except ImportError:
    from pdb_dataset import (
        ATOM_ENCODER, HYBRIDIZATION_ENCODER, BOND_ENCODER,
        _read_mol_and_structure, _extract_atom_bond_features,
        _has_supported_features,
    )

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")

# ---------------------------------------------------------------------------
# Hydrogen stripping — PDBBind pocket.pdb files retain explicit H atoms;
# _extract_atom_bond_features skips them in BioPython but RDKit includes them,
# causing name-match failures.  Write a tmp PDB without H/D lines first.
# ---------------------------------------------------------------------------

def _strip_h_from_pdb(pdb_path: str) -> str:
    """Return path to a temp PDB with H/D ATOM/HETATM lines removed."""
    lines = []
    with open(pdb_path, "r", errors="ignore") as fh:
        for line in fh:
            if line.startswith(("ATOM  ", "HETATM")):
                padded = line.rstrip("\n").ljust(80)
                elem = padded[76:78].strip().upper()
                if not elem:
                    # Fall back to atom-name column: first alpha character
                    alpha = "".join(c for c in padded[12:16].strip() if c.isalpha())
                    elem = alpha[:1].upper() if alpha else ""
                if elem in ("H", "D"):
                    continue
            lines.append(line)
    fd, tmp = tempfile.mkstemp(prefix="pdbbind_noh_", suffix=".pdb")
    os.close(fd)
    with open(tmp, "w") as fh:
        fh.writelines(lines)
    return tmp


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def _build_pdbbind_graph(
    pocket_pdb: str,
    ligand_sdf: str,
    complex_id: str,
    edge_radius: float = 8.0,
) -> Optional[Data]:
    """Build a PyG Data graph for one PDBBind complex.

    Returns None if the complex is skipped (unsupported elements, too large,
    too few ligand atoms).  Raises on hard parse errors.
    """
    # ---- Parse pocket (PDB) — strip H first so RDKit/BioPython agree ----
    tmp_pocket = _strip_h_from_pdb(pocket_pdb)
    try:
        mol_pkt, struct_pkt = _read_mol_and_structure(tmp_pocket)
        pkt_af, pkt_bf = _extract_atom_bond_features(mol_pkt, struct_pkt)
    finally:
        try:
            os.remove(tmp_pocket)
        except OSError:
            pass

    pkt_atoms_list = list(pkt_af.values())
    for a in pkt_atoms_list:
        a["coords"] = [float(v) for v in a["coords"]]

    # ---- Parse ligand (SDF) ----
    mol_lig = Chem.MolFromMolFile(ligand_sdf, removeHs=False, sanitize=False)
    if mol_lig is None:
        raise ValueError(f"RDKit could not parse ligand: {ligand_sdf}")
    try:
        Chem.SanitizeMol(mol_lig)
    except Exception as exc:
        # Kekulization or valence errors — try partial sanitization
        try:
            Chem.SanitizeMol(mol_lig,
                Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        except Exception as exc2:
            raise ValueError(f"Ligand sanitization failed: {exc2}") from exc2
    # Remove Hs after sanitization (removeHs=True at load time has no effect when
    # sanitize=False; explicit removal after sanitization is reliable)
    mol_lig = Chem.RemoveHs(mol_lig)
    if mol_lig is None:
        raise ValueError(f"RemoveHs returned None for ligand: {ligand_sdf}")
    Chem.AssignStereochemistry(mol_lig)

    if mol_lig.GetNumConformers() == 0:
        raise ValueError(f"Ligand SDF has no 3D conformer: {ligand_sdf}")
    conf = mol_lig.GetConformer()

    lig_atoms = []
    for atom in mol_lig.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        lig_atoms.append({
            "idx":          atom.GetIdx(),
            "coords":       [pos.x, pos.y, pos.z],
            "Atoms":        atom.GetSymbol(),
            "Charges":      atom.GetFormalCharge(),
            "Hybridization": str(atom.GetHybridization()),
            "Degree":       atom.GetDegree(),
            "Aromatic":     atom.GetIsAromatic(),
        })

    lig_bonds: dict = {}
    for bond in mol_lig.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        lig_bonds[(i, j)] = {
            "Type": str(bond.GetBondType()),
            "Is Aromatic": bond.GetIsAromatic(),
        }

    # ---- Feature / size checks ----
    if len(lig_atoms) < 4:
        return None
    # Ligand must be fully representable
    if not _has_supported_features(lig_atoms):
        return None
    # Pocket: filter out atoms with unsupported element/hybridisation
    # (e.g. metal ions, unusual HETATM records) rather than discarding the complex
    pkt_atoms_list = [
        a for a in pkt_atoms_list
        if a["Atoms"] in ATOM_ENCODER and a["Hybridization"] in HYBRIDIZATION_ENCODER
    ]
    if not pkt_atoms_list:
        return None

    n_lig = len(lig_atoms)
    n_pkt = len(pkt_atoms_list)
    num_atoms = n_lig + n_pkt

    if num_atoms > 400:
        return None

    # ---- Build tensors ----
    # Ligand indices: 0 .. n_lig-1  (is_ligand=1)
    # Pocket indices: n_lig .. n_lig+n_pkt-1  (is_ligand=0)
    atom_types    = torch.zeros(num_atoms, dtype=torch.long)
    charges       = torch.zeros(num_atoms, dtype=torch.float)
    is_aromatic   = torch.zeros(num_atoms, dtype=torch.float)
    hybridization = torch.zeros(num_atoms, dtype=torch.long)
    degree        = torch.zeros(num_atoms, dtype=torch.long)
    position      = torch.zeros(num_atoms, 3, dtype=torch.float)
    is_ligand     = torch.zeros(num_atoms, dtype=torch.float)
    is_backbone   = torch.zeros(num_atoms, dtype=torch.float)   # always 0

    for k, a in enumerate(lig_atoms):
        atom_types[k]    = ATOM_ENCODER[a["Atoms"]]
        charges[k]       = float(a["Charges"])
        is_aromatic[k]   = float(a["Aromatic"])
        hybridization[k] = HYBRIDIZATION_ENCODER[a["Hybridization"]]
        degree[k]        = int(a["Degree"])
        position[k]      = torch.tensor(a["coords"], dtype=torch.float)
        is_ligand[k]     = 1.0

    for k, a in enumerate(pkt_atoms_list):
        nid = n_lig + k
        atom_types[nid]    = ATOM_ENCODER[a["Atoms"]]
        charges[nid]       = float(a["Charges"])
        is_aromatic[nid]   = float(a["Aromatic"])
        hybridization[nid] = HYBRIDIZATION_ENCODER[a["Hybridization"]]
        degree[nid]        = int(a["Degree"])
        position[nid]      = torch.tensor(a["coords"], dtype=torch.float)

    # ---- Edge construction ----
    edge_src, edge_dst, edge_lig_flag = [], [], []

    lig_ids = list(range(n_lig))
    # lig–lig: fully connected
    for i in range(n_lig):
        for j in range(i + 1, n_lig):
            edge_src += [i, j]
            edge_dst += [j, i]
            edge_lig_flag += [1, 1]

    pkt_ids = list(range(n_lig, n_lig + n_pkt))
    if n_pkt > 0:
        # pkt–pkt: within edge_radius (vectorised)
        pkt_coords = np.array([a["coords"] for a in pkt_atoms_list])
        dist_mat = cdist(pkt_coords, pkt_coords)
        ii, jj = np.where(np.triu(dist_mat < edge_radius, k=1))
        for i, j in zip(ii.tolist(), jj.tolist()):
            edge_src += [pkt_ids[i], pkt_ids[j]]
            edge_dst += [pkt_ids[j], pkt_ids[i]]
            edge_lig_flag += [0, 0]

        # lig–pkt: fully cross-connected
        for lid in lig_ids:
            for pid in pkt_ids:
                edge_src += [lid, pid]
                edge_dst += [pid, lid]
                edge_lig_flag += [0, 0]

    if not edge_src:
        return None

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_type  = torch.zeros(edge_index.size(1), dtype=torch.long)

    # Assign bond types where known
    for k in range(edge_index.size(1)):
        src, dst = int(edge_index[0, k]), int(edge_index[1, k])
        if src < n_lig and dst < n_lig:
            # ligand–ligand bond
            bd = lig_bonds.get((src, dst)) or lig_bonds.get((dst, src))
            if bd:
                bt = bd.get("Type")
                if bt and bt in BOND_ENCODER:
                    edge_type[k] = BOND_ENCODER[bt]
        elif src >= n_lig and dst >= n_lig:
            # pocket–pocket bond
            orig_s = pkt_atoms_list[src - n_lig]["idx"]
            orig_d = pkt_atoms_list[dst - n_lig]["idx"]
            bd = pkt_bf.get((orig_s, orig_d)) or pkt_bf.get((orig_d, orig_s))
            if bd:
                bt = bd.get("Type")
                if bt and bt in BOND_ENCODER:
                    edge_type[k] = BOND_ENCODER[bt]
        # lig–pkt cross edges: bond type stays 0 (no covalent bond)

    edge_ligand = torch.tensor(edge_lig_flag, dtype=torch.float)
    all_ids = torch.arange(num_atoms, dtype=torch.long)

    return Data(
        x=atom_types,
        pos=position,
        edge_index=edge_index,
        edge_attr=edge_type,
        edge_ligand=edge_ligand,
        charges=charges,
        degree=degree,
        is_aromatic=is_aromatic,
        hybridization=hybridization,
        is_ligand=is_ligand,
        is_backbone=is_backbone,
        id=all_ids,
        ids=all_ids,
        compound_id=complex_id,
        source_name="pdbbind",
        center_residue="",
        residue_name="",
    )


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------

def _process_complex(args):
    """Parse one complex and return (key_bytes, pickled_graph) or status string."""
    complex_dir, complex_id, edge_radius = args
    pocket_pdb = os.path.join(complex_dir, f"{complex_id}_pocket.pdb")
    ligand_sdf = os.path.join(complex_dir, f"{complex_id}_ligand.sdf")

    if not os.path.isfile(pocket_pdb):
        return f"MISSING_POCKET\t{complex_id}"
    if not os.path.isfile(ligand_sdf):
        return f"MISSING_LIGAND\t{complex_id}"

    try:
        graph = _build_pdbbind_graph(pocket_pdb, ligand_sdf, complex_id, edge_radius)
    except Exception as exc:
        return f"ERROR\t{complex_id}\t{exc}"

    if graph is None:
        return f"SKIP\t{complex_id}"

    return (complex_id.encode(), pickle.dumps(graph, protocol=pickle.HIGHEST_PROTOCOL))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir",    required=True,
                    help="Path to the pbpp-2020 directory (one sub-dir per complex)")
    ap.add_argument("--out-path",  required=True,
                    help="Output LMDB file path (e.g. /scratch/.../PDBBind.lmdb)")
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--edge-radius", type=float, default=8.0)
    ap.add_argument("--log-path",  default=None)
    args = ap.parse_args()

    in_dir   = args.in_dir
    out_path = args.out_path

    # Discover all complex subdirectories
    complex_ids = sorted(
        d for d in os.listdir(in_dir)
        if os.path.isdir(os.path.join(in_dir, d)) and not d.startswith(".")
    )
    print(f"Found {len(complex_ids)} complexes in {in_dir}", flush=True)

    log_fh = open(args.log_path, "w") if args.log_path else sys.stdout

    # LMDB: map_size = 20 GB (well above typical ~500 MB actual use)
    env = lmdb.open(out_path, map_size=20 * 1024 ** 3, subdir=False)

    ok = skip = err = 0
    keys_written: list[bytes] = []

    work_items = [
        (os.path.join(in_dir, cid), cid, args.edge_radius)
        for cid in complex_ids
    ]

    with Pool(processes=args.n_workers) as pool:
        for result in pool.imap_unordered(_process_complex, work_items, chunksize=20):
            if isinstance(result, tuple):
                key_bytes, data_bytes = result
                with env.begin(write=True) as txn:
                    txn.put(key_bytes, data_bytes)
                keys_written.append(key_bytes)
                ok += 1
            else:
                tag = result.split("\t")[0]
                if tag == "SKIP":
                    skip += 1
                else:
                    err += 1
                    print(result, file=log_fh, flush=True)

    # Store the key list as a special metadata entry
    with env.begin(write=True) as txn:
        txn.put(b"__keys__", pickle.dumps(keys_written, protocol=pickle.HIGHEST_PROTOCOL))

    env.close()

    summary = (
        f"Done: {ok} written, {skip} skipped (filtered), {err} errors "
        f"out of {len(complex_ids)} complexes → {out_path}"
    )
    print(summary, flush=True)
    if log_fh is not sys.stdout:
        print(summary, file=log_fh)
        log_fh.close()


if __name__ == "__main__":
    main()
