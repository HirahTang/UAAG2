"""
UAAG2DatasetPDB
===============
On-the-fly training dataset that reads pre-cleaned PDB files and builds
residue-pocket graphs at training time.  Outputs the exact same ``Data``
format as UAAG2Dataset (LMDB-backed) so the training loop is unchanged.

Key speed optimisations over the original build_eqgat_lmdb_from_pdb pipeline:
  1. Pocket–pocket edge construction uses scipy cdist — one C call instead of
     an O(N²) pure-Python nested loop.  (~60-100× faster on that step.)
  2. Neighbour search is fully vectorised with numpy broadcasting.
  3. Parsed PDB structures (atom_features, bond_features, residues) are cached
     in a per-worker LRU cache so the RDKit + BioPython parse is paid only
     once per PDB per worker process, not once per residue or per epoch.
  4. ProteinMPNN latent normalised-key maps are built once per file load and
     cached alongside the data, rather than rebuilt on every residue lookup.
  5. The flat (pdb_path, aa_ordinal) index is saved to disk after the first
     construction so startup time on subsequent runs is negligible.

Usage:
    dataset = UAAG2DatasetPDB(
        pdb_dir="/scratch/project_465002574/PDB/PDB_processed",
        latent_root_128="/scratch/project_465002574/PDB/PDB_128",
        latent_root_20="/scratch/project_465002574/PDB/PDB_20",
        mask_rate=0.0,
        params=hparams,
    )
"""

import gc
import json
import os
import pickle
import sys
import tempfile
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import lmdb
import numpy as np
import torch
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.Polypeptide import is_aa
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import GetPeriodicTable
from scipy.spatial.distance import cdist
from torch_geometric.data import Data
from tqdm import tqdm

warnings.filterwarnings("ignore", category=PDBConstructionWarning)
RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")

# ---------------------------------------------------------------------------
# Encoders — must match build_eqgat_lmdb_from_pdb.py exactly
# ---------------------------------------------------------------------------
ATOM_ENCODER = {"C": 0, "N": 1, "O": 2, "S": 3, "P": 4, "Cl": 5, "F": 6, "Br": 7}
HYBRIDIZATION_ENCODER = {"SP": 0, "SP2": 1, "SP3": 2, "SP3D": 3}
BOND_ENCODER = {"SINGLE": 1, "DOUBLE": 2, "AROMATIC": 3, "TRIPLE": 4}
CHARGE_EMB = {-1: 0, 0: 1, 1: 2, 2: 3}

PERIODIC_TABLE = GetPeriodicTable()
SUPPORTED_LATENT_EXT = (".pt", ".pth", ".pkl", ".pickle", ".npy", ".npz", ".json")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ResidueRecord:
    res_id: int
    res_name: str
    chain_id: str
    resseq: int
    identity: str
    aa_order_index: Optional[int]
    is_amino_acid: bool
    atom_indices: List[int]
    mass_center: np.ndarray


# ---------------------------------------------------------------------------
# Element helpers
# ---------------------------------------------------------------------------
def _is_valid_element(sym: str) -> bool:
    if not sym:
        return False
    try:
        return PERIODIC_TABLE.GetAtomicNumber(sym.capitalize()) > 0
    except Exception:
        return False


def _atomic_number(sym: str) -> int:
    if not sym:
        return -1
    try:
        return int(PERIODIC_TABLE.GetAtomicNumber(sym.capitalize()))
    except Exception:
        return -1


def _infer_element_from_name(atom_name: str) -> str:
    token = "".join(c for c in atom_name.strip() if c.isalpha()).upper()
    if not token:
        return ""
    if len(token) >= 2 and _is_valid_element(token[:2].capitalize()):
        return token[:2].capitalize()
    if _is_valid_element(token[0]):
        return token[0]
    return ""


# ---------------------------------------------------------------------------
# PDB repair & parsing
# ---------------------------------------------------------------------------
def _needs_repair(pdb_path: str) -> bool:
    with open(pdb_path, "r", errors="ignore") as fh:
        for line in fh:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            padded = line.rstrip("\n")
            if len(padded) < 80:
                return True
            elem = padded[76:78].strip().capitalize()
            if not _is_valid_element(elem):
                return True
    return False


def _repair(pdb_path: str) -> str:
    lines = []
    with open(pdb_path, "r", errors="ignore") as fh:
        for line in fh:
            if line.startswith(("ATOM  ", "HETATM")):
                padded = line.rstrip("\n").ljust(80)
                elem = padded[76:78].strip().capitalize()
                if not _is_valid_element(elem):
                    elem = _infer_element_from_name(padded[12:16])
                if elem:
                    line = f"{padded[:76]}{elem.rjust(2)}{padded[78:]}\n"
            lines.append(line)
    fd, tmp = tempfile.mkstemp(prefix="uaag2_repair_", suffix=".pdb")
    os.close(fd)
    with open(tmp, "w") as fh:
        fh.writelines(lines)
    return tmp


def _read_mol_and_structure(pdb_path: str):
    parse_path = _repair(pdb_path) if _needs_repair(pdb_path) else pdb_path
    try:
        mol = Chem.MolFromPDBFile(parse_path, removeHs=True, sanitize=False)
        if mol is None and parse_path == pdb_path:
            parse_path = _repair(pdb_path)
            mol = Chem.MolFromPDBFile(parse_path, removeHs=True, sanitize=False)
        if mol is None:
            raise ValueError(f"RDKit could not parse {pdb_path}")
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistry(mol)
        structure = PDBParser(QUIET=True).get_structure("mol", parse_path)
        return mol, structure
    finally:
        if parse_path != pdb_path and os.path.exists(parse_path):
            try:
                os.remove(parse_path)
            except OSError:
                pass


def _extract_atom_bond_features(mol, structure) -> Tuple[Dict, Dict]:
    rdkit_atoms = list(mol.GetAtoms())
    bond_features: Dict[Tuple[int, int], Dict] = {}
    for bond in mol.GetBonds():
        bond_features[(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())] = {
            "Type": str(bond.GetBondType()),
            "Is Aromatic": bond.GetIsAromatic(),
        }

    # Force 'A' selection for DisorderedResidue objects (entire residue with
    # alternate conformations) so we never iterate 'B'-conformation atoms.
    for chain in structure[0]:
        for residue in chain:
            if residue.is_disordered() == 2 and "A" in residue.child_dict:
                residue.disordered_select("A")

    # Build a lookup keyed by (chain_id, resseq, icode, atom_name) for every
    # non-hydrogen BioPython atom.  When an atom exists in both 'A' and 'B'
    # conformations, prefer 'A'; otherwise keep whichever is present.
    _biopython_lookup: Dict[Tuple, object] = {}
    _biopython_meta: Dict[Tuple, Tuple] = {}
    for chain in structure[0]:
        for residue in chain:
            rname = residue.get_resname().strip()
            cid   = chain.id.strip()   # strip so blank chain ' ' matches RDKit ''
            rseq  = int(residue.id[1])
            icode = residue.id[2].strip()
            for atom in residue:
                elem = (atom.element or "").upper().strip()
                if elem in ("H", "D"):
                    continue
                aname  = atom.get_name().strip()
                altloc = getattr(atom, "altloc", " ").strip()
                key    = (cid, rseq, icode, aname)
                existing_altloc = getattr(_biopython_lookup.get(key), "altloc", "").strip()
                # Prefer 'A' (or blank) over 'B'; don't overwrite a primary atom
                if key not in _biopython_lookup or altloc in ("", "A") and existing_altloc not in ("", "A"):
                    _biopython_lookup[key] = atom
                    _biopython_meta[key]   = (rname, cid, rseq)

    # Re-order BioPython atoms to match RDKit's ordering using MonomerInfo.
    # This makes matching robust to any altloc/ordering discrepancies.
    pdb_atoms, res_id_list, residue_meta = [], [], []
    use_name_match = False
    for rdkit_atom in rdkit_atoms:
        info = rdkit_atom.GetMonomerInfo()
        if info is not None:
            use_name_match = True
            aname = info.GetName().strip()
            cid   = info.GetChainId().strip()
            rseq  = int(info.GetResidueNumber())
            icode = info.GetInsertionCode().strip()
            key   = (cid, rseq, icode, aname)
            atom  = _biopython_lookup.get(key)
            if atom is None:
                raise ValueError(
                    f"No BioPython atom for RDKit atom chain={cid!r} "
                    f"res={rseq} icode={icode!r} name={aname!r}"
                )
            rname, cid2, rseq2 = _biopython_meta[key]
            rid = f"{rname}_{rseq2}_{cid2}"
            pdb_atoms.append(atom)
            res_id_list.append(rid)
            residue_meta.append((rname, cid2, rseq2))

    if not use_name_match:
        # Fallback: positional match (old behaviour for files without MonomerInfo)
        for chain in structure[0]:
            for residue in chain:
                rname = residue.get_resname().strip()
                cid   = chain.id.strip()
                rseq  = int(residue.id[1])
                rid   = f"{rname}_{rseq}_{cid}"
                for atom in residue:
                    elem = (atom.element or "").upper().strip()
                    if elem in ("H", "D"):
                        continue
                    pdb_atoms.append(atom)
                    res_id_list.append(rid)
                    residue_meta.append((rname, cid, rseq))
        if len(rdkit_atoms) != len(pdb_atoms):
            raise ValueError(
                f"Atom count mismatch RDKit={len(rdkit_atoms)} PDB={len(pdb_atoms)}"
            )

    start_res_idx, start_res = 0, res_id_list[0]
    atom_features: Dict[int, Dict] = {}
    for idx, (rdkit_atom, pdb_atom) in enumerate(zip(rdkit_atoms, pdb_atoms)):
        pdb_z   = _atomic_number(str(pdb_atom.element).strip())
        rdkit_z = _atomic_number(rdkit_atom.GetSymbol())
        if pdb_z != rdkit_z:
            raise ValueError(
                f"Element mismatch at atom {idx}: "
                f"PDB={pdb_atom.element} RDKit={rdkit_atom.GetSymbol()}"
            )
        cur_res = res_id_list[idx]
        if cur_res != start_res:
            start_res = cur_res
            start_res_idx += 1
        rname, cid, rseq = residue_meta[idx]
        atom_features[idx] = {
            "idx": idx,
            "coords": pdb_atom.get_coord(),
            "Atoms": rdkit_atom.GetSymbol(),
            "Charges": rdkit_atom.GetFormalCharge(),
            "Hybridization": str(rdkit_atom.GetHybridization()),
            "Degree": rdkit_atom.GetDegree(),
            "Aromatic": rdkit_atom.GetIsAromatic(),
            "Residue": cur_res,
            "Res_name": rname,
            "Chain": cid,
            "Resseq": rseq,
            "Residue ID": start_res_idx,
        }

    return atom_features, bond_features


def _build_residue_records(atom_features: Dict) -> List[ResidueRecord]:
    grouped: Dict[int, list] = {}
    for atom in atom_features.values():
        grouped.setdefault(atom["Residue ID"], []).append(atom)

    records, aa_count = [], 0
    for res_id in sorted(grouped):
        atoms = grouped[res_id]
        first = atoms[0]
        coords = np.array([a["coords"] for a in atoms], dtype=float)
        is_amino = is_aa(first["Res_name"], standard=False)
        records.append(ResidueRecord(
            res_id=res_id,
            res_name=first["Res_name"],
            chain_id=first["Chain"],
            resseq=int(first["Resseq"]),
            identity=f"{first['Res_name']}_{first['Resseq']}_{first['Chain']}_{res_id}",
            aa_order_index=aa_count if is_amino else None,
            is_amino_acid=is_amino,
            atom_indices=[a["idx"] for a in atoms],
            mass_center=np.mean(coords, axis=0),
        ))
        if is_amino:
            aa_count += 1
    return records


# ---------------------------------------------------------------------------
# Module-level LRU cache for parsed structures
# Cached per worker process (fork-safe).  Size is set at import time;
# call _set_parse_cache_size() before constructing any dataset to resize.
# ---------------------------------------------------------------------------
_PARSE_CACHE_SIZE = 64


def _set_parse_cache_size(n: int):
    """Call once at startup if you want a different cache size."""
    global _parse_pdb_cached, _PARSE_CACHE_SIZE
    _PARSE_CACHE_SIZE = n
    _parse_pdb_cached = lru_cache(maxsize=n)(_parse_pdb_cached.__wrapped__)


@lru_cache(maxsize=_PARSE_CACHE_SIZE)
def _parse_pdb_cached(pdb_path: str):
    """Parse a PDB file and return (atom_features, bond_features, residues).

    Result is cached per worker process so subsequent residues from the same
    PDB pay zero I/O or RDKit cost.
    """
    mol, structure = _read_mol_and_structure(pdb_path)
    atom_features, bond_features = _extract_atom_bond_features(mol, structure)
    residues = _build_residue_records(atom_features)
    return atom_features, bond_features, residues


# ---------------------------------------------------------------------------
# Vectorised graph construction helpers
# ---------------------------------------------------------------------------
def _get_neighbors_fast(
    all_residues: List[ResidueRecord],
    center_idx: int,
    radius: float,
) -> List[ResidueRecord]:
    """Return residues within `radius` Å of center_residue (vectorised)."""
    centers = np.array([r.mass_center for r in all_residues])   # (R, 3)
    center = all_residues[center_idx].mass_center
    diffs = np.abs(centers - center)
    bbox_pass = np.all(diffs <= radius, axis=1)
    candidates = np.where(bbox_pass)[0]
    norms = np.linalg.norm(centers[candidates] - center, axis=1)
    within = candidates[norms <= radius]
    return [all_residues[i] for i in within if i != center_idx]


def _has_supported_features(atoms) -> bool:
    for a in atoms:
        if a["Atoms"] not in ATOM_ENCODER:
            return False
        if a["Hybridization"] not in HYBRIDIZATION_ENCODER:
            return False
    return True


def _build_graph(
    center_residue: ResidueRecord,
    center_idx: int,
    all_residues: List[ResidueRecord],
    atom_features: Dict,
    bond_features: Dict,
    pocket_radius: float,
    edge_radius: float,
    latent_128: Optional[np.ndarray],
    latent_20: Optional[np.ndarray],
    compound_id: str,
    source_name: str = "",
) -> Optional[Data]:
    """Build a PyG Data object for one AA residue.  Returns None if invalid."""
    # ---- ligand atoms / bonds ----
    lig_atoms = [atom_features[i].copy() for i in center_residue.atom_indices]
    lig_idx_set = set(center_residue.atom_indices)
    lig_bonds = [
        b for b in bond_features.items()
        if b[0][0] in lig_idx_set or b[0][1] in lig_idx_set
    ]
    for a in lig_atoms:
        a["coords"] = [float(v) for v in a["coords"]]

    if len(lig_atoms) < 4:
        return None

    # ---- pocket neighbours ----
    neighbors = _get_neighbors_fast(all_residues, center_idx, pocket_radius)
    pkt_atoms = []
    for res in neighbors:
        for i in res.atom_indices:
            a = atom_features[i].copy()
            a["coords"] = [float(v) for v in a["coords"]]
            pkt_atoms.append(a)
    pkt_idx_set = {a["idx"] for a in pkt_atoms}
    pkt_bonds = [
        b for b in bond_features.items()
        if b[0][0] in pkt_idx_set and b[0][1] in pkt_idx_set
    ]

    # ---- filter: skip if any atom has unsupported type / hybridisation ----
    if not _has_supported_features(lig_atoms) or not _has_supported_features(pkt_atoms):
        return None

    # ---- unified atom index mapping ----
    all_ids = sorted(
        {a["idx"] for a in lig_atoms} | {a["idx"] for a in pkt_atoms}
    )
    id_map = {aid: i for i, aid in enumerate(all_ids)}
    num_atoms = len(all_ids)

    if num_atoms > 400:
        return None

    atom_types   = torch.zeros(num_atoms, dtype=torch.long)
    charges      = torch.zeros(num_atoms, dtype=torch.float)
    is_aromatic  = torch.zeros(num_atoms, dtype=torch.float)
    hybridization = torch.zeros(num_atoms, dtype=torch.long)
    degree       = torch.zeros(num_atoms, dtype=torch.long)
    position     = torch.zeros(num_atoms, 3, dtype=torch.float)
    is_ligand    = torch.zeros(num_atoms, dtype=torch.float)
    is_backbone  = torch.zeros(num_atoms, dtype=torch.float)

    # ligand atoms
    lig_new_ids = []
    for k, a in enumerate(lig_atoms):
        nid = id_map[a["idx"]]
        lig_new_ids.append(nid)
        atom_types[nid]    = ATOM_ENCODER[a["Atoms"]]
        charges[nid]       = float(a["Charges"])
        is_aromatic[nid]   = float(a["Aromatic"])
        hybridization[nid] = HYBRIDIZATION_ENCODER[a["Hybridization"]]
        degree[nid]        = int(a["Degree"])
        position[nid]      = torch.tensor(a["coords"], dtype=torch.float)
        is_ligand[nid]     = 1.0
        if k < 4:
            is_backbone[nid] = 1.0

    # pocket atoms
    pkt_keys = [a["idx"] for a in pkt_atoms]
    for a in pkt_atoms:
        nid = id_map[a["idx"]]
        atom_types[nid]    = ATOM_ENCODER[a["Atoms"]]
        charges[nid]       = float(a["Charges"])
        is_aromatic[nid]   = float(a["Aromatic"])
        hybridization[nid] = HYBRIDIZATION_ENCODER[a["Hybridization"]]
        degree[nid]        = int(a["Degree"])
        position[nid]      = torch.tensor(a["coords"], dtype=torch.float)

    # ---- edge construction ----
    edge_src, edge_dst, edge_lig_flag = [], [], []

    # ligand–ligand (fully connected)
    for i in range(len(lig_new_ids)):
        for j in range(i + 1, len(lig_new_ids)):
            edge_src += [lig_new_ids[i], lig_new_ids[j]]
            edge_dst += [lig_new_ids[j], lig_new_ids[i]]
            edge_lig_flag += [1, 1]

    # pocket–pocket: vectorised cdist
    if len(pkt_atoms) > 0:
        pkt_coords = np.array([a["coords"] for a in pkt_atoms])   # (P, 3)
        dist_mat = cdist(pkt_coords, pkt_coords)                   # (P, P)
        ii, jj = np.where(np.triu(dist_mat < edge_radius, k=1))
        pkt_new_ids = [id_map[k] for k in pkt_keys]
        for i, j in zip(ii.tolist(), jj.tolist()):
            edge_src += [pkt_new_ids[i], pkt_new_ids[j]]
            edge_dst += [pkt_new_ids[j], pkt_new_ids[i]]
            edge_lig_flag += [0, 0]

        # ligand–pocket cross edges
        pkt_new_set = pkt_new_ids
        for lid in lig_new_ids:
            for pid in pkt_new_set:
                edge_src += [lid, pid]
                edge_dst += [pid, lid]
                edge_lig_flag += [0, 0]

    if not edge_src:
        return None

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_type  = torch.zeros(edge_index.size(1), dtype=torch.long)

    # assign bond types
    lig_bonds_d = {}
    for b in lig_bonds:
        i, j = id_map.get(b[0][0]), id_map.get(b[0][1])
        if i is not None and j is not None:
            lig_bonds_d[(i, j)] = b[1]["Type"]

    pkt_bonds_d = {}
    for b in pkt_bonds:
        i, j = id_map.get(b[0][0]), id_map.get(b[0][1])
        if i is not None and j is not None:
            pkt_bonds_d[(i, j)] = b[1]["Type"]

    for k in range(edge_index.size(1)):
        src, dst = int(edge_index[0, k]), int(edge_index[1, k])
        bt = lig_bonds_d.get((src, dst)) or lig_bonds_d.get((dst, src)) \
             or pkt_bonds_d.get((src, dst)) or pkt_bonds_d.get((dst, src))
        if bt and bt in BOND_ENCODER:
            edge_type[k] = BOND_ENCODER[bt]

    edge_ligand = torch.tensor(edge_lig_flag, dtype=torch.float)

    # ---- ProteinMPNN latents ----
    lat128_node = None
    if latent_128 is not None:
        t = torch.tensor(latent_128, dtype=torch.float)
        lat128_node = t.unsqueeze(0).expand(num_atoms, -1)

    lat20_node = None
    if latent_20 is not None:
        t = torch.tensor(latent_20, dtype=torch.float)
        lat20_node = t.unsqueeze(0).expand(num_atoms, -1)

    kwargs = dict(
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
        id=torch.tensor(all_ids, dtype=torch.long),
        ids=torch.arange(num_atoms, dtype=torch.long),
        compound_id=compound_id,
        source_name=source_name,
        center_residue=f"{center_residue.res_name}_{center_residue.resseq}_{center_residue.chain_id}",
        residue_name=center_residue.res_name,
        protein_mpnn_latent_128=torch.tensor(latent_128, dtype=torch.float) if latent_128 is not None else None,
        protein_mpnn_latent_20=torch.tensor(latent_20,  dtype=torch.float) if latent_20  is not None else None,
        protein_mpnn_latent_node_128=lat128_node,
        protein_mpnn_latent_node_20=lat20_node,
    )
    return Data(**{k: v for k, v in kwargs.items() if v is not None})


# ---------------------------------------------------------------------------
# LatentStore — identical logic to build_eqgat_lmdb_from_pdb.py but with
# the normalised-key map cached alongside the raw payload so it is never
# rebuilt twice for the same file.
# ---------------------------------------------------------------------------
def _normalize_key(raw: str) -> str:
    return "".join(c for c in raw.upper() if c.isalnum())


class LatentStore:
    def __init__(self, root: str, latent_dim: int, max_cache_files: int = 32):
        self.root = root
        self.latent_dim = latent_dim
        self.max_cache = max(1, max_cache_files)
        self.path_index = self._index()
        # cache: path -> (raw_payload, norm_key_map_or_None)
        self.cache: OrderedDict = OrderedDict()

    def _index(self):
        idx = {}
        for dirpath, _, filenames in os.walk(self.root):
            for name in filenames:
                if name.endswith(SUPPORTED_LATENT_EXT):
                    idx[os.path.splitext(name)[0]] = os.path.join(dirpath, name)
        return idx

    def _load(self, path: str):
        if path.endswith((".pt", ".pth")):
            return torch.load(path, map_location="cpu")
        if path.endswith((".pkl", ".pickle")):
            with open(path, "rb") as fh:
                return pickle.load(fh)
        if path.endswith(".npy"):
            return np.load(path, allow_pickle=True)
        if path.endswith(".npz"):
            return np.load(path, allow_pickle=True)
        if path.endswith(".json"):
            with open(path) as fh:
                return json.load(fh)
        raise ValueError(f"Unsupported: {path}")

    def _to_residue_map(self, payload):
        if isinstance(payload, np.lib.npyio.NpzFile):
            try:
                if "latents" in payload and "keys" in payload:
                    lat, keys = payload["latents"], payload["keys"]
                    return {str(k): np.asarray(lat[i]) for i, k in enumerate(keys)}
                return np.asarray(payload[list(payload.keys())[0]])
            finally:
                payload.close()
        if isinstance(payload, dict):
            if "latents" in payload and "keys" in payload:
                lat, keys = payload["latents"], payload["keys"]
                return {str(k): lat[i] for i, k in enumerate(keys)}
            if "embeddings" in payload and "residue_ids" in payload:
                lat, keys = payload["embeddings"], payload["residue_ids"]
                return {str(k): lat[i] for i, k in enumerate(keys)}
            return payload
        return payload

    def _candidate_stems(self, pdb_name: str) -> List[str]:
        stem = os.path.splitext(pdb_name)[0]
        stems = [pdb_name, stem]
        if stem.endswith("_tidy"):
            stems.append(stem[: -len("_tidy")])
        return list(dict.fromkeys(stems))

    def get_latent(self, pdb_name: str, residue: ResidueRecord) -> Optional[np.ndarray]:
        for stem in self._candidate_stems(pdb_name):
            path = self.path_index.get(stem)
            if not path:
                continue
            entry = self.cache.get(path)
            if entry is None:
                raw = self._to_residue_map(self._load(path))
                # Build and cache the normalised-key map once per file load
                norm = (
                    {_normalize_key(str(k)): v for k, v in raw.items()}
                    if isinstance(raw, dict) else None
                )
                entry = (raw, norm)
                self.cache[path] = entry
                if len(self.cache) > self.max_cache:
                    self.cache.popitem(last=False)
            else:
                self.cache.move_to_end(path)

            raw, norm = entry
            if isinstance(raw, dict) and norm is not None:
                variants = [
                    residue.identity,
                    f"{residue.res_name}_{residue.resseq}_{residue.chain_id}",
                    f"{residue.chain_id}_{residue.resseq}",
                    f"{residue.chain_id}:{residue.resseq}",
                    f"{residue.resseq}_{residue.chain_id}",
                    str(residue.resseq),
                ]
                for key in variants:
                    val = norm.get(_normalize_key(key))
                    if val is None:
                        continue
                    arr = np.asarray(val).astype(np.float32).reshape(-1)
                    if arr.shape[0] == self.latent_dim:
                        return arr
            elif isinstance(raw, np.ndarray) and residue.aa_order_index is not None:
                if raw.ndim == 2 and residue.aa_order_index < raw.shape[0]:
                    if raw.shape[1] == self.latent_dim:
                        return raw[residue.aa_order_index].astype(np.float32)
        return None


# ---------------------------------------------------------------------------
# Fast AA residue counter (no RDKit, just file scan)
# ---------------------------------------------------------------------------
def _count_aa_residues_fast(pdb_path: str) -> int:
    seen: set = set()
    with open(pdb_path, "r", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("ATOM  "):
                continue
            resname = line[17:20].strip()
            if not is_aa(resname, standard=False):
                continue
            key = (line[21], line[22:26].strip(), line[26].strip())
            seen.add(key)
    return len(seen)


# ---------------------------------------------------------------------------
# Shared post-processing (used by both UAAG2DatasetPDB and PDBBindDataset)
# ---------------------------------------------------------------------------
def _post_process_graph(
    graph_data: Data,
    *,
    charge_emb: dict,
    mask_rate: float = 0.0,
    pocket_noise: bool = False,
    noise_scale: float = 0.1,
    pocket_dropout_prob: float = 0.0,
    params=None,
) -> Data:
    """Centre, cast, mask backbone, optionally add virtual nodes.

    Identical logic to the former UAAG2DatasetPDB._post_process; extracted
    so that PDBBindDataset can reuse it without inheriting the full class.
    """
    # --- Pocket dropout ---
    # With probability pocket_dropout_prob, remove all pocket atoms (is_ligand==0),
    # keeping only the centre residue atoms (is_ligand==1, including backbone N/CA/C/O).
    if pocket_dropout_prob > 0.0 and torch.rand(1).item() < pocket_dropout_prob:
        keep = (graph_data.is_ligand > 0.5) | (graph_data.is_backbone > 0.5)
        if not keep.all():
            n_orig = graph_data.x.size(0)
            new_id = torch.full((n_orig,), -1, dtype=torch.long)
            keep_idx = torch.where(keep)[0]
            new_id[keep_idx] = torch.arange(keep.sum(), dtype=torch.long)
            # Filter and remap edges
            src_e, dst_e = graph_data.edge_index
            edge_keep = keep[src_e] & keep[dst_e]
            graph_data.edge_index = new_id[graph_data.edge_index[:, edge_keep]]
            for attr in ("edge_attr", "edge_ligand"):
                t = getattr(graph_data, attr, None)
                if t is not None and isinstance(t, torch.Tensor):
                    setattr(graph_data, attr, t[edge_keep])
            # Filter node features
            for attr in ("x", "pos", "charges", "degree", "is_aromatic",
                         "hybridization", "is_backbone", "is_ligand"):
                t = getattr(graph_data, attr, None)
                if t is not None and isinstance(t, torch.Tensor) and t.size(0) == n_orig:
                    setattr(graph_data, attr, t[keep])
            # Filter latent if present
            lat = getattr(graph_data, "protein_mpnn_latent_node_128", None)
            if lat is not None and isinstance(lat, torch.Tensor) and lat.size(0) == n_orig:
                graph_data.protein_mpnn_latent_node_128 = lat[keep]

    # Centre positions on pocket mean
    if graph_data.is_ligand.sum() < len(graph_data.is_ligand):
        pocket_mean = graph_data.pos[graph_data.is_ligand == 0].mean(dim=0)
    else:
        pocket_mean = graph_data.pos.mean(dim=0)
    graph_data.pos = graph_data.pos - pocket_mean

    CoM = graph_data.pos[graph_data.is_ligand == 1].mean(dim=0)

    # Cast
    graph_data.x             = graph_data.x.float()
    graph_data.pos           = graph_data.pos.float()
    graph_data.edge_attr      = graph_data.edge_attr.float()
    graph_data.edge_index     = graph_data.edge_index.long()
    graph_data.degree         = graph_data.degree.float()
    graph_data.is_aromatic    = graph_data.is_aromatic.float()
    graph_data.hybridization  = graph_data.hybridization.float()
    graph_data.is_backbone    = graph_data.is_backbone.float()
    graph_data.is_ligand      = graph_data.is_ligand.float()

    # Map charges
    charges_np = graph_data.charges.numpy()
    graph_data.charges = torch.from_numpy(
        np.vectorize(charge_emb.get)(charges_np)
    ).float()

    # Handle latent shape
    lat128 = None
    if hasattr(graph_data, "protein_mpnn_latent_node_128") and graph_data.protein_mpnn_latent_node_128 is not None:
        lat128 = graph_data.protein_mpnn_latent_node_128.float()
        if lat128.dim() == 1:
            lat128 = lat128.unsqueeze(0)

    # Backbone masking
    reconstruct_mask = graph_data.is_ligand - graph_data.is_backbone
    new_bb = torch.bernoulli(
        torch.ones(reconstruct_mask.eq(1).sum()) * mask_rate
    ).float()
    graph_data.is_backbone[reconstruct_mask == 1] = new_bb

    # Optional pocket noise
    if pocket_noise:
        noise = torch.randn_like(graph_data.pos[reconstruct_mask == 1]) * noise_scale
        graph_data.pos[reconstruct_mask == 1] += noise

    # Virtual nodes
    if params is not None and params.virtual_node:
        sample_n = int(np.random.randint(1, params.max_virtual_nodes))
        vx    = torch.ones(sample_n) * 8
        vpos  = CoM.unsqueeze(0).expand(sample_n, -1).clone()
        graph_data.x            = torch.cat([graph_data.x,           vx])
        graph_data.pos          = torch.cat([graph_data.pos,          vpos])
        graph_data.charges      = torch.cat([graph_data.charges,      torch.ones(sample_n)])
        graph_data.degree       = torch.cat([graph_data.degree,       torch.zeros(sample_n)])
        graph_data.is_aromatic  = torch.cat([graph_data.is_aromatic,  torch.zeros(sample_n)])
        graph_data.hybridization= torch.cat([graph_data.hybridization,torch.zeros(sample_n)])
        graph_data.is_backbone  = torch.cat([graph_data.is_backbone,  torch.zeros(sample_n)])
        graph_data.is_ligand    = torch.cat([graph_data.is_ligand,    torch.ones(sample_n)])

        all_ids   = torch.arange(len(graph_data.x))
        virt_ids  = all_ids[-sample_n:]
        exist_ids = all_ids[:-sample_n]

        g1, g2 = torch.meshgrid(virt_ids, exist_ids, indexing="ij")
        bi_new = torch.cat([
            torch.stack([g1.flatten(), g2.flatten()]),
            torch.stack([g2.flatten(), g1.flatten()]),
        ], dim=1)

        g1v, g2v = torch.meshgrid(virt_ids, virt_ids, indexing="ij")
        mask_v = g1v != g2v
        bi_vv = torch.stack([g1v[mask_v], g2v[mask_v]])

        new_ei = torch.cat([graph_data.edge_index, bi_new, bi_vv], dim=1)
        n_new  = bi_new.size(1) + bi_vv.size(1)
        new_ea = torch.cat([graph_data.edge_attr,  torch.zeros(n_new)])
        new_el = torch.cat([
            torch.tensor(graph_data.edge_ligand).float(),
            torch.zeros(bi_new.size(1)),
            torch.ones(bi_vv.size(1)),
        ])
        graph_data.edge_index = new_ei
        graph_data.edge_attr  = new_ea
        graph_data.edge_ligand = new_el

    # Final dtype pass (matches UAAG2Dataset exactly)
    graph_data.degree        = graph_data.degree.float()
    graph_data.is_aromatic   = graph_data.is_aromatic.float()
    graph_data.hybridization = graph_data.hybridization.float()
    graph_data.is_backbone   = graph_data.is_backbone.float()
    graph_data.is_ligand     = graph_data.is_ligand.float()

    num_nodes = graph_data.x.size(0)
    if lat128 is not None and lat128.size(0) != num_nodes:
        lat128 = lat128[0].unsqueeze(0).expand(num_nodes, -1)

    data_kwargs = dict(
        x=graph_data.x,
        pos=graph_data.pos,
        edge_index=graph_data.edge_index,
        edge_attr=graph_data.edge_attr,
        edge_ligand=torch.tensor(graph_data.edge_ligand).float(),
        charges=graph_data.charges,
        degree=graph_data.degree,
        is_aromatic=graph_data.is_aromatic,
        hybridization=graph_data.hybridization,
        is_backbone=graph_data.is_backbone,
        is_ligand=graph_data.is_ligand,
        ligand_size=torch.tensor(
            int(graph_data.is_ligand.sum() - graph_data.is_backbone.sum())
        ).long(),
        id=graph_data.compound_id,
    )
    if lat128 is not None:
        data_kwargs["protein_mpnn_latent_node_128"] = lat128

    return Data(**data_kwargs)


# ---------------------------------------------------------------------------
# Main Dataset
# ---------------------------------------------------------------------------
class UAAG2DatasetPDB(torch.utils.data.Dataset):
    """On-the-fly PDB → residue-pocket graph dataset.

    Drop-in replacement for UAAG2Dataset; outputs identical Data objects.

    Parameters
    ----------
    pdb_dir : str
        Directory containing pre-cleaned .pdb files
        (e.g. /scratch/project_465002574/PDB/PDB_processed).
    latent_root_128, latent_root_20 : str
        Directories with pre-computed ProteinMPNN embeddings (dim 128 / 20).
        Pass empty string "" to run without latents.
    pocket_radius : float
        Radius (Å) for selecting pocket residues around the centre AA.
    edge_radius : float
        Distance threshold (Å) for pocket–pocket edges.
    mask_rate : float
        Fraction of backbone atoms to randomly mask (0 = reconstruct all).
    pocket_noise : bool / noise_scale : float
        Add Gaussian noise to masked positions during training.
    params : object
        Hparams object; must expose ``params.virtual_node`` (bool) and
        ``params.max_virtual_nodes`` (int).
    latent_cache_files : int
        Max latent files cached in RAM per LatentStore per worker.
    index_cache_path : str or None
        Path to cache the flat (pdb_path, aa_ordinal) index.  Saves ~30 s
        on subsequent runs over 67 k PDBs.  Defaults to
        ``{pdb_dir}/.uaag2_index_cache.pkl``.
    parse_cache_size : int
        LRU cache size for parsed PDB structures per worker process.
    max_retries : int
        Number of adjacent indices to try before raising if a graph cannot
        be built (e.g. unsupported atoms, RDKit parse failure).
    """

    def __init__(
        self,
        pdb_dir: str,
        latent_root_128: str = "",
        latent_root_20: str = "",
        pocket_radius: float = 10.0,
        edge_radius: float = 8.0,
        mask_rate: float = 0.0,
        pocket_noise: bool = False,
        noise_scale: float = 0.1,
        pocket_dropout_prob: float = 0.0,
        params=None,
        latent_cache_files: int = 32,
        index_cache_path: Optional[str] = None,
        parse_cache_size: int = 64,
        max_retries: int = 100,
        pdb_fraction: float = 1.0,
        max_pdb_files: int = 0,
    ):
        super().__init__()
        self.pdb_dir       = pdb_dir
        self.pocket_radius = pocket_radius
        self.edge_radius   = edge_radius
        self.mask_rate     = mask_rate
        self.pocket_noise  = pocket_noise
        self.noise_scale   = noise_scale
        self.pocket_dropout_prob = pocket_dropout_prob
        self.params        = params
        self.max_retries   = max_retries

        # Resize the module-level LRU cache if requested
        if parse_cache_size != _PARSE_CACHE_SIZE:
            _set_parse_cache_size(parse_cache_size)

        # LatentStores (None if no root given)
        self.latent_128 = (
            LatentStore(latent_root_128, 128, latent_cache_files)
            if latent_root_128 else None
        )
        self.latent_20 = (
            LatentStore(latent_root_20, 20, latent_cache_files)
            if latent_root_20 else None
        )

        # Build or load flat index
        _base_cache = index_cache_path or os.path.join(pdb_dir, ".uaag2_index_cache.pkl")
        if max_pdb_files > 0:
            cache_path = _base_cache.replace(".pkl", f"_top{max_pdb_files}.pkl")
        else:
            cache_path = _base_cache
        self._flat_index: List[Tuple[str, int]] = self._build_or_load_index(cache_path, max_pdb_files=max_pdb_files)
        if 0.0 < pdb_fraction < 1.0:
            n = max(1, int(len(self._flat_index) * pdb_fraction))
            self._flat_index = self._flat_index[:n]
            print(f"[UAAG2DatasetPDB] Subset {pdb_fraction:.0%}: using {n:,} residues")

        self.charge_emb = CHARGE_EMB

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------
    def _build_or_load_index(self, cache_path: str, max_pdb_files: int = 0) -> List[Tuple[str, int]]:
        if os.path.isfile(cache_path):
            try:
                with open(cache_path, "rb") as fh:
                    idx = pickle.load(fh)
                print(f"[UAAG2DatasetPDB] Loaded index ({len(idx):,} residues) from {cache_path}")
                return idx
            except Exception as exc:
                print(f"[UAAG2DatasetPDB] Could not load index cache ({exc}), rebuilding …")

        pdb_files = sorted(
            os.path.join(self.pdb_dir, f)
            for f in os.listdir(self.pdb_dir)
            if f.lower().endswith(".pdb")
        )
        if max_pdb_files > 0:
            pdb_files = pdb_files[:max_pdb_files]
        print(f"[UAAG2DatasetPDB] Building flat index over {len(pdb_files):,} PDB files …")
        flat: List[Tuple[str, int]] = []
        for pdb_path in tqdm(pdb_files, desc="Indexing PDBs"):
            try:
                n = _count_aa_residues_fast(pdb_path)
            except Exception:
                continue
            for i in range(n):
                flat.append((pdb_path, i))

        try:
            with open(cache_path, "wb") as fh:
                pickle.dump(flat, fh, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[UAAG2DatasetPDB] Saved index ({len(flat):,} residues) to {cache_path}")
        except Exception as exc:
            print(f"[UAAG2DatasetPDB] Could not save index: {exc}")

        return flat

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._flat_index)

    def __getitem__(self, idx: int) -> Data:
        n = len(self)
        for attempt in range(self.max_retries):
            try:
                # First attempt: exact index. Subsequent: random index
                # to escape clusters of bad residues in a single PDB file.
                i = idx if attempt == 0 else int(torch.randint(n, (1,)).item())
                graph = self._try_get(i)
                if graph is not None:
                    return graph
            except Exception:
                pass
        raise RuntimeError(
            f"[UAAG2DatasetPDB] Could not build a valid graph "
            f"near index {idx} after {self.max_retries} attempts."
        )

    def _try_get(self, idx: int) -> Optional[Data]:
        pdb_path, aa_ordinal = self._flat_index[idx]
        pdb_name = os.path.basename(pdb_path)

        atom_features, bond_features, residues = _parse_pdb_cached(pdb_path)

        aa_residues = [r for r in residues if r.is_amino_acid]
        if aa_ordinal >= len(aa_residues):
            return None
        residue   = aa_residues[aa_ordinal]
        center_idx = residues.index(residue)

        # Latent lookup
        lat128 = self.latent_128.get_latent(pdb_name, residue) if self.latent_128 else None
        lat20  = self.latent_20.get_latent(pdb_name, residue)  if self.latent_20  else None

        compound_id = f"{os.path.splitext(pdb_name)[0]}_{residue.identity}"
        graph = _build_graph(
            center_residue=residue,
            center_idx=center_idx,
            all_residues=residues,
            atom_features=atom_features,
            bond_features=bond_features,
            pocket_radius=self.pocket_radius,
            edge_radius=self.edge_radius,
            latent_128=lat128,
            latent_20=lat20,
            compound_id=compound_id,
            source_name=pdb_name,
        )
        if graph is None:
            return None

        return self._post_process(graph)

    # ------------------------------------------------------------------
    # Post-processing — delegates to module-level _post_process_graph
    # ------------------------------------------------------------------
    def _post_process(self, graph_data: Data) -> Data:
        return _post_process_graph(
            graph_data,
            charge_emb=self.charge_emb,
            mask_rate=self.mask_rate,
            pocket_noise=self.pocket_noise,
            noise_scale=self.noise_scale,
            pocket_dropout_prob=self.pocket_dropout_prob,
            params=self.params,
        )


# ---------------------------------------------------------------------------
# PDBBind Dataset — reads pre-built graphs from an LMDB written by
# build_pdbbind_lmdb.py; applies _post_process at __getitem__ time.
# ---------------------------------------------------------------------------
class PDBBindDataset(torch.utils.data.Dataset):
    """Dataset backed by PDBBind.lmdb.

    Each LMDB entry is a raw (pre-_post_process) pickled PyG Data object
    representing one protein-ligand complex (pocket + small molecule).
    The ligand plays the same role as a target residue in UAAG2DatasetPDB:
    ``is_ligand=1`` for all ligand atoms, ``is_ligand=0`` for pocket atoms.

    Parameters
    ----------
    lmdb_path : str
        Path to PDBBind.lmdb (a flat LMDB file, not a directory).
    mask_rate, pocket_noise, noise_scale, params :
        Passed through to _post_process_graph — same semantics as in
        UAAG2DatasetPDB.
    """

    def __init__(
        self,
        lmdb_path: str,
        mask_rate: float = 0.0,
        pocket_noise: bool = False,
        noise_scale: float = 0.1,
        pocket_dropout_prob: float = 0.0,
        params=None,
    ):
        super().__init__()
        self.lmdb_path   = lmdb_path
        self.mask_rate   = mask_rate
        self.pocket_noise = pocket_noise
        self.noise_scale = noise_scale
        self.pocket_dropout_prob = pocket_dropout_prob
        self.params      = params
        self.charge_emb  = CHARGE_EMB

        # Read the key list eagerly (fast — stored as a single LMDB entry).
        env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False,
                        map_size=20 * 1024 ** 3)
        with env.begin() as txn:
            meta = txn.get(b"__keys__")
            if meta is not None:
                self._keys: List[bytes] = pickle.loads(meta)
            else:
                self._keys = [k for k, _ in txn.cursor() if not k.startswith(b"__")]
        env.close()
        self._env = None  # re-opened lazily inside each worker

    # ------------------------------------------------------------------
    # Lazy LMDB handle — safe across fork() because we open after fork
    # ------------------------------------------------------------------
    def _get_env(self):
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path, readonly=True, lock=False, subdir=False,
                map_size=20 * 1024 ** 3,
            )
        return self._env

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, idx: int) -> Data:
        env = self._get_env()
        with env.begin() as txn:
            val = txn.get(self._keys[idx])
        if val is None:
            raise KeyError(f"[PDBBindDataset] LMDB key not found: {self._keys[idx]}")
        graph = pickle.loads(val)
        return _post_process_graph(
            graph,
            charge_emb=self.charge_emb,
            mask_rate=self.mask_rate,
            pocket_noise=self.pocket_noise,
            noise_scale=self.noise_scale,
            pocket_dropout_prob=self.pocket_dropout_prob,
            params=self.params,
        )


# ---------------------------------------------------------------------------
# CombinedPDBDataset — wraps UAAG2DatasetPDB and/or PDBBindDataset so the
# training loop can sample from either or both with a single Dataset object.
# ---------------------------------------------------------------------------
class CombinedPDBDataset(torch.utils.data.Dataset):
    """Union of an on-the-fly PDB dataset and a pre-built PDBBind LMDB dataset.

    Pass ``pdb_dataset=None`` to use only PDBBind, or ``pdbbind_dataset=None``
    to use only the on-the-fly PDB dataset.  Both can be provided to train on
    the combined pool (indices 0..len(pdb)-1 map to pdb_dataset; remaining
    indices map to pdbbind_dataset).

    Parameters
    ----------
    pdb_dataset : UAAG2DatasetPDB or None
    pdbbind_dataset : PDBBindDataset or None
    """

    def __init__(
        self,
        pdb_dataset: Optional["UAAG2DatasetPDB"] = None,
        pdbbind_dataset: Optional[PDBBindDataset] = None,
    ):
        if pdb_dataset is None and pdbbind_dataset is None:
            raise ValueError("At least one of pdb_dataset or pdbbind_dataset must be provided.")
        super().__init__()
        self.pdb_dataset     = pdb_dataset
        self.pdbbind_dataset = pdbbind_dataset
        self._pdb_len        = len(pdb_dataset)     if pdb_dataset     is not None else 0
        self._pdbbind_len    = len(pdbbind_dataset)  if pdbbind_dataset is not None else 0

    def __len__(self) -> int:
        return self._pdb_len + self._pdbbind_len

    def __getitem__(self, idx: int) -> Data:
        if idx < self._pdb_len:
            return self.pdb_dataset[idx]
        return self.pdbbind_dataset[idx - self._pdb_len]
