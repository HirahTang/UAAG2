import argparse
import json
import lmdb
import os
import pickle
import tempfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from rdkit import Chem
from rdkit.Chem.rdchem import GetPeriodicTable
from torch_geometric.data import Data
from tqdm import tqdm


ATOM_ENCODER = {
    "C": 0,
    "N": 1,
    "O": 2,
    "S": 3,
    "P": 4,
    "Cl": 5,
    "F": 6,
    "Br": 7,
}

HYBRIDIZATION_ENCODER = {
    "SP": 0,
    "SP2": 1,
    "SP3": 2,
    "SP3D": 3,
}

BOND_ENCODER = {
    "SINGLE": 1,
    "DOUBLE": 2,
    "AROMATIC": 3,
    "TRIPLE": 4,
}

AA_DICT = {
    "ALA": 0,
    "ASX": 1,
    "CYS": 2,
    "ASP": 3,
    "GLU": 4,
    "PHE": 5,
    "GLY": 6,
    "HIS": 7,
    "ILE": 8,
    "LYS": 9,
    "LEU": 10,
    "MET": 11,
    "ASN": 12,
    "PRO": 13,
    "GLN": 14,
    "ARG": 15,
    "SER": 16,
    "THR": 17,
    "VAL": 18,
    "TRP": 19,
    "TYR": 20,
    "GLX": 21,
    "SEC": 22,
    "UNK": 23,
}

SUPPORTED_LATENT_EXT = (".pt", ".pth", ".pkl", ".pickle", ".npy", ".npz", ".json")
PERIODIC_TABLE = GetPeriodicTable()


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


def normalize_key(raw: str) -> str:
    return "".join(ch for ch in raw.upper() if ch.isalnum())


def safe_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


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


def _infer_element_from_atom_name(atom_name: str) -> str:
    token = "".join(ch for ch in atom_name.strip() if ch.isalpha())
    if not token:
        return ""
    token = token.upper()

    # Try two-letter symbol first (e.g. CL, BR, MG), then one-letter.
    if len(token) >= 2:
        two = token[:2].capitalize()
        if _is_valid_element(two):
            return two
    one = token[0].upper()
    if _is_valid_element(one):
        return one
    return ""


def _repair_pdb_element_columns(pdb_path: str) -> str:
    repaired_lines = []
    changed = 0

    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith(("ATOM  ", "HETATM")):
                padded = line.rstrip("\n")
                if len(padded) < 80:
                    padded = padded.ljust(80)

                atom_name = padded[12:16]
                elem_col = padded[76:78].strip()

                elem = elem_col.capitalize() if elem_col else ""
                if not _is_valid_element(elem):
                    elem = _infer_element_from_atom_name(atom_name)

                if elem:
                    new_line = f"{padded[:76]}{elem.rjust(2)}{padded[78:]}"
                    if new_line != padded:
                        changed += 1
                    repaired_lines.append(new_line + "\n")
                    continue

            repaired_lines.append(line)

    if changed == 0:
        return pdb_path

    fd, tmp_path = tempfile.mkstemp(prefix="rdkit_repaired_", suffix=".pdb")
    os.close(fd)
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.writelines(repaired_lines)
    return tmp_path


def _needs_pdb_repair(pdb_path: str) -> bool:
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue

            padded = line.rstrip("\n")
            if len(padded) < 80:
                return True

            elem_col = padded[76:78].strip()
            elem = elem_col.capitalize() if elem_col else ""
            if not _is_valid_element(elem):
                return True
    return False


def read_structure_and_mol(pdb_path: str):
    parse_path = _repair_pdb_element_columns(pdb_path) if _needs_pdb_repair(pdb_path) else pdb_path
    rdkit_mol = Chem.MolFromPDBFile(parse_path, removeHs=True, sanitize=False)
    if rdkit_mol is None and parse_path == pdb_path:
        repaired = _repair_pdb_element_columns(pdb_path)
        parse_path = repaired
        rdkit_mol = Chem.MolFromPDBFile(parse_path, removeHs=True, sanitize=False)
    if rdkit_mol is None:
        raise ValueError(f"RDKit failed to parse {pdb_path}")
    try:
        Chem.SanitizeMol(rdkit_mol)
        Chem.AssignStereochemistry(rdkit_mol)
    except Exception as exc:
        raise ValueError(f"RDKit sanitize/stereo failed: {exc}") from exc

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("molecule", parse_path)

    if parse_path != pdb_path and os.path.exists(parse_path):
        try:
            os.remove(parse_path)
        except OSError:
            pass

    return rdkit_mol, structure


def loop_over_atoms(rdkit_mol, structure):
    atom_features = {}
    bond_features = {}

    rdkit_atom_list = [atom for atom in rdkit_mol.GetAtoms()]
    rdkit_bond_list = [bond for bond in rdkit_mol.GetBonds()]

    pdb_atom_list = []
    res_id_list = []
    residue_meta = []

    for chain in structure[0]:
        for residue in chain:
            resname = residue.get_resname().strip()
            chain_id = chain.id
            resseq = int(residue.id[1])
            res_id = f"{resname}_{resseq}_{chain_id}"
            for atom in residue:
                if atom.element == "H":
                    continue
                res_id_list.append(res_id)
                residue_meta.append((resname, chain_id, resseq))
                pdb_atom_list.append(atom)

    if len(rdkit_atom_list) != len(pdb_atom_list):
        raise ValueError(
            f"Atom count mismatch RDKit={len(rdkit_atom_list)} PDB={len(pdb_atom_list)}"
        )

    start_res_idx = 0
    start_res = res_id_list[0]

    for bond in rdkit_bond_list:
        bond_features[(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())] = {
            "Type": str(bond.GetBondType()),
            "Is Aromatic": bond.GetIsAromatic(),
        }

    for idx, atom in enumerate(rdkit_atom_list):
        pdb_res_id = res_id_list[idx]
        pdb_atom = pdb_atom_list[idx]
        resname, chain_id, resseq = residue_meta[idx]

        if idx != atom.GetIdx():
            raise ValueError(f"Atom {idx} has different index")
        pdb_z = _atomic_number(str(pdb_atom.element).strip())
        rdkit_z = _atomic_number(atom.GetSymbol())
        if pdb_z != rdkit_z:
            raise ValueError(
                f"Atom {idx} symbol mismatch PDB={pdb_atom.element} RDKit={atom.GetSymbol()}"
            )

        if pdb_res_id != start_res:
            start_res = pdb_res_id
            start_res_idx += 1

        atom_features[idx] = {
            "idx": idx,
            "coords": pdb_atom.get_coord(),
            "Atoms": atom.GetSymbol(),
            "Charges": atom.GetFormalCharge(),
            "Hybridization": str(atom.GetHybridization()),
            "Degree": atom.GetDegree(),
            "Rings": atom.IsInRing(),
            "Aromatic": atom.GetIsAromatic(),
            "Residue": pdb_res_id,
            "Res_name": resname,
            "Chain": chain_id,
            "Resseq": resseq,
            "Residue ID": start_res_idx,
        }

    return atom_features, bond_features


def build_residue_records(atom_features: Dict[int, Dict]) -> List[ResidueRecord]:
    grouped: Dict[int, List[Dict]] = {}
    for _, atom in atom_features.items():
        grouped.setdefault(atom["Residue ID"], []).append(atom)

    residues = []
    aa_count = 0
    for res_id in sorted(grouped.keys()):
        atoms = grouped[res_id]
        first = atoms[0]
        res_name = first["Res_name"]
        chain_id = first["Chain"]
        resseq = int(first["Resseq"])
        is_amino = is_aa(res_name, standard=False)
        coords = np.array([a["coords"] for a in atoms], dtype=float)
        identity = f"{res_name}_{resseq}_{chain_id}_{res_id}"
        aa_order_index = aa_count if is_amino else None
        if is_amino:
            aa_count += 1
        residues.append(
            ResidueRecord(
                res_id=res_id,
                res_name=res_name,
                chain_id=chain_id,
                resseq=resseq,
                identity=identity,
                aa_order_index=aa_order_index,
                is_amino_acid=is_amino,
                atom_indices=[a["idx"] for a in atoms],
                mass_center=np.mean(coords, axis=0),
            )
        )
    return residues


def get_neighbors(
    all_residues: List[ResidueRecord],
    center_residue: ResidueRecord,
    atom_features: Dict[int, Dict],
    bond_features: Dict[Tuple[int, int], Dict],
    radius: float,
):
    neighbors = []
    center = center_residue.mass_center
    for residue in all_residues:
        if residue.res_id == center_residue.res_id:
            continue
        delta = np.abs(residue.mass_center - center)
        if delta[0] > radius or delta[1] > radius or delta[2] > radius:
            continue
        if np.linalg.norm(residue.mass_center - center) <= radius:
            neighbors.append(residue)

    pocket_atoms = []
    for residue in neighbors:
        for idx in residue.atom_indices:
            pocket_atoms.append(atom_features[idx].copy())

    atom_idx_set = {atom["idx"] for atom in pocket_atoms}
    pocket_bonds = []
    for bond_item in bond_features.items():
        if bond_item[0][0] in atom_idx_set and bond_item[0][1] in atom_idx_set:
            pocket_bonds.append(bond_item)

    for atom in pocket_atoms:
        atom["coords"] = [float(i) for i in atom["coords"]]

    return pocket_atoms, pocket_bonds


def get_ligand_atoms_and_bonds(
    center_residue: ResidueRecord,
    atom_features: Dict[int, Dict],
    bond_features: Dict[Tuple[int, int], Dict],
):
    ligand_atoms = [atom_features[idx].copy() for idx in center_residue.atom_indices]
    ligand_idx_set = set(center_residue.atom_indices)

    # Keep peptide or covalent attachments crossing the target residue boundary.
    ligand_bonds = []
    for bond_item in bond_features.items():
        if bond_item[0][0] in ligand_idx_set or bond_item[0][1] in ligand_idx_set:
            ligand_bonds.append(bond_item)

    for atom in ligand_atoms:
        atom["coords"] = [float(i) for i in atom["coords"]]

    return ligand_atoms, ligand_bonds


def has_supported_atom_features(atoms: Iterable[Dict]) -> bool:
    for atom in atoms:
        if atom["Atoms"] not in ATOM_ENCODER:
            return False
        if atom["Hybridization"] not in HYBRIDIZATION_ENCODER:
            return False
    return True


def dictionary_to_data(
    ligand_atoms: List[Dict],
    ligand_bonds: List[Tuple[Tuple[int, int], Dict]],
    pocket_atoms: List[Dict],
    pocket_bonds: List[Tuple[Tuple[int, int], Dict]],
    compound_id: str,
    edge_radius: float,
    protein_mpnn_latent_128: np.ndarray,
    protein_mpnn_latent_20: np.ndarray,
):
    if len(ligand_atoms) < 4:
        return None

    if not has_supported_atom_features(ligand_atoms) or not has_supported_atom_features(
        pocket_atoms
    ):
        return None

    num_atoms = len(ligand_atoms) + len(pocket_atoms)
    all_ids = [atom["idx"] for atom in ligand_atoms] + [atom["idx"] for atom in pocket_atoms]
    all_ids.sort()
    id_mapping = {atom_id: i for i, atom_id in enumerate(all_ids)}

    atom_types = torch.zeros(num_atoms, dtype=torch.long)
    all_charges = torch.zeros(num_atoms, dtype=torch.float)
    is_aromatic = torch.zeros(num_atoms, dtype=torch.float)
    is_in_ring = torch.zeros(num_atoms, dtype=torch.float)
    hybridization = torch.zeros(num_atoms, dtype=torch.long)
    degree = torch.zeros(num_atoms, dtype=torch.long)
    position = torch.zeros(num_atoms, 3, dtype=torch.float)
    is_ligand = torch.zeros(num_atoms, dtype=torch.float)
    is_backbone = torch.zeros(num_atoms, dtype=torch.float)

    backbone_count = 0
    for atom in ligand_atoms:
        new_id = id_mapping[atom["idx"]]
        atom_types[new_id] = ATOM_ENCODER[atom["Atoms"]]
        all_charges[new_id] = float(atom["Charges"])
        is_aromatic[new_id] = float(atom["Aromatic"])
        is_in_ring[new_id] = float(atom["Rings"])
        hybridization[new_id] = HYBRIDIZATION_ENCODER[atom["Hybridization"]]
        degree[new_id] = int(atom["Degree"])
        position[new_id] = torch.tensor(atom["coords"], dtype=torch.float)
        is_ligand[new_id] = 1.0
        if backbone_count in (0, 1, 2, 3):
            is_backbone[new_id] = 1.0
        backbone_count += 1

    ligand_idx = [id_mapping[atom["idx"]] for atom in ligand_atoms]
    edge_index = []
    edge_ligand = []

    for i in range(len(ligand_idx)):
        for j in range(i + 1, len(ligand_idx)):
            edge_index.append([ligand_idx[i], ligand_idx[j]])
            edge_index.append([ligand_idx[j], ligand_idx[i]])
            edge_ligand.append(1)
            edge_ligand.append(1)

    for atom in pocket_atoms:
        new_id = id_mapping[atom["idx"]]
        atom_types[new_id] = ATOM_ENCODER[atom["Atoms"]]
        all_charges[new_id] = float(atom["Charges"])
        is_aromatic[new_id] = float(atom["Aromatic"])
        is_in_ring[new_id] = float(atom["Rings"])
        hybridization[new_id] = HYBRIDIZATION_ENCODER[atom["Hybridization"]]
        degree[new_id] = int(atom["Degree"])
        position[new_id] = torch.tensor(atom["coords"], dtype=torch.float)

    protein_atom_dict = {atom["idx"]: atom for atom in pocket_atoms}
    protein_keys = list(protein_atom_dict.keys())
    for i in range(len(protein_keys)):
        for j in range(i + 1, len(protein_keys)):
            atom_i = protein_atom_dict[protein_keys[i]]
            atom_j = protein_atom_dict[protein_keys[j]]
            dist = np.linalg.norm(np.array(atom_i["coords"]) - np.array(atom_j["coords"]))
            if dist < edge_radius:
                edge_index.append([id_mapping[atom_i["idx"]], id_mapping[atom_j["idx"]]])
                edge_index.append([id_mapping[atom_j["idx"]], id_mapping[atom_i["idx"]]])
                edge_ligand.append(0)
                edge_ligand.append(0)

    for i in range(len(ligand_idx)):
        for pocket_id in protein_keys:
            edge_index.append([ligand_idx[i], id_mapping[pocket_id]])
            edge_index.append([id_mapping[pocket_id], ligand_idx[i]])
            edge_ligand.append(0)
            edge_ligand.append(0)

    if not edge_index:
        return None

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_type = torch.zeros(edge_index.size(1), dtype=torch.long)

    ligand_bonds_dict = {}
    for atom_bond in ligand_bonds:
        idx_i = id_mapping.get(atom_bond[0][0])
        idx_j = id_mapping.get(atom_bond[0][1])
        if idx_i is None or idx_j is None:
            continue
        ligand_bonds_dict[(idx_i, idx_j)] = atom_bond[1]["Type"]

    pocket_bonds_dict = {}
    for atom_bond in pocket_bonds:
        idx_i = id_mapping.get(atom_bond[0][0])
        idx_j = id_mapping.get(atom_bond[0][1])
        if idx_i is None or idx_j is None:
            continue
        pocket_bonds_dict[(idx_i, idx_j)] = atom_bond[1]["Type"]

    for i in range(edge_index.size(1)):
        current = (int(edge_index[0, i]), int(edge_index[1, i]))
        reverse = (current[1], current[0])
        if current in ligand_bonds_dict or reverse in ligand_bonds_dict:
            bt = ligand_bonds_dict.get(current, ligand_bonds_dict.get(reverse))
            if bt in BOND_ENCODER:
                edge_type[i] = BOND_ENCODER[bt]
        elif current in pocket_bonds_dict or reverse in pocket_bonds_dict:
            bt = pocket_bonds_dict.get(current, pocket_bonds_dict.get(reverse))
            if bt in BOND_ENCODER:
                edge_type[i] = BOND_ENCODER[bt]

    latent_128_tensor = torch.tensor(protein_mpnn_latent_128, dtype=torch.float)
    latent_20_tensor = torch.tensor(protein_mpnn_latent_20, dtype=torch.float)
    if latent_128_tensor.ndim != 1 or latent_20_tensor.ndim != 1:
        return None

    compound_graph = Data(
        x=atom_types,
        pos=position,
        edge_index=edge_index,
        edge_attr=edge_type,
        edge_ligand=torch.tensor(edge_ligand, dtype=torch.float),
        charges=all_charges,
        degree=degree,
        is_aromatic=is_aromatic,
        is_in_ring=is_in_ring,
        hybridization=hybridization,
        is_ligand=is_ligand,
        is_backbone=is_backbone,
        id=torch.tensor(all_ids, dtype=torch.long),
        ids=torch.arange(len(all_ids), dtype=torch.long),
        compound_id=compound_id,
        # Keep both latent sizes in one graph to avoid duplicating LMDB content.
        protein_mpnn_latent_128=latent_128_tensor,
        protein_mpnn_latent_20=latent_20_tensor,
        protein_mpnn_latent_node_128=latent_128_tensor.unsqueeze(0).repeat(num_atoms, 1),
        protein_mpnn_latent_node_20=latent_20_tensor.unsqueeze(0).repeat(num_atoms, 1),
        # Backward-compatible aliases used by existing consumers.
        # protein_mpnn_latent=latent_128_tensor,
        # protein_mpnn_latent_node=latent_128_tensor.unsqueeze(0).repeat(num_atoms, 1),
    )
    return compound_graph


class LatentStore:
    def __init__(self, root: str, latent_dim: int):
        self.root = root
        self.latent_dim = latent_dim
        self.path_index = self._index_paths()
        self.cache = {}

    def _index_paths(self) -> Dict[str, str]:
        index = {}
        for dirpath, _, filenames in os.walk(self.root):
            for name in filenames:
                if not name.endswith(SUPPORTED_LATENT_EXT):
                    continue
                stem = os.path.splitext(name)[0]
                index[stem] = os.path.join(dirpath, name)
        return index

    def _candidate_stems(self, pdb_name: str) -> List[str]:
        stem = os.path.splitext(pdb_name)[0]
        stems = [pdb_name, stem]
        if stem.endswith("_tidy"):
            stems.append(stem[: -len("_tidy")])
        if pdb_name.endswith("_tidy.pdb"):
            stems.append(pdb_name[: -len("_tidy.pdb")])
        return list(dict.fromkeys(stems))

    def _load_file(self, path: str):
        if path.endswith((".pt", ".pth")):
            return torch.load(path, map_location="cpu")
        if path.endswith((".pkl", ".pickle")):
            with open(path, "rb") as handle:
                return pickle.load(handle)
        if path.endswith(".npy"):
            return np.load(path, allow_pickle=True)
        if path.endswith(".npz"):
            return np.load(path, allow_pickle=True)
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        raise ValueError(f"Unsupported latent file: {path}")

    def _to_residue_map(self, payload):
        if isinstance(payload, np.lib.npyio.NpzFile):
            if "latents" in payload and "keys" in payload:
                lat = payload["latents"]
                keys = payload["keys"]
                return {str(k): lat[i] for i, k in enumerate(keys)}
            first_key = list(payload.keys())[0]
            return payload[first_key]

        if isinstance(payload, dict):
            if "latents" in payload and "keys" in payload:
                lat = payload["latents"]
                keys = payload["keys"]
                return {str(k): lat[i] for i, k in enumerate(keys)}
            if "embeddings" in payload and "residue_ids" in payload:
                lat = payload["embeddings"]
                keys = payload["residue_ids"]
                return {str(k): lat[i] for i, k in enumerate(keys)}
            return payload

        return payload

    def _lookup_in_map(self, residue_map: Dict, residue: ResidueRecord) -> Optional[np.ndarray]:
        normalized_map = {normalize_key(str(k)): v for k, v in residue_map.items()}
        variants = [
            residue.identity,
            f"{residue.res_name}_{residue.resseq}_{residue.chain_id}",
            f"{residue.chain_id}_{residue.resseq}",
            f"{residue.chain_id}:{residue.resseq}",
            f"{residue.resseq}_{residue.chain_id}",
            str(residue.resseq),
        ]
        for key in variants:
            value = normalized_map.get(normalize_key(key))
            if value is None:
                continue
            arr = safe_numpy(value).astype(np.float32).reshape(-1)
            if arr.shape[0] == self.latent_dim:
                return arr
        return None

    def get_latent(self, pdb_name: str, residue: ResidueRecord) -> Optional[np.ndarray]:
        for stem in self._candidate_stems(pdb_name):
            path = self.path_index.get(stem)
            if not path:
                continue

            if path not in self.cache:
                self.cache[path] = self._to_residue_map(self._load_file(path))
            payload = self.cache[path]

            if isinstance(payload, dict):
                latent = self._lookup_in_map(payload, residue)
                if latent is not None:
                    return latent
            else:
                arr = safe_numpy(payload)
                if arr.ndim == 2 and residue.aa_order_index is not None:
                    if residue.aa_order_index < arr.shape[0] and arr.shape[1] == self.latent_dim:
                        return arr[residue.aa_order_index].astype(np.float32)

        return None


def build_graphs_from_pdb(
    pdb_path: str,
    pocket_radius: float,
    edge_radius: float,
    latent_store_128: LatentStore,
    latent_store_20: LatentStore,
):
    pdb_name = os.path.basename(pdb_path)
    rdkit_mol, structure = read_structure_and_mol(pdb_path)
    atom_features, bond_features = loop_over_atoms(rdkit_mol, structure)
    residues = build_residue_records(atom_features)

    graphs = []
    counters = {
        "total_aa": 0,
        "missing_latent_128": 0,
        "missing_latent_20": 0,
        "unsupported_atoms": 0,
        "too_small": 0,
        "ok": 0,
    }

    for residue in residues:
        if not residue.is_amino_acid:
            continue

        counters["total_aa"] += 1

        latent_128 = latent_store_128.get_latent(pdb_name, residue)
        latent_20 = latent_store_20.get_latent(pdb_name, residue)
        if latent_128 is None:
            counters["missing_latent_128"] += 1
            continue
        if latent_20 is None:
            counters["missing_latent_20"] += 1
            continue

        ligand_atoms, ligand_bonds = get_ligand_atoms_and_bonds(
            residue, atom_features, bond_features
        )
        pocket_atoms, pocket_bonds = get_neighbors(
            residues, residue, atom_features, bond_features, radius=pocket_radius
        )

        compound_id = f"{os.path.splitext(pdb_name)[0]}_{residue.identity}"
        graph = dictionary_to_data(
            ligand_atoms=ligand_atoms,
            ligand_bonds=ligand_bonds,
            pocket_atoms=pocket_atoms,
            pocket_bonds=pocket_bonds,
            compound_id=compound_id,
            edge_radius=edge_radius,
            protein_mpnn_latent_128=latent_128,
            protein_mpnn_latent_20=latent_20,
        )
        if graph is None:
            # Reasons are intentionally coarse so this remains fast.
            if len(ligand_atoms) < 4:
                counters["too_small"] += 1
            else:
                counters["unsupported_atoms"] += 1
            continue

        graph.source_name = pdb_name
        graph.center_residue = f"{residue.res_name}_{residue.resseq}_{residue.chain_id}"
        graph.residue_name = residue.res_name
        counters["ok"] += 1
        graphs.append(graph)

    return graphs, counters


def write_lmdb(graphs: List[Data], lmdb_path: str, metadata_path: str):
    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)
    env = lmdb.open(
        lmdb_path,
        map_size=1 << 40,
        subdir=False,
        readonly=False,
        meminit=False,
        map_async=True,
    )

    metadata = {}
    txn = env.begin(write=True)
    for i, sample in enumerate(graphs):
        key = f"{i:08}".encode("ascii")
        metadata[key] = getattr(sample, "source_name", "unknown")
        txn.put(key, pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL))
        if (i + 1) % 10000 == 0:
            txn.commit()
            txn = env.begin(write=True)

    txn.put(b"__len__", pickle.dumps(len(graphs)))
    txn.commit()
    env.sync()
    env.close()

    with open(metadata_path, "wb") as handle:
        pickle.dump(metadata, handle)


def run_pipeline(
    pdb_dir: str,
    output_dir: str,
    output_prefix: str,
    pocket_radius: float,
    edge_radius: float,
    latent_root_128: str,
    latent_root_20: str,
    num_shards: int = 1,
    shard_index: int = 0,
):
    latent_store_128 = LatentStore(latent_root_128, 128)
    latent_store_20 = LatentStore(latent_root_20, 20)

    pdb_files = sorted(
        name
        for name in os.listdir(pdb_dir)
        if name.lower().endswith(".pdb") and os.path.isfile(os.path.join(pdb_dir, name))
    )

    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"shard_index must be in [0, {num_shards - 1}], got {shard_index}"
        )

    # Deterministic modulo split so array tasks can run independently.
    if num_shards > 1:
        pdb_files = [name for i, name in enumerate(pdb_files) if i % num_shards == shard_index]

    all_graphs = []
    summary = {
        "pdb_total": len(pdb_files),
        "pdb_ok": 0,
        "total_aa": 0,
        "missing_latent_128": 0,
        "missing_latent_20": 0,
        "unsupported_atoms": 0,
        "too_small": 0,
        "ok": 0,
    }

    for pdb_name in tqdm(pdb_files, desc="Building (lat20+lat128)"):
        pdb_path = os.path.join(pdb_dir, pdb_name)
        try:
            graphs, counters = build_graphs_from_pdb(
                pdb_path=pdb_path,
                pocket_radius=pocket_radius,
                edge_radius=edge_radius,
                latent_store_128=latent_store_128,
                latent_store_20=latent_store_20,
            )
        except Exception as exc:
            print(f"[WARN] Failed {pdb_name}: {exc}")
            continue

        if graphs:
            summary["pdb_ok"] += 1
            all_graphs.extend(graphs)

        for key in (
            "total_aa",
            "missing_latent_128",
            "missing_latent_20",
            "unsupported_atoms",
            "too_small",
            "ok",
        ):
            summary[key] += counters[key]

    lmdb_path = os.path.join(output_dir, f"{output_prefix}.lmdb")
    metadata_path = os.path.join(output_dir, f"{output_prefix}.metadata.pkl")
    write_lmdb(all_graphs, lmdb_path, metadata_path)

    print(f"\nDone {output_prefix}")
    print(f"  LMDB: {lmdb_path}")
    print(f"  Metadata: {metadata_path}")
    print(f"  Graphs: {len(all_graphs)}")
    print(f"  Processed PDBs: {summary['pdb_ok']}/{summary['pdb_total']}")
    print(f"  Candidate AA positions: {summary['total_aa']}")
    print(f"  Missing latent_128: {summary['missing_latent_128']}")
    print(f"  Missing latent_20: {summary['missing_latent_20']}")
    print(f"  Unsupported atom/hybridization: {summary['unsupported_atoms']}")
    print(f"  Ligand too small: {summary['too_small']}")
    if num_shards > 1:
        print(f"  Shard: {shard_index}/{num_shards}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="One-click .pdb -> residue-pocket graphs -> LMDB with ProteinMPNN latents"
    )
    parser.add_argument("--pdb_dir", type=str, required=True, help="Folder with .pdb files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output folder for LMDB files")
    parser.add_argument("--output_prefix", type=str, default="uaag_eqgat", help="Output prefix")
    parser.add_argument("--pocket_radius", type=float, default=10.0, help="Pocket residue radius in angstrom")
    parser.add_argument("--edge_radius", type=float, default=8.0, help="Pocket atom edge radius in angstrom")
    parser.add_argument(
        "--latent_root_128",
        type=str,
        default="/scratch/project_465002574/PDB/PDB_128",
        help="ProteinMPNN latent root (dim=128)",
    )
    parser.add_argument(
        "--latent_root_20",
        type=str,
        default="/scratch/project_465002574/PDB/PDB_20",
        help="ProteinMPNN latent root (dim=20)",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of shards for parallel processing",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="0-based shard index to process",
    )
    args = parser.parse_args()

    run_pipeline(
        pdb_dir=args.pdb_dir,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        pocket_radius=args.pocket_radius,
        edge_radius=args.edge_radius,
        latent_root_128=args.latent_root_128,
        latent_root_20=args.latent_root_20,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
