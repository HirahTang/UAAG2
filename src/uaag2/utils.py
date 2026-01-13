import json
import math
import os

import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
from openbabel import openbabel as ob
import torch
import torch.nn.functional as F
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.utils import sort_edge_index
from torch_scatter import scatter_mean
from tqdm import tqdm

from uaag2 import bond_analyze

obConversion = ob.OBConversion()
obConversion.SetInAndOutFormats("xyz", "mol")

with open("data/aa_graph.json", "rb") as json_file:
    AA_GRAPH_DICT = json.load(json_file)
    json_file.close()

BOND_ORDER_TO_EDGE_TYPE = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    4: Chem.BondType.AROMATIC,
}


def load_data(hparams, data_path: list, pdb_list: list) -> Data:
    data = []
    pdb_list_readout = []
    for file in tqdm(data_path):
        print(f"Loading {file} \n")
        data_file = torch.load(file)
        data.extend(data_file)

    for file in tqdm(pdb_list):
        print(f"Loading {file} \n")
        pdb_file = torch.load(file)
        pdb_list_readout.extend(pdb_file)

    # randomly split data into train, val, test
    np.random.seed(hparams.seed)
    np.random.shuffle(data)
    np.random.shuffle(pdb_list)
    num_data = len(data)
    num_train = math.floor(num_data * hparams.train_size)
    num_test = hparams.test_size

    train_data = data[:num_train]
    val_data = data[num_train:]
    test_data = pdb_list_readout[:num_test]

    # test_data = data[:num_test]
    # val_data = data[num_test:num_test + num_val]
    # train_data = data[num_test + num_val:]

    return train_data, val_data, test_data


def create_model(hparams, num_atom_features, num_bond_classes):
    from e3moldiffusion.coordsatomsbonds import DenoisingEdgeNetwork

    model = DenoisingEdgeNetwork(
        hn_dim=(hparams["sdim"], hparams["vdim"]),
        num_layers=hparams["num_layers"],
        latent_dim=None,
        use_cross_product=hparams["use_cross_product"],
        num_atom_features=num_atom_features,
        num_bond_types=num_bond_classes,
        edge_dim=hparams["edim"],
        cutoff_local=hparams["cutoff_local"],
        vector_aggr=hparams["vector_aggr"],
        fully_connected=hparams["fully_connected"],
        local_global_model=hparams["local_global_model"],
        recompute_edge_attributes=True,
        recompute_radius_graph=False,
        edge_mp=hparams["edge_mp"],
        context_mapping=hparams["context_mapping"],
        num_context_features=hparams["num_context_features"],
        bond_prediction=hparams["bond_prediction"],
        property_prediction=hparams["property_prediction"],
        coords_param=hparams["continuous_param"],
        use_pos_norm=hparams["use_pos_norm"],
    )
    return model


def load_model(filepath, num_atom_features, num_bond_classes, device="cpu", **kwargs):
    import re

    ckpt = torch.load(filepath, map_location="cpu")
    args = ckpt["hyper_parameters"]

    args["use_pos_norm"] = True

    model = create_model(args, num_atom_features, num_bond_classes)

    state_dict = ckpt["state_dict"]
    state_dict = {re.sub(r"^model\.", "", k): v for k, v in ckpt["state_dict"].items() if k.startswith("model")}
    state_dict = {k: v for k, v in state_dict.items() if not any(x in k for x in ["prior", "sde", "cat"])}
    model.load_state_dict(state_dict)
    return model.to(device)


def zero_mean(x, batch, dim_size: int, dim=0):
    out = x - scatter_mean(x, index=batch, dim=dim, dim_size=dim_size)[batch]
    return out


def initialize_edge_attrs_reverse(edge_index_global, n, bonds_prior, num_bond_classes, device):
    # edge types for FC graph
    j, i = edge_index_global
    mask = j < i
    mask_i = i[mask]
    mask_j = j[mask]
    nE = len(mask_i)
    edge_attr_triu = torch.multinomial(bonds_prior, num_samples=nE, replacement=True)

    j = torch.concat([mask_j, mask_i])
    i = torch.concat([mask_i, mask_j])
    edge_index_global = torch.stack([j, i], dim=0)
    edge_attr_global = torch.concat([edge_attr_triu, edge_attr_triu], dim=0)

    edge_index_global, edge_attr_global = sort_edge_index(
        edge_index=edge_index_global, edge_attr=edge_attr_global, sort_by_row=False
    )
    j, i = edge_index_global
    mask = j < i
    mask_i = i[mask]
    mask_j = j[mask]

    # some assert
    # from IPython import embed; embed()

    # edge_attr_global_dense = torch.zeros(size=(n, n), device=device, dtype=torch.long)
    # edge_attr_global_dense[
    #     edge_index_global[0], edge_index_global[1]
    # ] = edge_attr_global
    # from IPython import embed; embed()
    # assert (edge_attr_global_dense - edge_attr_global_dense.T).sum().float() == 0.0

    edge_attr_global = F.one_hot(edge_attr_global, num_bond_classes).float()

    return edge_attr_global, edge_index_global, mask, mask_i


def write_xyz_file(coords, atom_types, filename):
    out = f"{len(coords)}\n\n"
    assert len(coords) == len(atom_types)
    for i in range(len(coords)):
        out += f"{atom_types[i]} {coords[i, 0]:.3f} {coords[i, 1]:.3f} {coords[i, 2]:.3f}\n"
    with open(filename, "w") as f:
        f.write(out)


def write_xyz_file_from_batch(
    pos,
    atom_type,
    batch,
    atom_decoder=None,
    path=".",
    i=0,
):
    if not os.path.exists(path):
        os.makedirs(path)

    atomsxmol = batch.bincount()
    num_atoms_prev = 0
    for k, num_atoms in enumerate(atomsxmol):
        save_dir = os.path.join(path, f"batch_{k}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        ats = torch.argmax(atom_type[num_atoms_prev : num_atoms_prev + num_atoms], dim=1)
        types = [atom_decoder[int(a)] for a in ats]
        positions = pos[num_atoms_prev : num_atoms_prev + num_atoms]

        write_xyz_file(positions, types, os.path.join(save_dir, f"mol_{i}.xyz"))

        num_atoms_prev += num_atoms


def is_connected_molecule(atom_list, edge_list):
    G = nx.Graph()
    G.add_nodes_from(atom_list)
    for i, j in edge_list:
        G.add_edge(i, j)
    return nx.is_connected(G)


def check_sanitize_connectivity(mol_file):
    # read the molecule and convert it to a list of atoms and bonds
    # from IPython import embed; embed()
    mol = Chem.MolFromMolFile(mol_file)
    try:
        atoms = [atom.GetIdx() for atom in mol.GetAtoms()]
    except (AttributeError, TypeError):
        return False, False
    bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]
    connected_mol = is_connected_molecule(atoms, bonds)
    try:
        Chem.SanitizeMol(mol)
        sanitized = True
    except Exception:
        sanitized = False

    return connected_mol, sanitized


def visualize_mol_bond_abandoned(pos, atoms, bonds, edge_index, edge_decoder, val_check=False):
    # from IPython import embed; embed()
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]

    edge_dict = {}

    for i in range(edge_index.size(1)):
        start, end = edge_index[:, i].tolist()
        attr = bonds[i].item()
        edge_key = tuple(sorted((start, end)))

        if edge_key in edge_dict:
            if edge_dict[edge_key] != attr:
                print(f"Warning: Edge {edge_key} has conflicting attributes {edge_dict[edge_key]} and {attr}")
        else:
            edge_dict[edge_key] = attr

    edge_list = []
    for (start, end), attr in edge_dict.items():
        if attr in edge_decoder:
            edge_list.append(([start, end], edge_decoder[attr]))

    if val_check:
        connected_mol = is_connected_molecule(list(range(len(x))), [i[0] for i in edge_list])

    molB = Chem.RWMol()

    for atom in atoms:
        molB.AddAtom(Chem.Atom(atom))

    for bond in edge_list:
        molB.AddBond(bond[0][0], bond[0][1], bond[1])
    conf = Chem.Conformer()

    for idx, (x_pos, y_pos, z_pos) in enumerate(list(zip(x, y, z))):
        conf.SetAtomPosition(idx, (float(x_pos), float(y_pos), float(z_pos)))

    molB.AddConformer(conf)

    final_mol = molB.GetMol()

    try:
        Chem.SanitizeMol(final_mol)
        sanitized = True
    except Exception:
        sanitized = False

    mol_block = Chem.MolToMolBlock(final_mol)

    if val_check:
        return mol_block, (connected_mol, sanitized)
    else:
        return mol_block


def visualize_mol(atom_sets, val_check=False):
    pos, atoms = atom_sets
    print("Visualizing molecule")
    new_atoms = []
    new_pos = []
    for atom_idx in range(len(atoms)):
        if atoms[atom_idx] != "NOATOM":
            new_atoms.append(atoms[atom_idx])
            new_pos.append(pos[atom_idx])
    # from IPython import embed; embed()
    pos = torch.stack(new_pos)
    atoms = new_atoms

    x = pos[:, 0].tolist()
    y = pos[:, 1].tolist()
    z = pos[:, 2].tolist()

    edge_list = []

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = atoms[i], atoms[j]

            draw_edge_int = bond_analyze.get_bond_order(atom1, atom2, dist)
            if draw_edge_int:
                edge_list.append(([i, j], draw_edge_int))
    if val_check:
        connected_mol = is_connected_molecule(list(range(len(x))), [i[0] for i in edge_list])

    molB = Chem.RWMol()

    for atom in atoms:
        molB.AddAtom(Chem.Atom(atom))

    for bond in edge_list:
        molB.AddBond(bond[0][0], bond[0][1], BOND_ORDER_TO_EDGE_TYPE[bond[1]])
    conf = Chem.Conformer()
    for idx, (x_pos, y_pos, z_pos) in enumerate(list(zip(x, y, z))):
        conf.SetAtomPosition(idx, (float(x_pos), float(y_pos), float(z_pos)))

    molB.AddConformer(conf)

    final_mol = molB.GetMol()

    # Santize the molecule
    try:
        Chem.SanitizeMol(final_mol)
        sanitized = True
    except Exception:
        sanitized = False

    mol_block = Chem.MolToMolBlock(final_mol)

    if val_check:
        return mol_block, (connected_mol, sanitized)
    else:
        return mol_block


def convert_edge_to_bond(
    batch,
    out_dict,
    path,
    reconstruct_mask,
    atom_decoder,
    edge_decoder,
):
    """Convert edge_attr_global to bond type"""
    # batch: the input batch
    # edge_attr_global_ligand: Tensor (num_of_ligand_edges, num_bond_classes=5)

    connected_list = []
    sanitized_list = []

    ligand_pos = out_dict["coords_pred"]
    ligand_atom_type = out_dict["atoms_pred"].argmax(dim=-1)

    batch_pos = batch.pos
    batch_pos[reconstruct_mask == 1] = ligand_pos

    batch_atom_type = batch.x
    batch_atom_type[reconstruct_mask == 1] = ligand_atom_type.float()

    ligand_edge_batches = batch.batch[batch.edge_index[0]][batch.edge_ligand == 1]
    ligand_edge_index = batch.edge_index[:, batch.edge_ligand == 1]
    batch_atom_idx = torch.tensor(range(len(batch.x)), device=batch.x.device)
    batch_atom_idx_ligand = batch_atom_idx[batch.is_ligand == 1]

    backbone_size = batch.batch[batch.is_backbone == 1].bincount()
    ligand_size = batch.batch[batch.is_ligand == 1].bincount()
    edge_size = ligand_edge_batches.bincount()

    start_idx_ligand = 0
    start_idx_edge = 0

    for i in range(len(backbone_size)):
        path_batch = os.path.join(path, f"batch_{i}", "final")
        if not os.path.exists(path_batch):
            os.makedirs(path_batch)

        end_idx_ligand_atom = start_idx_ligand + ligand_size[i]
        end_idx_ligand_edge = start_idx_edge + edge_size[i]

        ligand_atom_idx = batch_atom_idx_ligand[start_idx_ligand:end_idx_ligand_atom]
        ligand_bond_idx = ligand_edge_index[:, start_idx_edge:end_idx_ligand_edge]

        ligand_pos_batch = batch_pos[ligand_atom_idx]
        ligand_atom_batch = batch_atom_type[ligand_atom_idx]

        # skip the ones with value=8 in ligand_atom_batch, and the corresponding ligand_pos_batch
        ligand_pos_batch = ligand_pos_batch[ligand_atom_batch != 8]
        ligand_atom_batch = ligand_atom_batch[ligand_atom_batch != 8]

        # from IPython import embed; embed()
        idx_map = {idx.detach().cpu().item(): i for i, idx in enumerate(ligand_atom_idx)}

        ligand_bond_idx = torch.stack(
            [
                torch.tensor([idx_map[idx.detach().cpu().item()] for idx in ligand_bond_idx[0]]),
                torch.tensor([idx_map[idx.detach().cpu().item()] for idx in ligand_bond_idx[1]]),
            ],
            dim=0,
        )

        ligand_path = os.path.join(path_batch, "ligand.xyz")

        write_xyz_file(ligand_pos_batch.cpu().detach(), [atom_decoder[int(a)] for a in ligand_atom_batch], ligand_path)
        ob_mol = ob.OBMol()
        obConversion.ReadFile(ob_mol, ligand_path)
        obConversion.WriteFile(ob_mol, os.path.join(path_batch, "ligand.mol"))
        connected, sanitized = check_sanitize_connectivity(os.path.join(path_batch, "ligand.mol"))
        # mol_block, (connected, sanitized) = visualize_mol_bond(
        #     ligand_pos_batch,
        #     [atom_decoder[int(a)] for a in ligand_atom_batch],
        #     ligand_bond_attr,
        #     ligand_bond_idx,
        #     edge_decoder,
        #     val_check=True
        #     )
        connected_list.append(connected)
        sanitized_list.append(sanitized)

        # from IPython import embed; embed()

        start_idx_ligand = end_idx_ligand_atom
        start_idx_edge = end_idx_ligand_edge

    return connected_list, sanitized_list


def get_molecules(
    out_dict,
    path,
    batch,
    reconstruct_mask,
    backbone_mask,
    pocket_mask,
    atom_decoder,
):
    connected_list = []
    sanitized_list = []
    backbone_size = batch[backbone_mask == 1].bincount()
    ligand_size = batch[reconstruct_mask == 1].bincount()
    pocket_size = batch[pocket_mask == 0].bincount()

    true_pos = out_dict["coords_true"]
    true_atom_type = out_dict["atoms_true"]

    ligand_pos = out_dict["coords_pred"]
    ligand_atom_type = out_dict["atoms_pred"]

    backbone_pos = out_dict["coords_backbone"]
    backbone_atom_type = out_dict["atoms_backbone"]

    pocket_pos = out_dict["coords_pocket"]
    pocket_atom_type = out_dict["atoms_pocket"]

    start_idx_ligand = 0
    start_idx_backbone = 0
    start_idx_pocket = 0
    start_idx_true = 0
    for i in range(len(ligand_size)):
        end_idx_ligand = start_idx_ligand + ligand_size[i]
        ligand_pos_i = ligand_pos[start_idx_ligand:end_idx_ligand]
        ligand_atom_type_i = ligand_atom_type[start_idx_ligand:end_idx_ligand]
        ligand_atom_type_i = [atom_decoder[int(a)] for a in ligand_atom_type_i.argmax(dim=1)]

        end_idx_backbone = start_idx_backbone + backbone_size[i]
        backbone_pos_i = backbone_pos[start_idx_backbone:end_idx_backbone]
        backbone_atom_type_i = backbone_atom_type[start_idx_backbone:end_idx_backbone]
        backbone_atom_type_i = [atom_decoder[int(a)] for a in backbone_atom_type_i.argmax(dim=1)]

        end_idx_pocket = start_idx_pocket + pocket_size[i]
        pocket_pos_i = pocket_pos[start_idx_pocket:end_idx_pocket]
        pocket_atom_type_i = pocket_atom_type[start_idx_pocket:end_idx_pocket]
        # from IPython import embed; embed()
        pocket_atom_type_i = [atom_decoder[int(a)] for a in pocket_atom_type_i.argmax(dim=1)]

        end_idx_true = start_idx_true + ligand_size[i]
        true_pos_i = true_pos[start_idx_true:end_idx_true]
        true_atom_type_i = true_atom_type[start_idx_true:end_idx_true]
        true_atom_type_i = [atom_decoder[int(a)] for a in true_atom_type_i]

        # from IPython import embed; embed()
        ligand_pos_all = torch.cat([ligand_pos_i, backbone_pos_i], dim=0)
        ligand_atom_type_all = ligand_atom_type_i + backbone_atom_type_i
        atom_set = (ligand_pos_all.cpu().detach(), ligand_atom_type_all)
        mol_block, (connected, sanitized) = visualize_mol(atom_set, val_check=True)

        connected_list.append(connected)
        sanitized_list.append(sanitized)

        true_set = (true_pos_i.cpu().detach(), true_atom_type_i)
        true_mol_block = visualize_mol(true_set, val_check=False)

        path_batch = os.path.join(path, f"batch_{i}", "final")
        if not os.path.exists(path_batch):
            os.makedirs(path_batch)
        # with open(os.path.join(path_batch, "ligand.mol"), "w") as f:
        #     f.write(mol_block)

        with open(os.path.join(path_batch, "true.mol"), "w") as f:
            f.write(true_mol_block)

        start_idx_true = end_idx_true
        start_idx_ligand = end_idx_ligand
        start_idx_backbone = end_idx_backbone
        start_idx_pocket = end_idx_pocket
        try:
            pocket_pos_i = pocket_pos_i.cpu().detach()
            pocket_set = (pocket_pos_i, pocket_atom_type_i)
            pocket_mol_block = visualize_mol(pocket_set, val_check=False)
            with open(os.path.join(path_batch, "pocket.mol"), "w") as f:
                f.write(pocket_mol_block)
        except Exception:
            print("Pocket is empty")
            pass
    return connected_list, sanitized_list


def mol_to_graph(mol):  # Convert mol to nx.graph for isomorphism checking
    #    mol = Chem.MolFromMolFile(mol)
    MolGraph = nx.Graph()
    for atom in mol.GetAtoms():
        MolGraph.add_node(atom.GetIdx(), symbol=atom.GetSymbol())
    for bond in mol.GetBonds():
        MolGraph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    return MolGraph


def aa_check(gen_mol):
    nm = iso.categorical_node_match("symbol", "C")
    gen_aa_graph = mol_to_graph(gen_mol)
    if not nx.is_connected(gen_aa_graph):
        return "INV"
    for aa in AA_GRAPH_DICT:
        aa_graph = nx.node_link_graph(AA_GRAPH_DICT[aa])
        if nx.is_isomorphic(gen_aa_graph, aa_graph, node_match=nm):
            return aa
    return "UNK"


class Statistics:
    def __init__(
        self,
        num_nodes,
        atom_types,
        bond_types,
        charge_types,
        valencies,
        bond_lengths,
        bond_angles,
        dihedrals=None,
        is_in_ring=None,
        is_aromatic=None,
        hybridization=None,
        degree=None,
        force_norms=None,
    ):
        self.num_nodes = num_nodes
        # print("NUM NODES IN STATISTICS", num_nodes)
        self.atom_types = atom_types
        self.bond_types = bond_types
        self.charge_types = charge_types
        self.valencies = valencies
        self.bond_lengths = bond_lengths
        self.bond_angles = bond_angles
        self.dihedrals = dihedrals
        self.is_in_ring = is_in_ring
        self.is_aromatic = is_aromatic
        self.hybridization = hybridization
        self.degree = degree
        self.force_norms = force_norms
