import json

import sys

sys.path.append('.')
import pickle
from tqdm import tqdm
import numpy as np
from torch_geometric.data import Batch
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from IPython import embed
import pickle
atom_encoder = {
    "C": 0,
    "N": 1,
    "O": 2,
    "S": 3,
    "P": 4,
}

hybridization_encoder = {
    "SP": 0,
    "SP2": 1,
    "SP3": 2,
    "SP3D": 3,
}

res_encoder = {}

bond_encoder = {
    'SINGLE': 1,
    'DOUBLE': 2,
    'AROMATIC': 3,
}
bond_aromatic_encoder = {}

def dictionary_to_data(data, compound_id, radius=8):
    ligand = data['ligand']
    pocket = data['pocket']
    ligand_atoms = ligand[0]
    ligand_bonds = ligand[1]
    pocket_atoms = pocket[0]
    pocket_bonds = pocket[1]
    
    num_atoms = len(ligand_atoms) + len(pocket_atoms)
    
    all_ids = [ligand_atoms[i]['idx'] for i in range(len(ligand_atoms))] + [pocket_atoms[i]['idx'] for i in range(len(pocket_atoms))]
    # sort all_ids
    all_ids.sort()
    id_mapping = {id: i for i, id in enumerate(all_ids)}
    reverse_id_mapping = {i: id for i, id in enumerate(all_ids)}
    
    atom_types = torch.zeros(num_atoms)
    all_charges = torch.zeros(num_atoms)
    is_aromatic = torch.zeros(num_atoms)
    is_in_ring = torch.zeros(num_atoms)
    hybridization = torch.zeros(num_atoms)
    degree = torch.zeros(num_atoms)
    position = torch.zeros(num_atoms, 3)
    is_ligand = torch.zeros(num_atoms)
    is_backbone = torch.zeros(num_atoms)
    
    
    backbone_count = 0
    for atom in ligand_atoms:
        new_id = id_mapping[atom['idx']]
        atom_types[new_id] = atom_encoder[atom['Atoms']]
        all_charges[new_id] = atom['Charges']
        is_aromatic[new_id] = atom['Aromatic']
        is_in_ring[new_id] = atom['Rings']
        hybridization[new_id] = hybridization_encoder[atom['Hybridization']]
        degree[new_id] = atom['Degree']
        position[new_id] = torch.tensor(atom['coords'])
        is_ligand[new_id] = 1
        if backbone_count < 4:
            is_backbone[new_id] = 1
            backbone_count += 1
        else:
            is_backbone[new_id] = 0
        
        # construct full connected edge_index for ligand
    ligand_idx = [id_mapping[atom['idx']] for atom in ligand_atoms]
    edge_index = []
    for i in range(len(ligand_idx)):
        for j in range(i+1, len(ligand_idx)):
            edge_index.append([ligand_idx[i], ligand_idx[j]])
            edge_index.append([ligand_idx[j], ligand_idx[i]])
                
    for atom in pocket_atoms:
        new_id = id_mapping[atom['idx']]
        atom_types[new_id] = atom_encoder[atom['Atoms']]
        all_charges[new_id] = atom['Charges']
        is_aromatic[new_id] = atom['Aromatic']
        is_in_ring[new_id] = atom['Rings']
        hybridization[new_id] = hybridization_encoder[atom['Hybridization']]
        degree[new_id] = atom['Degree']
        position[new_id] = torch.tensor(atom['coords'])
        is_ligand[new_id] = 0
        is_backbone[new_id] = 0
    
    protein_atom_dict = {atom['idx']: atom for atom in pocket_atoms}
    # protein_idx = [atom['idx'] for atom in pocket_atoms]
    protein_atom_dict_keys = list(protein_atom_dict.keys())
    for key_i in range(len(protein_atom_dict_keys)):
        for key_j in range(key_i+1, len(protein_atom_dict_keys)):
            atom_i = protein_atom_dict[protein_atom_dict_keys[key_i]]
            atom_j = protein_atom_dict[protein_atom_dict_keys[key_j]]
            dist = np.linalg.norm(np.array(atom_i['coords']) - np.array(atom_j['coords']))
            # embed()
            if dist < radius:
                edge_index.append([id_mapping[atom_i['idx']], id_mapping[atom_j['idx']]])
                edge_index.append([id_mapping[atom_j['idx']], id_mapping[atom_i['idx']]])
    edge_index = torch.tensor(edge_index).t().contiguous()
    edge_type = torch.zeros(edge_index.size(1))
    edge_ligand = torch.zeros(edge_index.size(1))
    
    ligand_index = torch.where(is_ligand == 1)[0]
    
    for bond_idx in range(edge_index.size(1)):
        idx_i = edge_index[0, bond_idx]
        idx_j = edge_index[1, bond_idx]
        
        if idx_i in ligand_index and idx_j in ligand_index:
            edge_ligand[bond_idx] = 1
            
        
    for atom_bond in ligand_bonds:
        idx_i = id_mapping[atom_bond[0][0]]
        idx_j = id_mapping[atom_bond[0][1]]
        bond_type = atom_bond[1]['Type']
        # search for [id_i, idx_j] in edge_index
        
        for i in range(edge_index.size(1)):
            if edge_index[0][i] == idx_i and edge_index[1][i] == idx_j:
                edge_type[i] = bond_encoder[bond_type]
            if edge_index[0][i] == idx_j and edge_index[1][i] == idx_i:
                edge_type[i] = bond_encoder[bond_type]
    
    for atom_bond in pocket_bonds:
        idx_i = id_mapping[atom_bond[0][0]]
        idx_j = id_mapping[atom_bond[0][1]]
        bond_type = atom_bond[1]['Type']
        # search for [id_i, idx_j] in edge_index
        
        for i in range(edge_index.size(1)):
            if edge_index[0][i] == idx_i and edge_index[1][i] == idx_j:
                edge_type[i] = bond_encoder[bond_type]
            if edge_index[0][i] == idx_j and edge_index[1][i] == idx_i:
                edge_type[i] = bond_encoder[bond_type]
    
    compound_graph = Data(
        x=atom_types,
        pos=position,
        edge_index=edge_index,
        edge_attr=edge_type,
        edge_ligand=edge_ligand,
        charges=all_charges,
        degree=degree,
        is_aromatic=is_aromatic,
        is_in_ring=is_in_ring,
        hybridization=hybridization,
        is_ligand=is_ligand,
        is_backbone=is_backbone,
        id=torch.tensor(all_ids),
        ids=torch.tensor(range(len(all_ids))),
        compound_id=compound_id,
        
    )
    return compound_graph
    # save the compound graph
    # torch.save(compound_graph, f"/home/qcx679/hantang/UAAG2/data/full_graph/{compound_id}_test.pt")
    # embed()
    
def split_list(lst, n_splits=50):
    # Calculate the approximate size of each sublist
    avg_size = len(lst) // n_splits
    remainder = len(lst) % n_splits

    sublists = []
    start = 0
    for i in range(n_splits):
        # Add an extra item to some sublists to account for the remainder
        end = start + avg_size + (1 if i < remainder else 0)
        sublists.append(lst[start:end])
        start = end

    return sublists

def json_to_torch_geometric_data(args):
    print(f"Processing {args.json_path}")
    with open(args.json_path, 'r') as f:
        data = json.load(f)
    print("Finished loading json file")
    data_list = []
    compound_id_list = list(data.keys())
    compound_id_list.sort()
    # compound = data[compound_id_list[0]]
    # dictionary_to_data(compound)
    # split compound_id_list to 20 parts
    new_compound_list = split_list(compound_id_list)
    new_compound_list = new_compound_list[args.split_num]
    for compound_id in tqdm(new_compound_list):
        compound = data[compound_id]
        compound_graph = dictionary_to_data(compound, compound_id)
        data_list.append(compound_graph)
        
        
    # save the data_list
    # embed()
    torch.save(data_list, f"/home/qcx679/hantang/UAAG2/data/full_graph/data/{args.output_name}_{args.split_num}.pt")

def main(args):

   # path = "/home/qcx679/hantang/UAAG2/data/processed/uaag_aa_eqgat_8.json"
    json_to_torch_geometric_data(args)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_num", type=int, default=0)
    parser.add_argument("--json_path", type=str, default="/home/qcx679/hantang/UAAG2/data/processed/uaag_aa_eqgat_test.json")
    parser.add_argument("--output_name", type=str, default="full_graph_0")
    args = parser.parse_args()
    main(args)