import json

import sys
import os

sys.path.append('.')
import pickle

import numpy as np
from torch_geometric.data import Batch
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from IPython import embed
from Bio.PDB import PDBParser
from Bio.PDB import Selection
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from tqdm import tqdm
from Bio.PDB import PDBParser
from torch_geometric.utils import dense_to_sparse, sort_edge_index


atom_encoder = {
    "C": 0,
    "N": 1,
    "O": 2,
    "S": 3,
    "P": 4,
    "Cl": 5,
    "F": 6,
    "Br": 7,
}

hybridization_encoder = {
    "SP": 0,
    "SP2": 1,
    "SP3": 2,
    "SP3D": 3,
}

bond_encoder = {
    'SINGLE': 1,
    'DOUBLE': 2,
    'AROMATIC': 3,
    "TRIPLE": 4,
}
degree_set = {
    0, 1, 2, 3, 4
}

def loop_over_mol(mol_file):
    compound_id = os.path.basename(mol_file).split(".")[0]
    x = []
    pos = []
    charges = []
    degree = []
    is_aromatic = []
    is_in_ring = []
    hybridization = []
    
    mol = Chem.MolFromMolFile(mol_file, sanitize=True, removeHs=True)
    
    if mol is None:
        print("Error parsing PDB file.")
        return
    
    rdkit_atom_list = [atom for atom in mol.GetAtoms()]
    
    conformer = mol.GetConformer()
    
    for idx, atom in enumerate(rdkit_atom_list):
        
        if atom.GetSymbol() not in atom_encoder:
            print(f"Atom {atom.GetSymbol()} not in atom encoder")
            return
        
        charge = atom.GetFormalCharge()
        if charge not in [-1, 0, 1]:
            print(f"Charge {charge} not supported")
            return
        
        degree_val = atom.GetDegree()
        if degree_val not in degree_set:
            print(f"Degree {degree_val} not supported")
            return
        
        hybridization_val = str(atom.GetHybridization())
        if hybridization_val not in hybridization_encoder:
            # raise ValueError(f"Hybridization {hybridization_val} not supported")
            print(f"Hybridization {hybridization_val} not supported")
            print(atom)
            return
        
        x.append(atom_encoder[atom.GetSymbol()])
        pos.append(conformer.GetAtomPosition(idx))
        charges.append(charge)
        degree.append(degree_val)
        is_aromatic.append(atom.GetIsAromatic())
        is_in_ring.append(atom.IsInRing())
        hybridization.append(hybridization_encoder[str(hybridization_val)])
    
    edge_index = []
    edge_attr = []
    
    for bond in mol.GetBonds():
        edge_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edge_index.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
        edge_attr.append(bond_encoder[str(bond.GetBondType())])
        edge_attr.append(bond_encoder[str(bond.GetBondType())])
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            if [i, j] not in edge_index \
                and [j, i] not in edge_index:
                edge_index.append([i, j])
                edge_index.append([j, i])
                edge_attr.append(0)
                edge_attr.append(0)
                
    edge_attr = torch.tensor(edge_attr)
    edge_index = torch.tensor(edge_index).t().contiguous()
    # sort edge_index and sort edge_attr along with it
    bond_edge_index, bond_edge_attr = sort_edge_index(edge_index=edge_index, edge_attr=edge_attr, sort_by_row=False)
    
    graph_data = Data(
        x=torch.tensor(x),
        pos=torch.tensor(pos),
        edge_index=bond_edge_index,
        edge_attr=bond_edge_attr,
        charges=torch.tensor(charges),
        degree=torch.tensor(degree),
        is_aromatic=torch.tensor(is_aromatic),
        is_in_ring=torch.tensor(is_in_ring),
        hybridization=torch.tensor(hybridization),
        is_backbone=torch.zeros(len(x)),
        is_ligand=torch.ones(len(x)),
        compound_id=compound_id
    )
    # from IPython import embed; embed()
    return graph_data
def main():
    output_list = []
    path = "/home/qcx679/hantang/UAAG/data/AACLBR"
    naa_dir = os.listdir(path)
    for mol in tqdm(naa_dir):
        try:
            graph_data = loop_over_mol(os.path.join(path, mol))
        except Exception as e:
            print(f"Error in {mol}: {e}")
            continue
        output_list.append(graph_data)
        
    torch.save(output_list, "/home/qcx679/hantang/UAAG2/data/full_graph/naa/AACLBR_data.pt")
if __name__ == '__main__':
    main()