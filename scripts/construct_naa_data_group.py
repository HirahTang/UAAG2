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

# def is_backbone_atom(atom):
#     """
#     Checks if an atom belongs to the protein backbone.
#     Backbone atoms are: N, CA, C, and O.
#     """
#     return atom.get_name() in {"N", "CA", "C", "O"}

# test_l_sidechain_path = "/home/qcx679/hantang/UAAG/data/L_sidechain/0A1/0A1.pdb"
# parser = PDBParser()
# structure = parser.get_structure('0A1', test_l_sidechain_path)
# backbone_atoms = []
# for model in structure:
#     for chain in model:
#         for residue in chain:
#             # if residue.get_id()[0] == ' ':  # Ignore heteroatoms (e.g., water, ligands)
#             for atom in residue:
#                 if is_backbone_atom(atom):
#                     backbone_atoms.append(atom)


# Define backbone atom names
BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}

def is_backbone_atom(atom):
    """
    Determines if an atom is part of the backbone based on its PDB info.
    """
    pdb_info = atom.GetPDBResidueInfo()
    if pdb_info:
        atom_name = pdb_info.GetName().strip()  # Atom name (e.g., "CA")
        return atom_name in BACKBONE_ATOMS
    return False

def loop_over_mol(mol2file, pdb_file):
    """
    Parses a PDB file and checks whether each atom is a backbone atom.
    """
    compound_id = os.path.basename(mol2file).split(".")[0]
    is_backbone = []
    x = []
    pos = []
    charges = []
    degree = []
    is_aromatic = []
    is_in_ring = []
    hybridization = []
    mol_pdb = Chem.MolFromPDBFile(pdb_file, sanitize=True, removeHs=True)
    mol = Chem.MolFromMol2File(mol2file, sanitize=True, removeHs=True)
    
    # check the atoms in mol and mol_pdb are the same
    
    
    if mol is None:
        print("Error parsing PDB file.")
        return
    rdkit_atom_list = [atom for atom in mol.GetAtoms()]
    rdkit_pdb_list = [atom for atom in mol_pdb.GetAtoms()]
    if len(rdkit_atom_list) != len(rdkit_pdb_list):
        print("Number of atoms in mol and mol_pdb are different")
        return
    for idx, atom in enumerate(rdkit_atom_list):
        if atom.GetSymbol() != rdkit_pdb_list[idx].GetSymbol():
            print(f"Atom {idx} in mol and mol_pdb are different")
            return
    conformer = mol.GetConformer()
    for idx, atom in enumerate(rdkit_atom_list):

        if is_backbone_atom(rdkit_pdb_list[idx]):
            # pdb_info = atom.GetPDBResidueInfo()
            # res_name = pdb_info.GetResidueName().strip()  # Residue name (e.g., "ALA")
            # chain_id = pdb_info.GetChainId()  # Chain ID (e.g., "A")
            # res_id = pdb_info.GetResidueNumber()  # Residue number (e.g., 12)
            # atom_name = pdb_info.GetName().strip()
            # print(f"Backbone Atom: {atom_name}, Residue: {res_name}, Chain: {chain_id}, Residue ID: {res_id}")
            is_backbone_i = 1
        else:
            is_backbone_i = 0
        
        if atom.GetSymbol() not in atom_encoder:
            print(f"Atom {atom.GetSymbol()} not in atom encoder")
            return
            raise ValueError(f"Atom {atom.GetSymbol()} not in atom encoder")
        
        charge = atom.GetFormalCharge()
        if charge not in [-1, 0, 1]:
            print(f"Charge {charge} not supported")
            return
            # raise ValueError(f"Charge {charge} not supported")

        degree_val = atom.GetDegree()
        if degree_val not in degree_set:
            print(f"Degree {degree_val} not supported")
            return
            # raise ValueError(f"Degree {degree_val} not supported")
        
        hybridization_val = str(atom.GetHybridization())
        if hybridization_val not in hybridization_encoder:
            # raise ValueError(f"Hybridization {hybridization_val} not supported")
            print(f"Hybridization {hybridization_val} not supported")
            print(atom)
            return
        is_backbone.append(is_backbone_i)
        charges.append(charge)
        hybridization.append(hybridization_encoder[hybridization_val])
        degree.append(degree_val)
        is_aromatic.append(int(atom.GetIsAromatic()))
        is_in_ring.append(int(atom.IsInRing()))
        pos_i = conformer.GetAtomPosition(idx)
        pos.append([pos_i.x, pos_i.y, pos_i.z])
        x.append(atom_encoder[atom.GetSymbol()])
        
    edge_index = []
    edge_attr = []
    # Full connected edge index

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
        is_backbone=torch.tensor(is_backbone),
        is_ligand=torch.ones(len(x)),
        componud_id=compound_id
        )
    # embed()
    return graph_data
# Example usage
pdb_file = "/home/qcx679/hantang/UAAG/data/L_sidechain/0A1/0A1.pdb"  # Replace with your PDB file
mol2_file = "/home/qcx679/hantang/UAAG/data/L_sidechain/0A1/0A1.mol2"
# try:
output = loop_over_mol(mol2_file, pdb_file)
# except Exception as e:
#     print(f"Error processing PDB file: {e}")
#     output = None
# embed()
def main():
    save_list = []
    L_sidechain_path = "/home/qcx679/hantang/UAAG/data/L_sidechain"
    L_sidechain_dir = os.listdir(L_sidechain_path)
    for aa in tqdm(L_sidechain_dir):
        pdb_file = os.path.join(L_sidechain_path, aa, f"{aa}.pdb")
        mol2_file = os.path.join(L_sidechain_path, aa, f"{aa}.mol2")
        try:
            output = loop_over_mol(mol2_file, pdb_file)
        except Exception as e:
            print(f"Error processing PDB file: {e}")
            output = None
        if output is not None:
            save_list.append(output)

    torch.save(save_list, "/home/qcx679/hantang/UAAG2/data/full_graph/naa/L_sidechain_data.pt")
            
if __name__ == "__main__":
    main()