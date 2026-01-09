import sys
sys.path.append('.')
import json
import pickle
from Bio.PDB import PDBParser
from Bio.PDB import Selection
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from tqdm import tqdm
from IPython import embed
from configs.datasets_config import aa_dict
import os
import pickle
import numpy as np

import argparse
import random
radius = 10



class amino_acid_pocket:
    def __init__(self, atoms, bonds, radius=10):
        self.atoms = atoms
        self.bonds = bonds
        self.radius = radius
        res_ids = list(set([item[1]['Residue ID'] for item in atoms.items()]))
        res_ids.sort()
        residues = []
        for res_id in res_ids:
            residue = {}
            residue['res_id'] = res_id
            residue['res_name'] = [item[1]['Res_name'] for item in atoms.items() if item[1]['Residue ID'] == res_id][0]
            residue['position'] = np.array([item[1]['coords'] for item in atoms.items() if item[1]['Residue ID'] == res_id])
            residue['mass_center'] = np.mean(residue['position'], axis=0)
            residue['identity'] = [item[1]['Residue'] for item in atoms.items() if item[1]['Residue ID'] == res_id][0] + '_' + str(res_id)
            residues.append(residue)
        self.residues = residues

        
    def get_neighbors(self, residue):
        neighbors = []
        for res in self.residues:
            if res['res_id'] == residue['res_id']:
                continue
            
            if np.abs(res['mass_center'][0] - residue['mass_center'][0]) > self.radius \
                or np.abs(res['mass_center'][1] - residue['mass_center'][1]) > self.radius \
                    or np.abs(res['mass_center'][2] - residue['mass_center'][2]) > self.radius:
                continue
            if np.linalg.norm(res['mass_center'] - residue['mass_center']) <= self.radius:
                neighbors.append(res)
                
        # get the atoms from neighbors
        atomwise_neighbors = []
        for neighbor in neighbors:
            atomwise_neighbors.append([item[1] for item in self.atoms.items() if item[1]['Residue ID'] == neighbor['res_id']])
        # covert the atomwise_neighbors to a 1d list
        atomwise_neighbors = [item for sublist in atomwise_neighbors for item in sublist]
        bond_neighbors = []
        atom_idx = set([item['idx'] for item in atomwise_neighbors])
        for bond in self.bonds.items():
            if bond[0][0] in atom_idx and bond[0][1] in atom_idx:
                bond_neighbors.append(bond)
        for item in atomwise_neighbors:
            item['coords'] = list([float(i) for i in item['coords']])
        return atomwise_neighbors, bond_neighbors
    
    def get_atoms(self, residue):
        return_atoms = [item[1] for item in self.atoms.items() if item[1]['Residue ID'] == residue['res_id']]
        atom_idx = set([item['idx'] for item in return_atoms])
        return_bonds = []
        for bond in self.bonds.items():
            if bond[0][0] in atom_idx or bond[0][1] in atom_idx:
                return_bonds.append(bond)
                
        for item in return_atoms:
            item['coords'] = list([float(i) for i in item['coords']])
            
        return return_atoms, return_bonds
    
def main(args):
    file_name = '1A1I_tidy_atom.pkl'
    file_name2 = '1A1I_tidy_bond.pkl'
    path = "/home/qcx679/hantang/UAAG/data/uaag_data_v2/pdb/1A1I_tidy"
    pdb_path = "/home/qcx679/hantang/UAAG/data/uaag_data_v2/pdb"
    pdb_path = args.pdb_dir
    # pdb_path = "/home/qcx679/hantang/UAAG2/data/intermediate_pickles/5ly1"
    pdb_dir = os.listdir(pdb_path)
    
    save_dict = {}
    # # split pdb_dir into 5 parts
    random.seed(42)
    pdb_dir_copy = pdb_dir.copy()
    random.shuffle(pdb_dir_copy)
    if args.split:
        pdb_dir_copy = np.array_split(pdb_dir_copy, 2)
        pdb_dir_current = list(pdb_dir_copy[args.split_num])
    else:
        pdb_dir_current = pdb_dir_copy
    # embed()
    for pdb_name in pdb_dir_current:
        save_dict = {}
        # pdb_path = "/home/qcx679/hantang/UAAG2/data/uaag_data_v2/pdb/DN7A_SACS2_tidy/"
        # pdb_name = "DN7A_SACS2_tidy"
        atom_file = os.path.join(pdb_path, f"{pdb_name}", f"{pdb_name}_atom.pkl")
        bond_file = os.path.join(pdb_path, f"{pdb_name}", f"{pdb_name}_bond.pkl")

        with open(atom_file, 'rb') as f:
            atom_features = pickle.load(f)
        with open(bond_file, 'rb') as f:
            bond_features = pickle.load(f)

        protein_object = amino_acid_pocket(atom_features, bond_features)
        for aa in tqdm(protein_object.residues):
            if aa['res_name'] in aa_dict.keys() and aa['res_name'] != 'GLY':
                output_dict = {}
                output_dict['ligand'] = protein_object.get_atoms(aa)
                output_dict['pocket'] = protein_object.get_neighbors(aa)
                save_dict[f'{pdb_name}_'+aa['identity']] = output_dict
        # embed()
        with open(f'data/benchmark/uaag_{pdb_name}.json', 'w') as json_file:
            json.dump(save_dict, json_file)
            json_file.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create pocket data')
    parser.add_argument('--pdb_dir', type=str, default='/home/qcx679/hantang/UAAG2/data/intermediate_pickles', help='Directory containing protein pdb files')
    parser.add_argument('--split', action='store_true', default=False, help='Split the PDBs for parallel processing')
    parser.add_argument('--split_num', type=int, default=0, help='Split number')
    args = parser.parse_args()
    main(args)