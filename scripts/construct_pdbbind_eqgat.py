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


def main():
    pdb_path = "/home/qcx679/hantang/UAAG2/data/pdbbind"
    pdb_dir = os.listdir(pdb_path)
    atom_dict = {}
    charge_dict = {}
    hybridization_dict = {}
    degree_dict = {}
    bond_dict = {}
    for pdb in tqdm(pdb_dir):
        atom_ligand_file = os.path.join(pdb_path, pdb, f'{pdb}_ligand_atom.pkl')
        bond_ligand_file = os.path.join(pdb_path, pdb, f'{pdb}_ligand_bond.pkl')
        atom_pocket_file = os.path.join(pdb_path, pdb, f'{pdb}_pocket_atom.pkl')
        bond_pocket_file = os.path.join(pdb_path, pdb, f'{pdb}_pocket_bond.pkl')
        
        with open(atom_ligand_file, 'rb') as f:
            ligand_atoms = pickle.load(f)
        with open(bond_ligand_file, 'rb') as f:
            ligand_bonds = pickle.load(f)
        with open(atom_pocket_file, 'rb') as f:
            pocket_atoms = pickle.load(f)
        with open(bond_pocket_file, 'rb') as f:
            pocket_bonds = pickle.load(f)
        
        for atom_idx in ligand_atoms:
            # embed()
            # atom_dict.add(ligand_atoms[atom_idx]['Atoms'])
            # charge_dict.add(str(ligand_atoms[atom_idx]['Charges']))
            # hybridization_dict.add(ligand_atoms[atom_idx]['Hybridization'])
            # degree_dict.add(str(ligand_atoms[atom_idx]['Degree']))
            if ligand_atoms[atom_idx]['Atoms'] not in atom_dict:
                atom_dict[ligand_atoms[atom_idx]['Atoms']] = 1
            else:
                atom_dict[ligand_atoms[atom_idx]['Atoms']] += 1
            if str(ligand_atoms[atom_idx]['Charges']) not in charge_dict:
                charge_dict[str(ligand_atoms[atom_idx]['Charges'])] = 1
            else:
                charge_dict[str(ligand_atoms[atom_idx]['Charges'])] += 1
            if ligand_atoms[atom_idx]['Hybridization'] not in hybridization_dict:
                hybridization_dict[ligand_atoms[atom_idx]['Hybridization']] = 1
            else:
                hybridization_dict[ligand_atoms[atom_idx]['Hybridization']] += 1
            if str(ligand_atoms[atom_idx]['Degree']) not in degree_dict:
                degree_dict[str(ligand_atoms[atom_idx]['Degree'])] = 1
            else:
                degree_dict[str(ligand_atoms[atom_idx]['Degree'])] += 1
                
        
        for atom_idx in pocket_atoms:
            if pocket_atoms[atom_idx]['Atoms'] not in atom_dict:
                atom_dict[pocket_atoms[atom_idx]['Atoms']] = 1
            else:
                atom_dict[pocket_atoms[atom_idx]['Atoms']] += 1
            if str(pocket_atoms[atom_idx]['Charges']) not in charge_dict:
                charge_dict[str(pocket_atoms[atom_idx]['Charges'])] = 1
            else:
                charge_dict[str(pocket_atoms[atom_idx]['Charges'])] += 1
            if pocket_atoms[atom_idx]['Hybridization'] not in hybridization_dict:
                hybridization_dict[pocket_atoms[atom_idx]['Hybridization']] = 1
            else:
                hybridization_dict[pocket_atoms[atom_idx]['Hybridization']] += 1
            if str(pocket_atoms[atom_idx]['Degree']) not in degree_dict:
                degree_dict[str(pocket_atoms[atom_idx]['Degree'])] = 1
            else:
                degree_dict[str(pocket_atoms[atom_idx]['Degree'])] += 1
        
        for bond_index, content in ligand_bonds.items():
            if content['Type'] not in bond_dict:
                bond_dict[content['Type']] = 1
            else:
                bond_dict[content['Type']] += 1
        
        for bond_index, content in pocket_bonds.items():
            if content['Type'] not in bond_dict:
                bond_dict[content['Type']] = 1
            else:
                bond_dict[content['Type']] += 1
            
    atom_dict = list(atom_dict.items())
    charge_dict = list(charge_dict.items())
    hybridization_dict = list(hybridization_dict.items())
    degree_dict = list(degree_dict.items())
    bond_dict = list(bond_dict.items())
    # output them into json
    with open('atom_dict_pdbbind.json', 'w') as f:
        json.dump(atom_dict, f)
    with open('charge_dict_pdbbind.json', 'w') as f:
        json.dump(charge_dict, f)
    with open('hybridization_dict_pdbbind.json', 'w') as f:
        json.dump(hybridization_dict, f)
    with open('degree_dict_pdbbind.json', 'w') as f:
        json.dump(degree_dict, f)
    with open('bond_dict_pdbbind.json', 'w') as f:
        json.dump(bond_dict, f)
        
if __name__ == '__main__':
    main()