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
import os
pdb_root_path = "/home/qcx679/hantang/UAAG/data/pdb_processed/"
pdb_list = os.listdir(pdb_root_path)
pdb_root_path = '/home/qcx679/hantang/UAAG/data/DMS/pdb_tidy/'
pdb_list = ['DN7A_SACS2_tidy.pdb']
# read the pdb file using rdkit

def loop_over_atoms(rdkit_mol, pdb_mol):
        
    atom_features = {}
    bond_features = {}
    
    rdkit_atom_list = [atom for atom in rdkit_mol.GetAtoms()]
    rdkit_bond_list = [bond for bond in rdkit_mol.GetBonds()]
    pdb_atom_list = []
    res_id_list = []
    for chain in pdb_mol[0]:
        for residue in chain:
            res_id = f"{residue.get_resname()}_{residue.id[1]}_{chain.id}"  # e.g., LIG_1_A
            for atom in residue:
                res_id_list.append(res_id)
                pdb_atom_list.append(atom)
    
    start_res_idx = 0
    start_res = res_id_list[0]
    
    for bond in tqdm(rdkit_bond_list):
        bond_features[(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())] = {
            "Type": str(bond.GetBondType()),
            "Is Aromatic": bond.GetIsAromatic(),
        }
    
    for idx, atom in tqdm(enumerate(rdkit_atom_list)):
        
        pdb_res_id = res_id_list[idx]
        pdb_atom = pdb_atom_list[idx]
        if idx != atom.GetIdx():
            raise ValueError(f"Atom {idx} has different IdX")
 
            
        if pdb_atom.element != atom.GetSymbol():
            raise ValueError(f"Atom {idx} has different element")
        
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
            "Res_name": pdb_atom.get_parent().get_resname(),
            "Residue ID": start_res_idx,
        }
    return atom_features, bond_features

def create_pickle(pdb_title, pdb_root_path):
    pdb_path = os.path.join(pdb_root_path, pdb_title)
    pdb_title = pdb_title.split(".")[0]
    example_mol = Chem.MolFromPDBFile(pdb_path, removeHs=True)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("molecule", pdb_path)

    atom_to_residue = {}
    pdb_atoms = []

    for chain in structure[0]:
        for residue in chain:
            res_id = f"{residue.get_resname()}_{residue.id[1]}_{chain.id}"  # e.g., LIG_1_A
            for atom in residue:
                atom_to_residue[atom.serial_number] = res_id
                pdb_atoms.append(atom)
    try:
        Chem.SanitizeMol(example_mol)
        Chem.AssignStereochemistry(example_mol)
    except Exception as e:
        print(f"Failed to sanitize {pdb_title}")
        print(e)
        return
    try:
        output = loop_over_atoms(example_mol, structure)
        atom_features, bond_features = output
        # embed()
        if not os.path.exists(f"data/uaag_data_v2/pdb/{pdb_title}"):
            os.makedirs(f"data/uaag_data_v2/pdb/{pdb_title}")
        
        with open(f"data/uaag_data_v2/pdb/{pdb_title}/{pdb_title}_atom.pkl", "wb") as f:
            pickle.dump(atom_features, f)
        with open(f"data/uaag_data_v2/pdb/{pdb_title}/{pdb_title}_bond.pkl", "wb") as f:
            pickle.dump(bond_features, f)    
            
    except Exception as e:
        print(f"Failed to process {pdb_title}")
        print(e)
 
if __name__ == "__main__":
    for pdb_title in tqdm(pdb_list):
        try:
            create_pickle(pdb_title, pdb_root_path)
        except Exception as e:
            print(f"Failed to process {pdb_title}")
            print(e)
            continue