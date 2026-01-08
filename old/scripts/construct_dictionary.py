import json
import sys
sys.path.append('.')
import pickle
from tqdm import tqdm
import numpy as np
import os
path = '/home/qcx679/hantang/UAAG2/data/processed'
path_ls = ['uaag_aa_eqgat_2.json',
 'uaag_aa_eqgat_6.json',
 'uaag_aa_eqgat_16.json',
 'uaag_aa_eqgat_0.json',
 'uaag_aa_eqgat_1.json',
 'uaag_aa_eqgat_7.json',
 'uaag_aa_eqgat_9.json',
 'uaag_aa_eqgat_19.json',
 'uaag_aa_eqgat_4.json',
 'uaag_aa_eqgat_11.json',
 'uaag_aa_eqgat_17.json',
 'uaag_aa_eqgat_13.json',
 'uaag_aa_eqgat_3.json',
 'uaag_aa_eqgat_18.json',
 'uaag_aa_eqgat_10.json',
 'uaag_aa_eqgat_15.json',
 'uaag_aa_eqgat_8.json',
 'uaag_aa_eqgat_14.json',
 'uaag_aa_eqgat_5.json',
 'uaag_aa_eqgat_12.json']

atom_dic = set()
charge_dic = set()
hybridization_dic = set()
degree_dict = set()
ring_dict = set()
aromatic_dict = set()
res_dict = set()
bond_dict = set()
bond_aromatic_dict = set()
for path_curr in path_ls:
    data_path = os.path.join(path, path_curr)
    with open(data_path, 'r') as f:
        data = json.load(f)
    print(f"Processing {path_curr}")
    for compound_id, compound in tqdm(data.items()):
        compound_ligand = compound['ligand']
        compound_pocket = compound['pocket']
        ligand_atoms = compound_ligand[0]
        for a in ligand_atoms:
            atom_dic.add(a['Atoms'])
            charge_dic.add(str(a['Charges']))
            hybridization_dic.add(a['Hybridization'])
            degree_dict.add(str(a['Degree']))
            ring_dict.add(a['Rings'])
            aromatic_dict.add(a['Aromatic'])
            res_dict.add(a['Res_name'])
        for b in compound_ligand[1]:
            bond_dict.add(b[1]['Type'])
            bond_aromatic_dict.add(b[1]['Is Aromatic'])
            
        pocket_atoms = compound_pocket[0]
        for a in pocket_atoms:
            atom_dic.add(a['Atoms'])
            charge_dic.add(str(a['Charges']))
            hybridization_dic.add(a['Hybridization'])
            degree_dict.add(str(a['Degree']))
            ring_dict.add(a['Rings'])
            aromatic_dict.add(a['Aromatic'])
            res_dict.add(a['Res_name'])
        for b in compound_pocket[1]:
            bond_dict.add(b[1]['Type'])
            bond_aromatic_dict.add(b[1]['Is Aromatic'])
            
atom_dic = list(atom_dic)
charge_dic = list(charge_dic)
hybridization_dic = list(hybridization_dic)
degree_dict = list(degree_dict)
ring_dict = list(ring_dict)
aromatic_dict = list(aromatic_dict)
res_dict = list(res_dict)
bond_dict = list(bond_dict)
bond_aromatic_dict = list(bond_aromatic_dict)
# save them to a json file
with open('atom_dic.json', 'w') as f:
    json.dump(atom_dic, f)
with open('charge_dic.json', 'w') as f:
    json.dump(charge_dic, f)
with open('hybridization_dic.json', 'w') as f:
    json.dump(hybridization_dic, f)
with open('degree_dict.json', 'w') as f:
    json.dump(degree_dict, f)
with open('ring_dict.json', 'w') as f:
    json.dump(ring_dict, f)
with open('aromatic_dict.json', 'w') as f:
    json.dump(aromatic_dict, f)
with open('res_dict.json', 'w') as f:
    json.dump(res_dict, f)
with open('bond_dict.json', 'w') as f:
    json.dump(bond_dict, f)
with open('bond_aromatic_dict.json', 'w') as f:
    json.dump(bond_aromatic_dict, f)
