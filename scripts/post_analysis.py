from random import sample
import pandas as pd
from rdkit import Chem
import os
import sys
import argparse
sys.path.append('.')
sys.path.append('..')
from uaag.utils import aa_check
from collections import Counter
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--analysis_path', type=str, default='/home/qcx679/hantang/UAAG2/ProteinGymSampling/runoverfitting_check_scale/Testing_set')
args = parser.parse_args()

analysis_path = args.analysis_path
aa_list = os.listdir(analysis_path)

dataframe = pd.DataFrame(columns=['aa', 'pos', 'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', \
    'MET', 'ASN', 'PRO', 'GLN', 'ARG', \
        'SER', 'THR', 'VAL', 'TRP', 'TYR', \
            'Abu', 'Nva', 'Nle', 'Ahp', 'Aoc', \
                'Tme', 'hSM', 'tBu', 'Cpa', 'Aib', \
                    'MeG', 'MeA', 'MeB', 'MeF', '2th', \
                        '3th', 'YMe', '2Np', 'Bzt', 'UNK', 'INV'])

for aa in tqdm(aa_list):
    gen_aa_list = []
    aa_name, aa_pos = aa.split("_")[0], int(aa.split("_")[1])
    aa_path = os.path.join(analysis_path, aa)
    sample_path_list = os.listdir(aa_path)
    for sample_path in sample_path_list:
        
        # if sample_path is not a directory, skip it
        if not os.path.isdir(os.path.join(aa_path, sample_path)):
            continue
        
        sample_path = os.path.join(aa_path, sample_path)
        
        
        batch_path_list = os.listdir(sample_path)
        for batch_path in batch_path_list:
            mol_path = os.path.join(sample_path, batch_path, 'final', 'ligand.mol')
            mol = Chem.MolFromMolFile(mol_path)
            try:
                gen_aa = aa_check(mol)
                gen_aa_list.append(gen_aa)
            except:
                gen_aa_list.append("INV")
                print(f"Error in {mol_path}")
                
    aa_counter = Counter(gen_aa_list)
    # convert all values in aa_counter to lists with one element
    aa_counter = {k: [v] for k, v in aa_counter.items()}
    # convert the content in aa_couunter to a row in dataframe
    new_row = {'aa': [aa_name], 'pos': [aa_pos], **aa_counter}
    dataframe = pd.concat([dataframe, pd.DataFrame(new_row)], ignore_index=True)

save_path = os.path.join(analysis_path, "aa_distribution.csv")
dataframe.to_csv(save_path, index=False)
    