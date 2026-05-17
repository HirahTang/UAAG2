from random import sample
import pandas as pd
from rdkit import Chem
import os
import sys
import argparse
sys.path.append('.')
sys.path.append('..')
from uaag2.utils import aa_check
from collections import Counter
from tqdm import tqdm


def get_aa_chirality(mol):
    """Classify the chirality at the amino-acid Cα.

    Returns one of:
      'L'        — L-amino acid (CIP S at Cα; R for Cys/Sec where S/Se on Cβ
                   inverts the CIP priority ordering)
      'D'        — D-amino acid (CIP R at Cα; S for Cys/Sec)
      'achiral'  — no Cα stereocenter (e.g. glycine)
      'unknown'  — Cα not found or stereochemistry undetermined
    """
    if mol is None:
        return 'unknown'
    try:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    except Exception:
        return 'unknown'
    # Cα = sp3 carbon bonded to an amine N AND to a carbonyl C(=O)
    patt = Chem.MolFromSmarts('[C;X4]([N;!$(N=*)])[C](=O)')
    matches = mol.GetSubstructMatches(patt) if patt is not None else []
    if not matches:
        return 'unknown'
    ca_idx = matches[0][0]
    ca = mol.GetAtomWithIdx(ca_idx)
    if not ca.HasProp('_CIPCode'):
        return 'achiral'
    cip = ca.GetProp('_CIPCode')
    # Cys / Sec convention: S or Se on Cβ inverts the CIP priority order
    # relative to the rest of the canonical amino acids.
    chalcogen_on_beta = False
    for nb in ca.GetNeighbors():
        if nb.GetSymbol() != 'C':
            continue
        # Skip the carbonyl carbon
        if any(b.GetBondType() == Chem.BondType.DOUBLE and
               b.GetOtherAtom(nb).GetSymbol() == 'O'
               for b in nb.GetBonds()):
            continue
        for nb2 in nb.GetNeighbors():
            if nb2.GetIdx() != ca_idx and nb2.GetSymbol() in ('S', 'Se'):
                chalcogen_on_beta = True
                break
        if chalcogen_on_beta:
            break
    if chalcogen_on_beta:
        return 'L' if cip == 'R' else 'D'
    return 'L' if cip == 'S' else 'D'


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
                        '3th', 'YMe', '2Np', 'Bzt', 'UNK', 'INV',
                        'chir_L', 'chir_D', 'chir_achiral', 'chir_unknown'])



for aa in tqdm(aa_list):
    per_aa_table = pd.DataFrame(columns=['iter', 'batch', 'identity', 'SMILES', 'chirality'])
    gen_aa_list = []
    chirality_list = []
    aa_name, aa_pos = aa.split("_")[0], int(aa.split("_")[1])
    aa_path = os.path.join(analysis_path, aa)
    sample_path_list = os.listdir(aa_path)
    for sample_path in sample_path_list:

        # if sample_path is not a directory, skip it
        if not os.path.isdir(os.path.join(aa_path, sample_path)):
            continue

        iter_num = int(sample_path.split("_")[-1])

        sample_path = os.path.join(aa_path, sample_path)


        batch_path_list = os.listdir(sample_path)
        for batch_path in batch_path_list:
            batch_num = int(batch_path.split("_")[-1])
            mol_path = os.path.join(sample_path, batch_path, 'final', 'ligand.mol')
            mol = Chem.MolFromMolFile(mol_path)
            chirality = get_aa_chirality(mol)
            try:
                gen_aa = aa_check(mol)
                gen_aa_list.append(gen_aa)
                chirality_list.append(chirality)
                if gen_aa == "UNK":
                    per_aa_table = pd.concat([per_aa_table, pd.DataFrame([[iter_num, batch_num, "UNK", Chem.MolToSmiles(mol), chirality]], columns=per_aa_table.columns)], ignore_index=True)
                else:
                    per_aa_table = pd.concat([per_aa_table, pd.DataFrame([[iter_num, batch_num, gen_aa, None, chirality]], columns=per_aa_table.columns)], ignore_index=True)
            except:
                per_aa_table = pd.concat([per_aa_table, pd.DataFrame([[iter_num, batch_num, "INV", None, "unknown"]], columns=per_aa_table.columns)], ignore_index=True)
                gen_aa_list.append("INV")
                chirality_list.append("unknown")
                print(f"Error in {mol_path}")

    # save per_aa_table
    save_per_aa_path = os.path.join(aa_path, f"{aa_name}_{aa_pos}_aa_table.csv")
    per_aa_table.to_csv(save_per_aa_path, index=False)
    aa_counter = Counter(gen_aa_list)
    # convert all values in aa_counter to lists with one element
    aa_counter = {k: [v] for k, v in aa_counter.items()}
    chir_counter = Counter(chirality_list)
    # convert the content in aa_couunter to a row in dataframe
    new_row = {'aa': [aa_name], 'pos': [aa_pos], **aa_counter,
               'chir_L':       [chir_counter.get('L', 0)],
               'chir_D':       [chir_counter.get('D', 0)],
               'chir_achiral': [chir_counter.get('achiral', 0)],
               'chir_unknown': [chir_counter.get('unknown', 0)]}
    dataframe = pd.concat([dataframe, pd.DataFrame(new_row)], ignore_index=True)

save_path = os.path.join(analysis_path, "aa_distribution.csv")
dataframe.to_csv(save_path, index=False)
    