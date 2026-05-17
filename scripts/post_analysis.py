from random import sample
import pandas as pd
import numpy as np
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
    """Classify the chirality at the amino-acid Cα using 3D geometry.

    Returns one of:
      'L'        — L-amino acid
      'D'        — D-amino acid
      'achiral'  — no Cα stereocenter (e.g. glycine)
      'unknown'  — Cα not found, no 3D conformer, or geometry degenerate

    Implementation: locates Cα as the carbon bonded to (a) an amine N AND
    (b) a carbonyl carbon (C=O), then computes the signed volume of the
    vectors (N − Cα), (C(=O) − Cα), (Cβ − Cα). A positive signed volume
    corresponds to L; negative to D. Calibrated against RDKit-embedded
    L/D conformers of Ala, Ser, Phe (heavy-atom-only). Independent of CIP
    priorities, so Cys/Sec don't need special handling.

    Note: this works on mol files that store only heavy atoms (no H), as is
    the case for UAAG2-generated ligand.mol files. RDKit's CIP-based stereo
    perception fails on such files because Cα has only 3 heavy neighbors.
    """
    if mol is None or mol.GetNumConformers() == 0:
        return 'unknown'
    ca = n_a = co_a = r_a = None
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'C':
            continue
        nbrs = list(atom.GetNeighbors())
        n_nb = next((x for x in nbrs if x.GetSymbol() == 'N'), None)
        co_nb = next((x for x in nbrs
                      if x.GetSymbol() == 'C'
                      and any(b.GetBondType() == Chem.BondType.DOUBLE
                              and b.GetOtherAtom(x).GetSymbol() == 'O'
                              for b in x.GetBonds())),
                     None)
        if n_nb and co_nb:
            others = [x for x in nbrs
                      if x.GetIdx() not in (n_nb.GetIdx(), co_nb.GetIdx())
                      and x.GetSymbol() != 'H']
            ca, n_a, co_a = atom, n_nb, co_nb
            r_a = others[0] if others else None
            break
    if ca is None:
        return 'unknown'
    if r_a is None:
        return 'achiral'
    conf = mol.GetConformer()
    def _pos(idx):
        p = conf.GetAtomPosition(idx)
        return np.array([p.x, p.y, p.z])
    p_ca = _pos(ca.GetIdx())
    v_n  = _pos(n_a.GetIdx())  - p_ca
    v_co = _pos(co_a.GetIdx()) - p_ca
    v_r  = _pos(r_a.GetIdx())  - p_ca
    vol = float(np.dot(v_n, np.cross(v_co, v_r)))
    if abs(vol) < 1e-4:
        return 'unknown'
    return 'L' if vol > 0 else 'D'


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
    