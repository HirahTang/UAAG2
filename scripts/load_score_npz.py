import numpy as np
import pandas as pd
from tqdm import tqdm
score_path = "/home/qcx679/hantang/ProteinMPNN/outputs/DN7A_SACS2/score_only/DN7A_SACS2_fasta_985.npz"
score_pdb_path = "/home/qcx679/hantang/ProteinMPNN/outputs/DN7A_SACS2/score_only/DN7A_SACS2_pdb.npz"
score_pdb = np.load(score_pdb_path)
baselines = "/home/qcx679/hantang/UAAG/data/DMS/full_benchmark/DN7A_SACS2_Tsuboyama_2023_1JIC.csv"

baselines_df = pd.read_csv(baselines)
baselines_df['ProteinMPNN_global_score'] = np.nan
baselines_df['ProteinMPNN_local_score'] = np.nan
for n in tqdm(range(1, 1009)):
    score_path = f"/home/qcx679/hantang/ProteinMPNN/outputs/DN7A_SACS2/score_only/DN7A_SACS2_fasta_{n}.npz"
    score = np.load(score_path)
    
    local_score = score_pdb['score'].mean() - score['score'].mean()
    global_score = score_pdb['global_score'].mean() - score['global_score'].mean()
    baselines_df.loc[n-1, 'ProteinMPNN_global_score'] = global_score
    baselines_df.loc[n-1, 'ProteinMPNN_local_score'] = local_score


from IPython import embed; embed()