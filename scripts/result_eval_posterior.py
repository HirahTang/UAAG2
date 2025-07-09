import pandas as pd
import os
from scipy.stats import spearmanr
import numpy as np
import argparse





aa_map = {
        'ALA': 'A',
        'CYS': 'C',
        'ASP': 'D',
        'GLU': 'E',
        'PHE': 'F',
        'GLY': 'G',
        'HIS': 'H',
        'ILE': 'I',
        'LYS': 'K',
        'LEU': 'L',
        'MET': 'M',
        'ASN': 'N',
        'PRO': 'P',
        'GLN': 'Q',
        'ARG': 'R',
        'SER': 'S',
        'THR': 'T',
        'VAL': 'V',
        'TRP': 'W',
        'TYR': 'Y',
        'UNK': 'X',
        'INV': 'Z'
    } 

def minmax(x):
    return ( (x - np.min(x)) / (np.max(x) - np.min(x)) ) 


def calc_ndcg(y_true, y_score, **kwargs):
    '''
    Inputs:
        y_true: an array of the true scores where higher score is better
        y_score: an array of the predicted scores where higher score is better
    Options:
        quantile: If True, uses the top k quantile of the distribution
        top: under the quantile setting this is the top quantile to
            keep in the gains calc. This is a PERCENTAGE (i.e input 10 for top 10%)
    Notes:
        Currently we're calculating NDCG on the continuous value of the DMS
        I tried it on the binary value as well and the metrics seemed mostly
        the same.
    '''
    if 'quantile' not in kwargs:
        kwargs['quantile'] = True
    if 'top' not in kwargs:
        kwargs['top'] = 10
    if kwargs['quantile']:
        k = np.floor(y_true.shape[0]*(kwargs['top']/100)).astype(int)
    else:
        k = kwargs['top']
    # k = 1
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_score, pd.Series):
        y_score = y_score.values
    gains = minmax(y_true)
    ranks = np.argsort(np.argsort(-y_score)) + 1
    
    if k == 'all':
        k = len(ranks)
    #sub to top k
    ranks_k = ranks[ranks <= k]
    gains_k = gains[ranks <= k]
    #all terms with a gain of 0 go to 0
    ranks_fil = ranks_k[gains_k != 0]
    gains_fil = gains_k[gains_k != 0]
    
    #if none of the ranks made it return 0
    if len(ranks_fil) == 0:
        return (0)
    
    #discounted cumulative gains
    dcg = np.sum([g/np.log2(r+1) for r,g in zip(ranks_fil, gains_fil)])
    
    #ideal dcg - calculated based on the top k actual gains
    ideal_ranks = np.argsort(np.argsort(-gains)) + 1
    ideal_ranks_k = ideal_ranks[ideal_ranks <= k]
    ideal_gains_k = gains[ideal_ranks <= k]
    ideal_ranks_fil = ideal_ranks_k[ideal_gains_k != 0]
    ideal_gains_fil = ideal_gains_k[ideal_gains_k != 0]
    idcg = np.sum([g/np.log2(r+1) for r,g in zip(ideal_ranks_fil, ideal_gains_fil)])
    
    #normalize
    ndcg = dcg/idcg
    
    return (ndcg)

def divide_mutant(mutant):
    return mutant[0], int(mutant[1:-1]), mutant[-1]

    

def main(args):
    ground_truth = "/home/qcx679/hantang/UAAG/data/DMS/DN7A_SACS2_Tsuboyama_2023_1JIC.csv"
    generated = args.generated
    baselines = "/home/qcx679/hantang/UAAG2/data/DN7A_SACS2_baselines.csv"
    softmax_temp = args.softmax_temp
    calc_mut = ['A', 'C', 'S', 'T', 'V', 'D', 'N', 'I', 'L', 'M', 'E', "Q", 'K', 'H', 'F', 'R', 'Y', 'W']
    df_ground_truth = pd.read_csv(ground_truth)
    df_generated = pd.read_csv(generated)
    df_baselines = pd.read_csv(baselines)
    df_baselines['wt'], df_baselines['pos'], df_baselines['mut'] = zip(*df_baselines['mutant'].map(divide_mutant))
    # for aa_pos in df_baselines['pos'].unique():
    #     # add new rows of df_baselines['mut'] = df_baselines['wt'] with ProteinMPNN_global_score = 0
    #     for aa_name in ['CYS', 'SER', 'THR', 'VAL', 'ASP', 'ASN', 'ILE', 'LEU', 'MET', 'GLU', 'GLN', 'LYS', 'PHE', 'ARG']:
            
            
    # from IPython import embed; embed()
    # Spearman correlation
    df_baselines['proteinmpnn_weight'] = np.nan
    df_baselines = df_baselines.set_index('mutant')
    baseline_list = ['Site_Independent', 'EVmutation', 'DeepSequence_single',
       'DeepSequence_ensemble', 'EVE_single', 'EVE_ensemble', 'Unirep',
       'Unirep_evotune', 'MSA_Transformer_single', 'MSA_Transformer_ensemble',
       'ESM1b', 'ESM1v_single', 'ESM1v_ensemble', 'ESM2_8M', 'ESM2_35M',
       'ESM2_150M', 'ESM2_650M', 'ESM2_3B', 'ESM2_15B', 'Wavenet', 'RITA_s',
       'RITA_m', 'RITA_l', 'RITA_xl', 'Progen2_small', 'Progen2_medium',
       'Progen2_base', 'Progen2_large', 'Progen2_xlarge', 'GEMME', 'VESPA',
       'VESPAl', 'ProtGPT2', 'Tranception_S_no_retrieval',
       'Tranception_M_no_retrieval', 'Tranception_L_no_retrieval',
       'Tranception_S', 'Tranception_M', 'Tranception_L', 'TranceptEVE_S',
       'TranceptEVE_M', 'TranceptEVE_L', 'CARP_38M', 'CARP_600K', 'CARP_640M',
       'CARP_76M', 'MIF', 'MIFST', 'ESM-IF1', 'ProteinMPNN',
       'ProtSSN_k10_h512', 'ProtSSN_k10_h768', 'ProtSSN_k10_h1280',
       'ProtSSN_k20_h512', 'ProtSSN_k20_h768', 'ProtSSN_k20_h1280',
       'ProtSSN_k30_h512', 'ProtSSN_k30_h768', 'ProtSSN_k30_h1280',
       'ProtSSN_ensemble', 'ProteinMPNN_global_score']
    
    generated_processed = {'aa': [], 'pred': []}
    
    for aa_pos in df_baselines['pos'].unique():
        
        pos_specific_df = df_baselines[df_baselines['pos'] == aa_pos]
        wt_pos = pos_specific_df['wt'].values[0]
        # short list the aa in calc_mut
        pos_specific_df = pos_specific_df[pos_specific_df['mut'].isin(calc_mut)]
        softmax_sum = np.sum(np.power(softmax_temp, pos_specific_df[args.weight_scheme]))
        pos_specific_df['proteinmpnn_weight'] = np.power(softmax_temp, pos_specific_df[args.weight_scheme]) / softmax_sum
        
        if (pos_specific_df['mut'] == 'C').any() and (pos_specific_df['mut'] == 'S').any():
            pos_specific_df.loc[pos_specific_df['mut'].isin(['C', 'S']), 'proteinmpnn_weight'] = pos_specific_df[pos_specific_df['mut'].isin(['C', 'S'])]['proteinmpnn_weight'].sum()
        # merge the column proteinmpnn_weight back to df_baselines
        
        if (pos_specific_df['mut'] == 'T').any() and (pos_specific_df['mut'] == 'V').any():
            pos_specific_df.loc[pos_specific_df['mut'].isin(['T', 'V']), 'proteinmpnn_weight'] = pos_specific_df[pos_specific_df['mut'].isin(['T', 'V'])]['proteinmpnn_weight'].sum()
            
        if (pos_specific_df['mut'] == 'D').any() and (pos_specific_df['mut'] == 'N').any() and (pos_specific_df['mut'] == 'I').any() and (pos_specific_df['mut'] == 'L').any() and (pos_specific_df['mut'] == 'M').any():
            pos_specific_df.loc[pos_specific_df['mut'].isin(['D', 'N', 'I', 'L', 'M']), 'proteinmpnn_weight'] = pos_specific_df[pos_specific_df['mut'].isin(['D', 'N', 'I', 'L', 'M'])]['proteinmpnn_weight'].sum()
            
        if (pos_specific_df['mut'] == 'E').any() and (pos_specific_df['mut'] == 'Q').any() and (pos_specific_df['mut'] == 'K').any():
            pos_specific_df.loc[pos_specific_df['mut'].isin(['E', 'Q', 'K']), 'proteinmpnn_weight'] = pos_specific_df[pos_specific_df['mut'].isin(['E', 'Q', 'K'])]['proteinmpnn_weight'].sum()
        
        if (pos_specific_df['mut'] == 'F').any() and (pos_specific_df['mut'] == 'R').any():
            pos_specific_df.loc[pos_specific_df['mut'].isin(['F', 'R']), 'proteinmpnn_weight'] = pos_specific_df[pos_specific_df['mut'].isin(['F', 'R'])]['proteinmpnn_weight'].sum()
        # from IPython import embed; embed()
        # pos_specific_df = pos_specific_df.set_index('mutant')
        
        df_baselines.update(pos_specific_df)
        
        
        # df_baselines.loc[df_baselines['mutant'] == pos_specific_df['mutant'], 'proteinmpnn_weight'] = pos_specific_df['proteinmpnn_weight'].values
        
    
    
    generated_processed = {'aa': [], 'pred': [], 'pred_orig': []}
    
    for index, row in df_generated.iterrows():
        mt_aa = aa_map[row['aa']]
        pos = row['pos']
        if pos == 1 or pos == 55:
            continue
        
        # generated_processed['aa'].append(row['aa'])
        # generated_processed['pred'].append(row['CYS'] * 2)
        
        # for aa_name in ['ALA', 'CYS', 'SER', 'THR', 'VAL', 'ASP', 'ASN', 'ILE', 'LEU', 'MET', 'GLU', 'GLN', 'LYS', 'PHE', 'ARG', 'TYR', 'HIS', 'TRP']:
        for aa_name in ['CYS', 'SER', 'THR', 'VAL', 'ASP', 'ASN', 'ILE', 'LEU', 'MET', 'GLU', 'GLN', 'LYS', 'PHE', 'ARG']:
            row_aa = mt_aa + str(pos) + aa_map[aa_name]
            if not np.isnan(row[aa_name]):
                # add new row to generated_processed
                
                # df_baselines[df_baselines.index == row_aa]['proteinmpnn_weight'].iloc[0]
                
                try:
                    
                    generated_processed['pred'].append((row[aa_name] / 100) * df_baselines[df_baselines.index == row_aa]['proteinmpnn_weight'].iloc[0])
                    generated_processed['pred_orig'].append(row[aa_name])
                    generated_processed['aa'].append(row_aa)
                except:
                    continue
            else:
                # add new row to generated_processed
                generated_processed['aa'].append(row_aa)
                generated_processed['pred'].append(1e-5)
                generated_processed['pred_orig'].append(1e-5)
    
    output_df = pd.DataFrame(generated_processed)
    # merge output_df to df_baselines, by the column mutant to aa
    df_baselines = df_baselines.merge(output_df, left_on='mutant', right_on='aa')
    # exlucde some rows
    
    
    # df_baselines = df_baselines[~((df_baselines['wt'].isin(['C', 'S'])) & (df_baselines['mut'].str[0].isin(['C', 'S'])))]
    # # df_baselines = df_filtered.drop(columns=['wt', 'mut'])

    # df_baselines = df_baselines[~((df_baselines['wt'].isin(['T', 'V'])) & (df_baselines['mut'].str[0].isin(['T', 'V'])))]
    
    # df_baselines = df_baselines[~((df_baselines['wt'].isin(['D', 'N', 'I', 'L', 'M'])) & (df_baselines['mut'].str[0].isin(['D', 'N', 'I', 'L', 'M'])))]
    
    # df_baselines = df_baselines[~((df_baselines['wt'].isin(['E', 'Q', 'K'])) & (df_baselines['mut'].str[0].isin(['E', 'Q', 'K'])))]
    
    # df_baselines = df_baselines[~((df_baselines['wt'].isin(['F', 'R'])) & (df_baselines['mut'].str[0].isin(['F', 'R'])))]
    
    
    spearmanr_pred = spearmanr(df_baselines['pred'], df_baselines['DMS_score'])
    
    ndcg_pred = calc_ndcg(df_baselines['DMS_score'], df_baselines['pred'])
    
    # from IPython import embed; embed()
    
    return spearmanr_pred, ndcg_pred
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--generated', type=str, default='/home/qcx679/hantang/UAAG2/ProteinGymSampling/runProteinGym_DN7A_SACS2_half_mask_2/DN7A_SACS2/aa_distribution.csv', help='path to the generated file')
    parser.add_argument('--weight_scheme', type=str, default='ProteinMPNN', help='weight scheme')
    parser.add_argument('--softmax_temp', type=float, default=2.0, help='softmax temperature')
    
    args = parser.parse_args()
    for weight_scheme in ['ProteinMPNN', 'ProteinMPNN_global_score', 'DMS_score']:
        args.weight_scheme = weight_scheme
        for softmax_temp in [10, 1e2, 1e3, 1e5, 1e10, 1e100]:
            args.softmax_temp = softmax_temp
            print(f'weight_scheme: {weight_scheme}, softmax_temp: {softmax_temp}')
            spearnman, ndcg = main(args)
            print(f'Spearman: {spearnman.correlation}, NDCG: {ndcg}')
            print('-----------------------------------')