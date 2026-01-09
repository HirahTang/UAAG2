import pandas as pd
import os
from scipy.stats import spearmanr
import numpy as np
import argparse
from matplotlib import pyplot as plt
import seaborn as sns

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

def get_plot(dataframe, x_col, y_col, hue_col, title, spr):
    spr = round(spr, 4)
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=dataframe[x_col], y=dataframe[y_col], hue=dataframe[hue_col])
    # plot the spr number on the top left corner
    plt.text(0.1, 0.9, f'Spearman: {spr}', fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)
    plt.legend()
    plt.savefig(f'{title}.png')

def main(args):
    # df_ground_truth = pd.read_csv(args.ground_truth)
    df_generated = pd.read_csv(args.generated)
    df_baselines = pd.read_csv(args.baselines)
    df_baselines['wt'], df_baselines['pos'], df_baselines['mut'] = zip(*df_baselines['mutant'].map(divide_mutant))
    # from IPython import embed; embed()
    # Spearman correlation
    
    
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
       'ProtSSN_ensemble']
    
    generated_processed = {'aa': [], 'pred': [], 'wt': []}
    
    for index, row in df_generated.iterrows():
        # from IPython import embed; embed()
        mt_aa = aa_map[row['aa']]
        pos = row['pos']
        wt_value = mt_aa + str(pos) + mt_aa
        if row['aa'] == "PRO":
            continue
        if not np.isnan(row[row['aa']]):
            wt_value = row[row['aa']]
        else:
            wt_value = 1
        # if row['aa'] in ['CYS', 'SER', 'THR', 'VAL', 'PHE', 'ARG']:
        #     wt_value = wt_value * (2 / 18)
        # elif row['aa'] in ['ASP', 'ASN', 'ILE', 'LEU', 'MET']:
        #     wt_value = wt_value * (5 / 18)
        # elif row['aa'] in ['GLU', 'GLN', 'LYS']:
        #     wt_value = wt_value * (3 / 18)
        # elif row['aa'] in ['ALA', 'HIS', 'TYR', 'TRP']:
        #     wt_value = wt_value * (1 / 18)
        for aa_name in aa_map.keys():
            
            row_aa = mt_aa + str(pos) + aa_map[aa_name]
            if aa_name == "PRO":
                continue
            if not np.isnan(row[aa_name]):
                generated_processed['aa'].append(row_aa)
                generated_processed['pred'].append(row[aa_name])
                generated_processed['wt'].append(wt_value)
            else:
                generated_processed['aa'].append(row_aa)
                generated_processed['pred'].append(1)
                generated_processed['wt'].append(wt_value)
                # generated_processed['aa'].append(row_aa)
                # generated_processed['pred'].append(1e-5)
                # generated_processed['wt'].append(wt_value)
        # for aa_name in ['CYS', 'SER', 'THR', 'VAL', 'PHE', 'ARG']:
        #     row_aa = mt_aa + str(pos) + aa_map[aa_name]
        #     if not np.isnan(row[aa_name]):
        #         # add new row to generated_processed
        #         generated_processed['aa'].append(row_aa)
        #         generated_processed['pred'].append((row[aa_name] / 100) * (2 / 18))
        #         generated_processed['wt'].append(wt_value)
        #     else:
        #         # add new row to generated_processed
        #         generated_processed['aa'].append(row_aa)
        #         generated_processed['pred'].append(1e-5 * (2 / 18))
        #         generated_processed['wt'].append(wt_value)
        
        # for aa_name in ['ASP', 'ASN', 'ILE', 'LEU', 'MET']:
        #     row_aa = mt_aa + str(pos) + aa_map[aa_name]
        #     if not np.isnan(row[aa_name]):
        #         # add new row to generated_processed
        #         generated_processed['aa'].append(row_aa)
        #         generated_processed['pred'].append((row[aa_name] / 100) * (5 / 18))
        #         generated_processed['wt'].append(wt_value)
        #     else:
        #         # add new row to generated_processed
        #         generated_processed['aa'].append(row_aa)
        #         generated_processed['pred'].append(1e-5 * (5 / 18))
        #         generated_processed['wt'].append(wt_value)
                
        # for aa_name in ['GLU', 'GLN', 'LYS']:
        #     row_aa = mt_aa + str(pos) + aa_map[aa_name]
        #     if not np.isnan(row[aa_name]):
        #         # add new row to generated_processed
        #         generated_processed['aa'].append(row_aa)
        #         generated_processed['pred'].append((row[aa_name] / 100) * (3 / 18))
        #         generated_processed['wt'].append(wt_value)
        #     else:
        #         # add new row to generated_processed
        #         generated_processed['aa'].append(row_aa)
        #         generated_processed['pred'].append(1e-5 * (3 / 18))
        #         generated_processed['wt'].append(wt_value)
        # # check if row['CYS'] is NaN
        
        # for aa_name in ['ALA', 'HIS', 'TYR', 'TRP']:
        #     row_aa = mt_aa + str(pos) + aa_map[aa_name]
        #     if not np.isnan(row[aa_name]):
        #         # add new row to generated_processed
        #         generated_processed['aa'].append(row_aa)
        #         generated_processed['pred'].append((row[aa_name]) / 100 * (1 / 18))
        #         generated_processed['wt'].append(wt_value)
        #     else:
        #         # add new row to generated_processed
        #         generated_processed['aa'].append(row_aa)
        #         generated_processed['pred'].append(1e-5 * (1 / 18))
        #         generated_processed['wt'].append(wt_value)
                
        
    
    output_df = pd.DataFrame(generated_processed)
    # from IPython import embed; embed()
    # output_df['UAAG'] = -np.log(output_df['wt']) - np.log(output_df['pred'])
    # output_df['UAAG'] = np.log(output_df['wt']) - np.log(output_df['pred'])
    output_df["pred"] = np.log(output_df["pred"]/args.total_num)
    output_df["wt"] = np.log(output_df["wt"]/args.total_num)
    output_df['UNAAGI'] =  -output_df['wt'] + output_df['pred']
    # output_df['UAAG'] = 
    
    # from IPython import embed; embed()
    # merge output_df to df_baselines, by the column mutant to aa
    df_baselines = df_baselines.merge(output_df, left_on='mutant', right_on='aa')
    
    # df_baselines = df_baselines[~((df_baselines['wt'].isin(['C', 'S'])) & (df_baselines['mut'].str[0].isin(['C', 'S'])))]
    # # df_baselines = df_filtered.drop(columns=['wt', 'mut'])

    # df_baselines = df_baselines[~((df_baselines['wt'].isin(['T', 'V'])) & (df_baselines['mut'].str[0].isin(['T', 'V'])))]
    
    # df_baselines = df_baselines[~((df_baselines['wt'].isin(['D', 'N', 'I', 'L', 'M'])) & (df_baselines['mut'].str[0].isin(['D', 'N', 'I', 'L', 'M'])))]
    
    # df_baselines = df_baselines[~((df_baselines['wt'].isin(['E', 'Q', 'K'])) & (df_baselines['mut'].str[0].isin(['E', 'Q', 'K'])))]
    
    # df_baselines = df_baselines[~((df_baselines['wt'].isin(['F', 'R'])) & (df_baselines['mut'].str[0].isin(['F', 'R'])))]
    
    spearmanr_pred = spearmanr(df_baselines['UNAAGI'], df_baselines['DMS_score'])
    # approach_a = spearmanr(df_baselines['wt_y'] - df_baselines['pred'], df_baselines['DMS_score'])
    # spearmanr(df_baselines['pred'], df_baselines['DMS_score'])
    # approch_b = spearmanr(-np.log(df_baselines['pred']/500) + np.log(df_baselines['wt_y']/500), df_baselines['DMS_score'])
    # spearmanr(-df_baselines['pred']/500 + df_baselines['wt_y']/500, df_baselines['DMS_score'])
    
    # spearmanr(df_baselines['wt_y'], df_baselines['DMS_score'])
    args.output_dir = os.path.join('results', args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    # from IPython import embed; embed()
    df_baselines.to_csv(f'{args.output_dir}/full_table.csv', index=False)
    try:
        ndcg_pred = calc_ndcg(df_baselines['DMS_score'], df_baselines['UNAAGI'])
    except:
        ndcg_pred = 0
    results = {'model': [],
               'spearmanr_pred': [], 'ndcg_pred': []}
    
    for model in baseline_list:
        spearmanr_model = spearmanr(df_baselines[model], df_baselines['DMS_score'])
        try:
            ndcg_model = calc_ndcg(df_baselines['DMS_score'], df_baselines[model])
        except:
            ndcg_model = 0
        results['model'].append(model)
        results['spearmanr_pred'].append(spearmanr_model.correlation)
        results['ndcg_pred'].append(ndcg_model)
        
    results['model'].append('UNAAGI')
    results['spearmanr_pred'].append(spearmanr_pred.correlation)
    results['ndcg_pred'].append(ndcg_pred)
    
    
    
    get_plot(df_baselines, 'UNAAGI', 'DMS_score', 'wt_x', f'{args.output_dir}/wt-mut_all_aa', spearmanr_pred.statistic)
    get_plot(df_baselines, 'pred', 'DMS_score', 'wt_x', f'{args.output_dir}/mt_all_aa', spearmanr(df_baselines['pred'], df_baselines['DMS_score']).statistic)
    get_plot(df_baselines, 'wt_y', 'DMS_score', 'wt_x', f'{args.output_dir}/wt_all_aa', spearmanr(df_baselines['wt_y'], df_baselines['DMS_score']).statistic)
    
    get_plot(df_baselines, 'UNAAGI', 'DMS_score', 'mut', f'{args.output_dir}/wt-mut_all_target', spearmanr_pred.statistic)
    get_plot(df_baselines, 'pred', 'DMS_score', 'mut', f'{args.output_dir}/mt_all_target', spearmanr(df_baselines['pred'], df_baselines['DMS_score']).statistic)
    get_plot(df_baselines, 'wt_y', 'DMS_score', 'mut', f'{args.output_dir}/wt_all_target', spearmanr(df_baselines['wt_y'], df_baselines['DMS_score']).statistic)
    
    
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{args.output_dir}/results.csv', index=False)
        
    # from IPython import embed; embed()
    
if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--ground_truth', type=str, default="/home/qcx679/hantang/UAAG/data/DMS/DN7A_SACS2_Tsuboyama_2023_1JIC.csv")
    parser.add_argument('--generated', type=str, default= "/home/qcx679/hantang/UAAG2/ProteinGymSampling/runDN7A_SACS2_half_mask_virtual_node_noise_01/DN7A_SACS2/aa_distribution.csv")
    parser.add_argument('--baselines', type=str, default="/home/qcx679/hantang/UAAG2/data/DN7A_SACS2_baselines.csv")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/",
        help="Directory to save the evaluation results.",
    )
    parser.add_argument(
        "--total_num",
        type=int,
        default=500,
        help="The total number of samples to be used for evaluation.",
    )
    args = parser.parse_args()
    main(args)