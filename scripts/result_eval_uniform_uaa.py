import pandas as pd
import os
from scipy.stats import spearmanr
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import argparse

# /home/qcx679/hantang/UAAG2/ProteinGymSampling/runCP2_NAA_half_mask_scaled/DN7A_SACS2/aa_distribution.csv
# /home/qcx679/hantang/UAAG2/data/2018_PNAS_DeltaG/210526_CP2_average_lin_fit_Delta_G.csv
# /home/qcx679/hantang/UAAG2/data/2018_PNAS_DeltaG/210526_PUMA_exp_fit_Delta_G.csv
# /home/qcx679/hantang/UAAG2/data/uaa_benchmark_csv/CP2_reframe.csv

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


uaa_identity = ['Abu', 'Nva', 'Nle', 'Ahp', 'Aoc', \
                'Tme', 'hSM', 'tBu', 'Cpa', 'Aib', \
                    'MeG', 'MeA', 'MeB', 'MeF', '2th', \
                        '3th', 'YMe', '2Np', 'Bzt']
uaa_identity = [x.upper() for x in uaa_identity]

aa_indentity = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', \
    'MET', 'ASN', 'PRO', 'GLN', 'ARG', \
        'SER', 'THR', 'VAL', 'TRP', 'TYR']

def get_plot(dataframe, x_col, y_col, hue_col, title, spr):
    spr = round(spr, 4)
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=dataframe[x_col], y=dataframe[y_col], hue=dataframe[hue_col])
    # plot the spr number on the top left corner
    plt.text(0.1, 0.9, f'Spearman: {spr}', fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)
    plt.legend()
    plt.savefig(f'{title}.png')


def special_plot(data_naa, data_uaa, title, spr):
    data_naa['source'] = 'NAA'
    data_uaa['source'] = 'NCAA'
    combined_df = pd.concat([data_naa, data_uaa], ignore_index=True)
    spr = round(spr, 4)
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=combined_df['pred'], y=combined_df['value'], hue=combined_df['source'], style=combined_df['source'], s=100)
    for name, group in combined_df.groupby('source'):
        sns.kdeplot(
        x=group['pred'], y=group['value'],
        fill=True, alpha=0.5, label=name
    )
        
    plt.text(0.1, 0.9, f'Spearman: {spr}', fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)
    plt.legend()
    plt.savefig(f'{title}.png')
    
def main(args):
    
    benchmark = args.benchmark
    aa_output = args.aa_output
    
    df_benchmark = pd.read_csv(benchmark)
    result = pd.read_csv(aa_output)
    # convert all the columns in result to capital letters
    result.columns = [col.upper() for col in result.columns]
    # # strip every space in the column names
    # result.columns = [col.replace(' ', '') for col in result.columns]
    # remove the space in every value in result, and convert them to interger, if not nan
    
    # for col in result.columns:
    #     result[col] = result[col].astype(str).str.replace(' ', '')
    # from IPython import embed; embed()
    # for col in ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE',
    #    'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL',
    #    'TRP', 'TYR', 'ABU', 'NVA', 'NLE', 'AHP', 'AOC', 'TME', 'HSM', 'TBU',
    #    'CPA', 'AIB', 'MEG', 'MEA', 'MEB', 'MEF', '2TH', '3TH', 'YME', '2NP',
    #    'BZT', 'UNK', 'INV']:
    #     result[col] = result[col].astype(float)
    # from IPython import embed; embed()
    df_benchmark['UAAG'] = np.nan
    df_benchmark['wt_UAAG'] = np.nan
    
    
    # from IPython import embed; embed()
    for index, row in result.iterrows():
        
        wt_aa = row['AA']
        pos = row['POS']
        if pos == 156:
            continue
        # from IPython import embed; embed()
        if not np.isnan(row[wt_aa]):
            wt_value = row[wt_aa]
        else:
            continue
        
        # if wt_aa in ['ALA', 'TRP', 'TYR']:
        #     wt_value *= (1/23)
        # elif wt_aa in ['HIS', 'CPA', 'PHE', 'ARG']:
        #     wt_value *= (2/23)
        # elif wt_aa in ['TYR', 'VAL', 'NLE', 'CYS', 'SER', 'ABU']:
        #     wt_value *= (3/23)
        # elif wt_aa in ['GLU', 'GLN', 'LYS', 'TBU']:
        #     wt_value *= (4/23)
        # elif wt_aa in ['ASP', 'ASN', 'ILE', 'LEU', 'MET', 'NLE']:
        #     wt_value *= (6/23)
            
        for aa_name in uaa_identity:
            if not np.isnan(row[aa_name]):
                mt_value = row[aa_name]
            else:
                continue
                # mt_value = 1e-5
            # add mt_value to df_benchmark by the same [wt_aa, pos, aa_name]
            df_benchmark.loc[
                (df_benchmark['aa'] == wt_aa) & 
                (df_benchmark['pos'] == pos) & 
                (df_benchmark['target'] == aa_name), 'UAAG'] = mt_value
            
            df_benchmark.loc[
                (df_benchmark['aa'] == wt_aa) & 
                (df_benchmark['pos'] == pos) & 
                (df_benchmark['target'] == aa_name), 'wt_UAAG'] = wt_value
        
        for aa_name in aa_indentity:
            if not np.isnan(row[aa_name]):
                mt_value = row[aa_name]
            else:
                continue
                # mt_value = 1e-5
            # add mt_value to df_benchmark by the same [wt_aa, pos, aa_name]
            df_benchmark.loc[
                (df_benchmark['aa'] == wt_aa) & 
                (df_benchmark['pos'] == pos) & 
                (df_benchmark['target'] == aa_name), 'UAAG'] = mt_value
            
            df_benchmark.loc[
                (df_benchmark['aa'] == wt_aa) & 
                (df_benchmark['pos'] == pos) & 
                (df_benchmark['target'] == aa_name), 'wt_UAAG'] = wt_value
        # for aa_name in ['TYR', 'VAL', 'NLE', 'CYS', 'SER', 'ABU']:
        #     if not np.isnan(row[aa_name]):
        #         mt_value = row[aa_name] / 100 * (3/23)
        #     else:
        #         mt_value = 1e-5 * (3/23)
        #     # add mt_value to df_benchmark by the same [wt_aa, pos, aa_name]
        #     df_benchmark.loc[
        #         (df_benchmark['aa'] == wt_aa) & 
        #         (df_benchmark['pos'] == pos) & 
        #         (df_benchmark['target'] == aa_name), 'UAAG'] = mt_value
            
        #     df_benchmark.loc[
        #         (df_benchmark['aa'] == wt_aa) & 
        #         (df_benchmark['pos'] == pos) & 
        #         (df_benchmark['target'] == aa_name), 'wt_UAAG'] = wt_value
            
        # for aa_name in ['GLU', 'GLN', 'LYS', 'TBU']:
        #     if not np.isnan(row[aa_name]):
        #         mt_value = row[aa_name] / 100 * (4/23)
        #     else:
        #         mt_value = 1e-5 * (4/23)
        #     # add mt_value to df_benchmark by the same [wt_aa, pos, aa_name]
        #     df_benchmark.loc[
        #         (df_benchmark['aa'] == wt_aa) & 
        #         (df_benchmark['pos'] == pos) & 
        #         (df_benchmark['target'] == aa_name), 'UAAG'] = mt_value
            
        #     df_benchmark.loc[
        #         (df_benchmark['aa'] == wt_aa) & 
        #         (df_benchmark['pos'] == pos) & 
        #         (df_benchmark['target'] == aa_name), 'wt_UAAG'] = wt_value
        # for aa_name in ['ASP', 'ASN', 'ILE', 'LEU', 'MET', 'NLE']:
        #     if not np.isnan(row[aa_name]):
        #         mt_value = row[aa_name] / 100 * (6/23)
        #     else:
        #         mt_value = 1e-5 * (6/23)
        #     # add mt_value to df_benchmark by the same [wt_aa, pos, aa_name]
        #     df_benchmark.loc[
        #         (df_benchmark['aa'] == wt_aa) & 
        #         (df_benchmark['pos'] == pos) & 
        #         (df_benchmark['target'] == aa_name), 'UAAG'] = mt_value
            
        #     df_benchmark.loc[
        #         (df_benchmark['aa'] == wt_aa) & 
        #         (df_benchmark['pos'] == pos) & 
        #         (df_benchmark['target'] == aa_name), 'wt_UAAG'] = wt_value
            
        
    
    # from IPython import embed; embed()
    df_benchmark['wt_UAAG'] = np.log(df_benchmark['wt_UAAG']/args.total_num)
    df_benchmark['UAAG'] = np.log(df_benchmark['UAAG']/args.total_num)
    df_benchmark['pred'] = df_benchmark['wt_UAAG'] - df_benchmark['UAAG']
    # df_benchmark['pred'] = np.log(df_benchmark['UAAG']) - np.log(df_benchmark['wt_UAAG'])
    # remove the rows with NaN in df_benchmark
    df_benchmark = df_benchmark.dropna()
    spearmanr_pred = spearmanr(df_benchmark['pred'], df_benchmark['value'])
    # spearmanr(df_benchmark['wt_UAAG'], df_benchmark['value'])
    spearmanr(df_benchmark['UAAG'], df_benchmark['value'])
    spearmanr(df_benchmark['wt_UAAG'], df_benchmark['value'])
    
    df_uaa = df_benchmark[df_benchmark['target'].isin(uaa_identity)]
    df_naa = df_benchmark[df_benchmark['target'].isin(aa_indentity)]
    
    # scatter dot coloured by aa
    
    # using seaborn to plot the scatter plot with dots coloured by aa
    
    args.output_dir = os.path.join('results', args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    
    get_plot(df_benchmark, 'pred', 'value', 'aa', f'{args.output_dir}/wt-my_all_aa', spearmanr(df_benchmark['pred'], df_benchmark['value']).statistic)
    get_plot(df_benchmark, 'UAAG', 'value', 'aa', f'{args.output_dir}/mt_all_aa', spearmanr(df_benchmark['UAAG'], df_benchmark['value']).statistic)
    get_plot(df_benchmark, 'wt_UAAG', 'value', 'aa', f'{args.output_dir}/wt_all_aa', spearmanr(df_benchmark['wt_UAAG'], df_benchmark['value']).statistic)
    get_plot(df_naa, 'pred', 'value', 'aa', f'{args.output_dir}/wt-mt_naa_aa', spearmanr(df_naa['pred'], df_naa['value']).statistic)
    get_plot(df_naa, 'UAAG', 'value', 'aa', f'{args.output_dir}/mt_naa_aa', spearmanr(df_naa['UAAG'], df_naa['value']).statistic)
    get_plot(df_naa, 'wt_UAAG', 'value', 'aa', f'{args.output_dir}/wt_naa_aa', spearmanr(df_naa['wt_UAAG'], df_naa['value']).statistic)
    get_plot(df_uaa, 'pred', 'value', 'aa', f'{args.output_dir}/wt-mt_uaa_aa', spearmanr(df_uaa['pred'], df_uaa['value']).statistic)
    get_plot(df_uaa, 'UAAG', 'value', 'aa', f'{args.output_dir}/mt_uaa_aa', spearmanr(df_uaa['UAAG'], df_uaa['value']).statistic)
    get_plot(df_uaa, 'wt_UAAG', 'value', 'aa', f'{args.output_dir}/wt_uaa_aa', spearmanr(df_uaa['wt_UAAG'], df_uaa['value']).statistic)
    
    get_plot(df_benchmark, 'pred', 'value', 'target', f'{args.output_dir}/wt-mt_all_target', spearmanr(df_benchmark['pred'], df_benchmark['value']).statistic)
    get_plot(df_benchmark, 'UAAG', 'value', 'target', f'{args.output_dir}/mt_all_target', spearmanr(df_benchmark['UAAG'], df_benchmark['value']).statistic)
    get_plot(df_benchmark, 'wt_UAAG', 'value', 'target', f'{args.output_dir}/wt_all_target', spearmanr(df_benchmark['wt_UAAG'], df_benchmark['value']).statistic)
    get_plot(df_naa, 'pred', 'value', 'target', f'{args.output_dir}/wt-mt_naa_target', spearmanr(df_naa['pred'], df_naa['value']).statistic)
    get_plot(df_naa, 'UAAG', 'value', 'target', f'{args.output_dir}/mt_naa_target', spearmanr(df_naa['UAAG'], df_naa['value']).statistic)
    get_plot(df_naa, 'wt_UAAG', 'value', 'target', f'{args.output_dir}/wt_naa_target', spearmanr(df_naa['wt_UAAG'], df_naa['value']).statistic)
    get_plot(df_uaa, 'pred', 'value', 'target', f'{args.output_dir}/wt-mt_uaa_target', spearmanr(df_uaa['pred'], df_uaa['value']).statistic)
    get_plot(df_uaa, 'UAAG', 'value', 'target', f'{args.output_dir}/mt_uaa_target', spearmanr(df_uaa['UAAG'], df_uaa['value']).statistic)
    get_plot(df_uaa, 'wt_UAAG', 'value', 'target', f'{args.output_dir}/wt_uaa_target', spearmanr(df_uaa['wt_UAAG'], df_uaa['value']).statistic)
    special_plot(df_naa, df_uaa, f'{args.output_dir}/poseter', spearmanr(df_benchmark['pred'], df_benchmark['value']).statistic)
    from IPython import embed; embed()
    
    ndcg_pred = calc_ndcg(df_benchmark['value'], df_benchmark['pred'])
    calc_ndcg(df_benchmark['value'], df_benchmark['wt_UAAG'])
    calc_ndcg(df_benchmark['value'], df_benchmark['UAAG'])
    calc_ndcg(df_uaa['value'], df_uaa['UAAG'])
    calc_ndcg(df_naa['value'], df_naa['UAAG'])
    spearmanr(df_uaa['pred'], df_uaa['value'])
    spearmanr(df_naa['pred'], df_naa['value'])
    spearmanr(df_naa['value'], df_naa['pred'])
    spearmanr(df_uaa['UAAG'], df_uaa['value'])
    spearmanr(df_naa['UAAG'], df_naa['value'])
    spearmanr(df_uaa['wt_UAAG'], df_uaa['value'])
    spearmanr(df_naa['wt_UAAG'], df_naa['value'])
    
        
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate UAA results.")
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="Path to the input CSV file containing UAAG2 results.",
    )
    parser.add_argument(
        "--aa_output",
        type=str,
        required=True,
        help="Path to the output CSV file for saving the evaluation results.",
    )
    
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
        help="Total number of samples to be used for evaluation.",
    )
    
    args = parser.parse_args()
    
    main(args)