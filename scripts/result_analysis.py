from unittest import result
import pandas as pd
import os
from scipy.stats import spearmanr
import numpy as np
from sklearn.metrics import roc_auc_score, matthews_corrcoef


def calc_toprecall(true_scores, model_scores, top_true=10, top_model=10):  
    top_true = (true_scores >= np.percentile(true_scores, 100-top_true))
    top_model = (model_scores >= np.percentile(model_scores, 100-top_model))
    
    TP = (top_true) & (top_model)
    recall = TP.sum() / (top_true.sum()) if top_true.sum() > 0 else 0
    
    return (recall)

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



def main(output_path):
    
    ground_truth = "/home/qcx679/hantang/UAAG/data/DMS/DN7A_SACS2_Tsuboyama_2023_1JIC.csv"
    generated = "/home/qcx679/hantang/UAAG2/ProteinGymSampling/runProteinGym_DN7A_SACS2_eval/DN7A_SACS2/aa_distribution.csv"
    baselines = "/home/qcx679/hantang/UAAG/data/DMS/full_benchmark/DN7A_SACS2_Tsuboyama_2023_1JIC.csv"

    df_gt = pd.read_csv(ground_truth)
    df_gen = pd.read_csv(generated)
    baselines = pd.read_csv(baselines)
    
    df_gt['aa'] = df_gt['mutant'].apply(lambda x: x[0])
    df_gt['pos'] = df_gt['mutant'].apply(lambda x: int(x[1:-1]))
    df_gt['mut'] = df_gt['mutant'].apply(lambda x: x[-1])

    baselines['pos'] = baselines['mutant'].apply(lambda x: int(x[1:-1]))
    baselines['aa'] = baselines['mutant'].apply(lambda x: x[0])
    baselines['mut'] = baselines['mutant'].apply(lambda x: x[-1])
    
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

    baseline_models = ['Site_Independent', 'EVmutation', 'DeepSequence_single',
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

    results = test_model(df_gt, df_gen)
    
    spearman_corr_2, ndcg_score_dict_2, topkrecall_dict_2, spearman_corr_3, \
        ndcg_score_dict_3, topkrecall_dict_3, spearman_corr_4, ndcg_score_dict_4, \
            topkrecall_dict_4 , spearman_corr_5, ndcg_score_dict_5, topkrecall_dict_5, \
                spearman_corr_6, ndcg_score_dict_6, topkrecall_dict_6 = results
    
    
   
    
    # convert dictionary to dataframe
    spearman_corr_2 = pd.DataFrame.from_dict(spearman_corr_2, orient='index', columns=['spearman_corr_2'])
    ndcg_score_dict_2 = pd.DataFrame.from_dict(ndcg_score_dict_2, orient='index', columns=['ndcg_score_dict_2'])
    topkrecall_dict_2 = pd.DataFrame.from_dict(topkrecall_dict_2, orient='index', columns=['topkrecall_dict_2'])
    
    spearman_corr_3 = pd.DataFrame.from_dict(spearman_corr_3, orient='index', columns=['spearman_corr_3'])
    ndcg_score_dict_3 = pd.DataFrame.from_dict(ndcg_score_dict_3, orient='index', columns=['ndcg_score_dict_3'])
    topkrecall_dict_3 = pd.DataFrame.from_dict(topkrecall_dict_3, orient='index', columns=['topkrecall_dict_3'])
    
    spearman_corr_4 = pd.DataFrame.from_dict(spearman_corr_4, orient='index', columns=['spearman_corr_4'])
    ndcg_score_dict_4 = pd.DataFrame.from_dict(ndcg_score_dict_4, orient='index', columns=['ndcg_score_dict_4'])
    topkrecall_dict_4 = pd.DataFrame.from_dict(topkrecall_dict_4, orient='index', columns=['topkrecall_dict_4'])
    
    spearman_corr_5 = pd.DataFrame.from_dict(spearman_corr_5, orient='index', columns=['spearman_corr_5'])
    ndcg_score_dict_5 = pd.DataFrame.from_dict(ndcg_score_dict_5, orient='index', columns=['ndcg_score_dict_5'])
    topkrecall_dict_5 = pd.DataFrame.from_dict(topkrecall_dict_5, orient='index', columns=['topkrecall_dict_5'])
    
    spearman_corr_6 = pd.DataFrame.from_dict(spearman_corr_6, orient='index', columns=['spearman_corr_6'])
    ndcg_score_dict_6 = pd.DataFrame.from_dict(ndcg_score_dict_6, orient='index', columns=['ndcg_score_dict_6'])
    topkrecall_dict_6 = pd.DataFrame.from_dict(topkrecall_dict_6, orient='index', columns=['topkrecall_dict_6'])
    
    result_df = pd.concat([spearman_corr_2, ndcg_score_dict_2, topkrecall_dict_2, spearman_corr_3, ndcg_score_dict_3, topkrecall_dict_3, \
                            spearman_corr_4, ndcg_score_dict_4, topkrecall_dict_4, spearman_corr_5, ndcg_score_dict_5, topkrecall_dict_5, \
                            spearman_corr_6, ndcg_score_dict_6, topkrecall_dict_6], axis=1)
    
    for model in baseline_models:

        result_baseline = test_baseline(df_gt, baselines, model)
        
        spearman_corr_2_baseline, ndcg_score_dict_2_baseline, topkrecall_dict_2_baseline, spearman_corr_3_baseline, \
            ndcg_score_dict_3_baseline, topkrecall_dict_3_baseline, spearman_corr_4_baseline, ndcg_score_dict_4_baseline, \
                topkrecall_dict_4_baseline , spearman_corr_5_baseline, ndcg_score_dict_5_baseline, topkrecall_dict_5_baseline, \
                    spearman_corr_6_baseline, ndcg_score_dict_6_baseline, topkrecall_dict_6_baseline = result_baseline
                    
        spearman_corr_2_baseline = pd.DataFrame.from_dict(spearman_corr_2_baseline, orient='index', columns=[f'spearman_corr_2_{model}'])
        ndcg_score_dict_2_baseline = pd.DataFrame.from_dict(ndcg_score_dict_2_baseline, orient='index', columns=[f'ndcg_score_dict_2_{model}'])
        topkrecall_dict_2_baseline = pd.DataFrame.from_dict(topkrecall_dict_2_baseline, orient='index', columns=[f'topkrecall_dict_2_{model}'])
        
        spearman_corr_3_baseline = pd.DataFrame.from_dict(spearman_corr_3_baseline, orient='index', columns=[f'spearman_corr_3_{model}'])
        ndcg_score_dict_3_baseline = pd.DataFrame.from_dict(ndcg_score_dict_3_baseline, orient='index', columns=[f'ndcg_score_dict_3_{model}'])
        topkrecall_dict_3_baseline = pd.DataFrame.from_dict(topkrecall_dict_3_baseline, orient='index', columns=[f'topkrecall_dict_3_{model}'])
        
        spearman_corr_4_baseline = pd.DataFrame.from_dict(spearman_corr_4_baseline, orient='index', columns=[f'spearman_corr_4_{model}'])
        ndcg_score_dict_4_baseline = pd.DataFrame.from_dict(ndcg_score_dict_4_baseline, orient='index', columns=[f'ndcg_score_dict_4_{model}'])
        topkrecall_dict_4_baseline = pd.DataFrame.from_dict(topkrecall_dict_4_baseline, orient='index', columns=[f'topkrecall_dict_4_{model}'])
        
        spearman_corr_5_baseline = pd.DataFrame.from_dict(spearman_corr_5_baseline, orient='index', columns=[f'spearman_corr_5_{model}'])
        ndcg_score_dict_5_baseline = pd.DataFrame.from_dict(ndcg_score_dict_5_baseline, orient='index', columns=[f'ndcg_score_dict_5_{model}'])
        topkrecall_dict_5_baseline = pd.DataFrame.from_dict(topkrecall_dict_5_baseline, orient='index', columns=[f'topkrecall_dict_5_{model}'])
        
        spearman_corr_6_baseline = pd.DataFrame.from_dict(spearman_corr_6_baseline, orient='index', columns=[f'spearman_corr_6_{model}'])
        ndcg_score_dict_6_baseline = pd.DataFrame.from_dict(ndcg_score_dict_6_baseline, orient='index', columns=[f'ndcg_score_dict_6_{model}'])
        topkrecall_dict_6_baseline = pd.DataFrame.from_dict(topkrecall_dict_6_baseline, orient='index', columns=[f'topkrecall_dict_6_{model}'])
        
    # merge all the dataframes
    
        result_df = pd.concat([result_df, spearman_corr_2_baseline, ndcg_score_dict_2_baseline, \
                            topkrecall_dict_2_baseline, spearman_corr_3_baseline, ndcg_score_dict_3_baseline, topkrecall_dict_3_baseline, \
                            spearman_corr_4_baseline, ndcg_score_dict_4_baseline, topkrecall_dict_4_baseline, spearman_corr_5_baseline, \
                            ndcg_score_dict_5_baseline, topkrecall_dict_5_baseline, spearman_corr_6_baseline, ndcg_score_dict_6_baseline, \
                            topkrecall_dict_6_baseline], axis=1)
    
    output_path = os.path.join(output_path, 'result_df.csv')
    result_df.to_csv(output_path)
    # from IPython import embed; embed()
    
def eval_2_atoms(df_gt, df_gen, position):
    
    if df_gen[df_gen['pos'] == position]['aa'].iloc[0] in ['CYS', 'SER']:
        return np.nan, np.nan, np.nan
    
    cys = df_gen[df_gen['pos'] == position][['CYS']].iloc[0]['CYS']
    ser = df_gen[df_gen['pos'] == position][['SER']].iloc[0]['SER']
    
    cys = 0 if pd.isna(cys) else cys
    ser = 0 if pd.isna(ser) else ser
    
    try:
        cys_gt = df_gt[(df_gt['pos'] == position) & (df_gt['mut'] == 'C')]['DMS_score'].iloc[0]
    except:
        return np.nan, np.nan, np.nan
    try:
        ser_gt = df_gt[(df_gt['pos'] == position) & (df_gt['mut'] == 'S')]['DMS_score'].iloc[0]
    except:
        return np.nan, np.nan, np.nan
    
    ndcg_score = calc_ndcg(np.array([cys_gt, ser_gt]), np.array([cys, ser]), top=1, quantile=False)
    spearman_score = spearmanr([cys, ser], [cys_gt, ser_gt]).statistic
    topkrecall_score = calc_toprecall(np.array([cys_gt, ser_gt]), np.array([cys, ser]), top_true=10, top_model=10)

    return spearman_score, ndcg_score, topkrecall_score

def eval_3_atoms(df_gt, df_gen, position):
    
    if df_gen[df_gen['pos'] == position]['aa'].iloc[0] in ['THR', 'VAL']:
        return np.nan, np.nan, np.nan
    
    thr = df_gen[df_gen['pos'] == position][['THR']].iloc[0]['THR']
    val = df_gen[df_gen['pos'] == position][['VAL']].iloc[0]['VAL']
    
    thr = 0 if pd.isna(thr) else thr
    val = 0 if pd.isna(val) else val
        
    try:
        thr_gt = df_gt[(df_gt['pos'] == position) & (df_gt['mut'] == 'T')]['DMS_score'].iloc[0]
    except:
        return np.nan, np.nan, np.nan
    try:
        val_gt = df_gt[(df_gt['pos'] == position) & (df_gt['mut'] == 'V')]['DMS_score'].iloc[0]
    except:
        return np.nan, np.nan, np.nan
    
    ndcg_score = calc_ndcg(np.array([thr_gt, val_gt]), np.array([thr, val]), top=1, quantile=False)
    spearman_score = spearmanr([thr, val], [thr_gt, val_gt]).statistic
    topkrecall_score = calc_toprecall(np.array([thr_gt, val_gt]), np.array([thr, val]), top_true=10, top_model=10)

    return spearman_score, ndcg_score, topkrecall_score

def eval_4_atoms(df_gt, df_gen, position):
    
    if df_gen[df_gen['pos'] == position]['aa'].iloc[0] in ['ASP', 'ASN', 'LEU', 'ILE', 'MET']:
        return np.nan, np.nan, np.nan
    
    asp = df_gen[df_gen['pos'] == position][['ASP']].iloc[0]['ASP']
    asn = df_gen[df_gen['pos'] == position][['ASN']].iloc[0]['ASN']
    leu = df_gen[df_gen['pos'] == position][['LEU']].iloc[0]['LEU']
    ile = df_gen[df_gen['pos'] == position][['ILE']].iloc[0]['ILE']
    met = df_gen[df_gen['pos'] == position][['MET']].iloc[0]['MET']
    
    asp = 0 if pd.isna(asp) else asp
    asn = 0 if pd.isna(asn) else asn
    leu = 0 if pd.isna(leu) else leu
    ile = 0 if pd.isna(ile) else ile
    met = 0 if pd.isna(met) else met
    
    try:
        asp_gt = df_gt[(df_gt['pos'] == position) & (df_gt['mut'] == 'D')]['DMS_score'].iloc[0]
    
        asn_gt = df_gt[(df_gt['pos'] == position) & (df_gt['mut'] == 'N')]['DMS_score'].iloc[0]
    
        leu_gt = df_gt[(df_gt['pos'] == position) & (df_gt['mut'] == 'L')]['DMS_score'].iloc[0]
        
        ile_gt = df_gt[(df_gt['pos'] == position) & (df_gt['mut'] == 'I')]['DMS_score'].iloc[0]
        
        met_gt = df_gt[(df_gt['pos'] == position) & (df_gt['mut'] == 'M')]['DMS_score'].iloc[0]
        
    except:
        return np.nan, np.nan, np.nan
    
    ndcg_score = calc_ndcg(np.array([asp_gt, asn_gt, leu_gt, ile_gt, met_gt]), np.array([asp, asn, leu, ile, met]), top=1, quantile=False)
    spearman_score = spearmanr([asp, asn, leu, ile, met], [asp_gt, asn_gt, leu_gt, ile_gt, met_gt]).statistic
    topkrecall_score = calc_toprecall(np.array([asp_gt, asn_gt, leu_gt, ile_gt, met_gt]), np.array([asp, asn, leu, ile, met]), top_true=10, top_model=10)
    
    return spearman_score, ndcg_score, topkrecall_score

def eval_5_atoms(df_gt, df_gen, position):
    
    if df_gen[df_gen['pos'] == position]['aa'].iloc[0] in ['GLN', 'GLU', 'LYS']:
        return np.nan, np.nan, np.nan
    
    gln = df_gen[df_gen['pos'] == position][['GLN']].iloc[0]['GLN']
    glu = df_gen[df_gen['pos'] == position][['GLU']].iloc[0]['GLU']
    lys = df_gen[df_gen['pos'] == position][['LYS']].iloc[0]['LYS']
    
    gln = 0 if pd.isna(gln) else gln
    glu = 0 if pd.isna(glu) else glu
    lys = 0 if pd.isna(lys) else lys
    
    try:
        gln_gt = df_gt[(df_gt['pos'] == position) & (df_gt['mut'] == 'Q')]['DMS_score'].iloc[0]
        
        glu_gt = df_gt[(df_gt['pos'] == position) & (df_gt['mut'] == 'E')]['DMS_score'].iloc[0]
        
        lys_gt = df_gt[(df_gt['pos'] == position) & (df_gt['mut'] == 'K')]['DMS_score'].iloc[0]
        
    except:
        return np.nan, np.nan, np.nan
    
    ndcg_score = calc_ndcg(np.array([gln_gt, glu_gt, lys_gt]), np.array([gln, glu, lys]), top=1, quantile=False)
    spearman_score = spearmanr([gln, glu, lys], [gln_gt, glu_gt, lys_gt]).statistic
    topkrecall_score = calc_toprecall(np.array([gln_gt, glu_gt, lys_gt]), np.array([gln, glu, lys]), top_true=10, top_model=10)
    
    return spearman_score, ndcg_score, topkrecall_score

def eval_6_atoms(df_gt, df_gen, position):
    
    if df_gen[df_gen['pos'] == position]['aa'].iloc[0] in ['ARG', 'HIS']:
        return np.nan, np.nan, np.nan
    
    arg = df_gen[df_gen['pos'] == position][['ARG']].iloc[0]['ARG']
    his = df_gen[df_gen['pos'] == position][['HIS']].iloc[0]['HIS']
    
    arg = 0 if pd.isna(arg) else arg
    his = 0 if pd.isna(his) else his
    
    try:
        arg_gt = df_gt[(df_gt['pos'] == position) & (df_gt['mut'] == 'R')]['DMS_score'].iloc[0]
        
        his_gt = df_gt[(df_gt['pos'] == position) & (df_gt['mut'] == 'H')]['DMS_score'].iloc[0]
        
    except:
        return np.nan, np.nan, np.nan
    
    ndcg_score = calc_ndcg(np.array([arg_gt, his_gt]), np.array([arg, his]), top=1, quantile=False)
    spearman_score = spearmanr([arg, his], [arg_gt, his_gt]).statistic
    topkrecall_score = calc_toprecall(np.array([arg_gt, his_gt]), np.array([arg, his]), top_true=10, top_model=10)
    
    return spearman_score, ndcg_score, topkrecall_score

def test_model(df_gt, df_gen):
    
    spearman_corr_2 = {}
    ndcg_score_dict_2 = {}
    topkrecall_dict_2 = {}
    
    spearman_corr_3 = {}
    ndcg_score_dict_3 = {}
    topkrecall_dict_3 = {}
    
    spearman_corr_4 = {}
    ndcg_score_dict_4 = {}
    topkrecall_dict_4 = {}
    
    spearman_corr_5 = {}
    ndcg_score_dict_5 = {}
    topkrecall_dict_5 = {}
    
    spearman_corr_6 = {}
    ndcg_score_dict_6 = {}
    topkrecall_dict_6 = {}
    
    for position in list(df_gen['pos']):
        spearman, ndcg, topkrecall = eval_2_atoms(df_gt, df_gen, position)
        spearman_corr_2[position] = spearman
        ndcg_score_dict_2[position] = ndcg
        topkrecall_dict_2[position] = topkrecall
        
        spearman, ndcg, topkrecall = eval_3_atoms(df_gt, df_gen, position)
        spearman_corr_3[position] = spearman
        ndcg_score_dict_3[position] = ndcg
        topkrecall_dict_3[position] = topkrecall
        
        spearman, ndcg, topkrecall = eval_4_atoms(df_gt, df_gen, position)
        spearman_corr_4[position] = spearman
        ndcg_score_dict_4[position] = ndcg
        topkrecall_dict_4[position] = topkrecall
        
        spearman, ndcg, topkrecall = eval_5_atoms(df_gt, df_gen, position)
        spearman_corr_5[position] = spearman
        ndcg_score_dict_5[position] = ndcg
        topkrecall_dict_5[position] = topkrecall
        
        spearman, ndcg, topkrecall = eval_6_atoms(df_gt, df_gen, position)
        spearman_corr_6[position] = spearman
        ndcg_score_dict_6[position] = ndcg
        topkrecall_dict_6[position] = topkrecall
        
    return spearman_corr_2, ndcg_score_dict_2, topkrecall_dict_2, spearman_corr_3, ndcg_score_dict_3, topkrecall_dict_3 \
        , spearman_corr_4, ndcg_score_dict_4, topkrecall_dict_4, spearman_corr_5, ndcg_score_dict_5, topkrecall_dict_5 \
        , spearman_corr_6, ndcg_score_dict_6, topkrecall_dict_6
        
def eval_2_atoms_baseline(df_gt, baselines, position, model):
    if df_gt[(df_gt['pos'] == position)]['aa'].iloc[0] in ['C', 'S']:
        return np.nan, np.nan, np.nan
    try:
        cys = baselines[(baselines['pos'] == position) & (baselines['mut'] == 'C')][model].iloc[0]
        ser = baselines[(baselines['pos'] == position) & (baselines['mut'] == 'S')][model].iloc[0]
        
        cys_gt = df_gt[(df_gt['pos'] == position) & (df_gt['mut'] == 'C')]['DMS_score'].iloc[0]
        ser_gt = df_gt[(df_gt['pos'] == position) & (df_gt['mut'] == 'S')]['DMS_score'].iloc[0]
    except:
        return np.nan, np.nan, np.nan
    
    ndcg_score = calc_ndcg(np.array([cys_gt, ser_gt]), np.array([cys, ser]), top=1, quantile=False)
    spearman_score = spearmanr([cys, ser], [cys_gt, ser_gt]).statistic
    topkrecall_score = calc_toprecall(np.array([cys_gt, ser_gt]), np.array([cys, ser]), top_true=10, top_model=10)
    
    return spearman_score, ndcg_score, topkrecall_score

def eval_3_atoms_baseline(df_gt, baselines, position, model):
    if df_gt[(df_gt['pos'] == position)]['aa'].iloc[0] in ['T', 'V']:
        return np.nan, np.nan, np.nan
    try:
        thr = baselines[(baselines['pos'] == position) & (baselines['mut'] == 'T')][model].iloc[0]
        val = baselines[(baselines['pos'] == position) & (baselines['mut'] == 'V')][model].iloc[0]
        
        thr_gt = df_gt[(df_gt['pos'] == position) & (df_gt['mut'] == 'T')]['DMS_score'].iloc[0]
        val_gt = df_gt[(df_gt['pos'] == position) & (df_gt['mut'] == 'V')]['DMS_score'].iloc[0]
    except:
        return np.nan, np.nan, np.nan
    
    ndcg_score = calc_ndcg(np.array([thr_gt, val_gt]), np.array([thr, val]), top=1, quantile=False)
    spearman_score = spearmanr([thr, val], [thr_gt, val_gt]).statistic
    topkrecall_score = calc_toprecall(np.array([thr_gt, val_gt]), np.array([thr, val]), top_true=10, top_model=10)
    
    return spearman_score, ndcg_score, topkrecall_score

def eval_4_atoms_baseline(df_gt, baselines, position, model):
    if df_gt[(df_gt['pos'] == position)]['aa'].iloc[0] in ['D', 'N', 'L', 'I', 'M']:
        return np.nan, np.nan, np.nan
    try:
        asp = baselines[(baselines['pos'] == position) & (baselines['mut'] == 'D')][model].iloc[0]
        asn = baselines[(baselines['pos'] == position) & (baselines['mut'] == 'N')][model].iloc[0]
        leu = baselines[(baselines['pos'] == position) & (baselines['mut'] == 'L')][model].iloc[0]
        ile = baselines[(baselines['pos'] == position) & (baselines['mut'] == 'I')][model].iloc[0]
        met = baselines[(baselines['pos'] == position) & (baselines['mut'] == 'M')][model].iloc[0]
        
        asp_gt = df_gt[(df_gt['pos'] == position) & (df_gt['mut'] == 'D')]['DMS_score'].iloc[0]
        asn_gt = df_gt[(df_gt['pos'] == position) & (df_gt['mut'] == 'N')]['DMS_score'].iloc[0]
        leu_gt = df_gt[(df_gt['pos'] == position) & (df_gt['mut'] == 'L')]['DMS_score'].iloc[0]
        ile_gt = df_gt[(df_gt['pos'] == position) & (df_gt['mut'] == 'I')]['DMS_score'].iloc[0]
        met_gt = df_gt[(df_gt['pos'] == position) & (df_gt['mut'] == 'M')]['DMS_score'].iloc[0]
    except:
        return np.nan, np.nan, np.nan  
    
    ndcg_score = calc_ndcg(np.array([asp_gt, asn_gt, leu_gt, ile_gt, met_gt]), np.array([asp, asn, leu, ile, met]), top=1, quantile=False)
    spearman_score = spearmanr([asp, asn, leu, ile, met], [asp_gt, asn_gt, leu_gt, ile_gt, met_gt]).statistic
    topkrecall_score = calc_toprecall(np.array([asp_gt, asn_gt, leu_gt, ile_gt, met_gt]), np.array([asp, asn, leu, ile, met]), top_true=10, top_model=10)
    
    return spearman_score, ndcg_score, topkrecall_score

def eval_5_atoms_baseline(df_gt, baselines, position, model):
    if df_gt[(df_gt['pos'] == position)]['aa'].iloc[0] in ['Q', 'E', 'K']:
        return np.nan, np.nan, np.nan
    try:
        gln = baselines[(baselines['pos'] == position) & (baselines['mut'] == 'Q')][model].iloc[0]
        glu = baselines[(baselines['pos'] == position) & (baselines['mut'] == 'E')][model].iloc[0]
        lys = baselines[(baselines['pos'] == position) & (baselines['mut'] == 'K')][model].iloc[0]
        
        gln_gt = df_gt[(df_gt['pos'] == position) & (df_gt['mut'] == 'Q')]['DMS_score'].iloc[0]
        glu_gt = df_gt[(df_gt['pos'] == position) & (df_gt['mut'] == 'E')]['DMS_score'].iloc[0]
        lys_gt = df_gt[(df_gt['pos'] == position) & (df_gt['mut'] == 'K')]['DMS_score'].iloc[0]
    except:
        return np.nan, np.nan, np.nan
    
    ndcg_score = calc_ndcg(np.array([gln_gt, glu_gt, lys_gt]), np.array([gln, glu, lys]), top=1, quantile=False)
    spearman_score = spearmanr([gln, glu, lys], [gln_gt, glu_gt, lys_gt]).statistic
    topkrecall_score = calc_toprecall(np.array([gln_gt, glu_gt, lys_gt]), np.array([gln, glu, lys]), top_true=10, top_model=10)
    
    return spearman_score, ndcg_score, topkrecall_score

def eval_6_atoms_baseline(df_gt, baselines, position, model):
    if df_gt[(df_gt['pos'] == position)]['aa'].iloc[0] in ['R', 'H']:
        return np.nan, np.nan, np.nan
    try:
        arg = baselines[(baselines['pos'] == position) & (baselines['mut'] == 'R')][model].iloc[0]
        his = baselines[(baselines['pos'] == position) & (baselines['mut'] == 'H')][model].iloc[0]
        
        arg_gt = df_gt[(df_gt['pos'] == position) & (df_gt['mut'] == 'R')]['DMS_score'].iloc[0]
        his_gt = df_gt[(df_gt['pos'] == position) & (df_gt['mut'] == 'H')]['DMS_score'].iloc[0]
    except:
        return np.nan, np.nan, np.nan
    
    ndcg_score = calc_ndcg(np.array([arg_gt, his_gt]), np.array([arg, his]), top=1, quantile=False)
    spearman_score = spearmanr([arg, his], [arg_gt, his_gt]).statistic
    topkrecall_score = calc_toprecall(np.array([arg_gt, his_gt]), np.array([arg, his]), top_true=10, top_model=10)
    
    return spearman_score, ndcg_score, topkrecall_score



def test_baseline(df_gt, baselines, model):
    spearman_corr_2 = {}
    ndcg_score_dict_2 = {}
    topkrecall_dict_2 = {}
    
    spearman_corr_3 = {}
    ndcg_score_dict_3 = {}
    topkrecall_dict_3 = {}
    
    spearman_corr_4 = {}
    ndcg_score_dict_4 = {}
    topkrecall_dict_4 = {}
    
    spearman_corr_5 = {}
    ndcg_score_dict_5 = {}
    topkrecall_dict_5 = {}
    
    spearman_corr_6 = {}
    ndcg_score_dict_6 = {}
    topkrecall_dict_6 = {}
    
    for position in list(baselines['pos']):
        
        spearman, ndcg, topkrecall = eval_2_atoms_baseline(df_gt, baselines, position, model)
        spearman_corr_2[position] = spearman
        ndcg_score_dict_2[position] = ndcg
        topkrecall_dict_2[position] = topkrecall
        
        spearman, ndcg, topkrecall = eval_3_atoms_baseline(df_gt, baselines, position, model)
        spearman_corr_3[position] = spearman
        ndcg_score_dict_3[position] = ndcg
        topkrecall_dict_3[position] = topkrecall
        
        spearman, ndcg, topkrecall = eval_4_atoms_baseline(df_gt, baselines, position, model)
        spearman_corr_4[position] = spearman
        ndcg_score_dict_4[position] = ndcg
        topkrecall_dict_4[position] = topkrecall
        
        spearman, ndcg, topkrecall = eval_5_atoms_baseline(df_gt, baselines, position, model)
        spearman_corr_5[position] = spearman
        ndcg_score_dict_5[position] = ndcg
        topkrecall_dict_5[position] = topkrecall
        
        spearman, ndcg, topkrecall = eval_6_atoms_baseline(df_gt, baselines, position, model)
        spearman_corr_6[position] = spearman
        ndcg_score_dict_6[position] = ndcg
        topkrecall_dict_6[position] = topkrecall
        
        
    return spearman_corr_2, ndcg_score_dict_2, topkrecall_dict_2, spearman_corr_3, ndcg_score_dict_3, topkrecall_dict_3 \
        , spearman_corr_4, ndcg_score_dict_4, topkrecall_dict_4, spearman_corr_5, ndcg_score_dict_5, topkrecall_dict_5 \
        , spearman_corr_6, ndcg_score_dict_6, topkrecall_dict_6
        
if __name__ == "__main__":
    main('/home/qcx679/hantang/UAAG2/ProteinGymSampling/runProteinGym_DN7A_SACS2_eval/DN7A_SACS2')