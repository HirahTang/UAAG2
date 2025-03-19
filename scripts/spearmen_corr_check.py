import pandas as pd
import os
from scipy.stats import spearmanr
import numpy as np
from sklearn.metrics import roc_auc_score, matthews_corrcoef

ground_truth = "/home/qcx679/hantang/UAAG/data/DMS/DN7A_SACS2_Tsuboyama_2023_1JIC.csv"
generated = "/home/qcx679/hantang/UAAG2/ProteinGymSampling/runProteinGym_DN7A_SACS2_eval/DN7A_SACS2/aa_distribution.csv"
baselines = "/home/qcx679/hantang/UAAG/data/DMS/full_benchmark/DN7A_SACS2_Tsuboyama_2023_1JIC.csv"
df_gt = pd.read_csv(ground_truth)
df_gen = pd.read_csv(generated)
baselines = pd.read_csv(baselines)
# from IPython import embed; embed()
# convert mutant from format like A43C to three columns: aa, pos, mut
df_gt['aa'] = df_gt['mutant'].apply(lambda x: x[0])
df_gt['pos'] = df_gt['mutant'].apply(lambda x: int(x[1:-1]))
df_gt['mut'] = df_gt['mutant'].apply(lambda x: x[-1])

baselines['pos'] = baselines['mutant'].apply(lambda x: int(x[1:-1]))
baselines['aa'] = baselines['mutant'].apply(lambda x: x[0])
baselines['mut'] = baselines['mutant'].apply(lambda x: x[-1])
# convert aa from three letters to one letter

model_name = 'ESM-IF1'
from IPython import embed; embed()
# amino acid map from three letters to one letter

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



roc_auc_score_dict = {}
roc_auc_score_dict_esmif = {}

mcc_score_dict = {}
mcc_score_dict_esmif = {}

spearman_corr = {}
spearman_corr_esmif = {}

ndcg_score_dict = {}
ndcg_score_dict_esmif = {}

topkrecall_dict = {}
topkrecall_dict_esmif = {}

for position in list(df_gen['pos']):
    glu = df_gen[df_gen['pos'] == position][['GLU']].iloc[0]['GLU']
    gln = df_gen[df_gen['pos'] == position][['GLN']].iloc[0]['GLN']
    lys = df_gen[df_gen['pos'] == position][['LYS']].iloc[0]['LYS']
    glu = 0 if pd.isna(glu) else glu
    gln = 0 if pd.isna(gln) else gln
    lys = 0 if pd.isna(lys) else lys
    
    # If the ground truth label is not available for some reason, set the value to -inf
    
    try:
        glu_gt = df_gt[(df_gt['pos'] == position) & (df_gt['mut'] == 'E')]['DMS_score'].iloc[0]
    except:
        glu_gt = -np.inf
    try:
        gln_gt = df_gt[(df_gt['pos'] == position) & (df_gt['mut'] == 'Q')]['DMS_score'].iloc[0]
    except:
        gln_gt = -np.inf
    try:
        lys_gt = df_gt[(df_gt['pos'] == position) & (df_gt['mut'] == 'K')]['DMS_score'].iloc[0]
    except:
        lys_gt = -np.inf
    
    # If the ground truth binning label is not available for some reason, set the value to 0
    
    try:
        glu_bin_gt = df_gt[(df_gt['pos'] == position) & (df_gt['mut'] == 'E')]['DMS_score_bin'].iloc[0]
    except:
        glu_bin_gt = 0
    try:
        gln_bin_gt = df_gt[(df_gt['pos'] == position) & (df_gt['mut'] == 'Q')]['DMS_score_bin'].iloc[0]
    except:
        gln_bin_gt = 0
    try:
        lys_bin_gt = df_gt[(df_gt['pos'] == position) & (df_gt['mut'] == 'K')]['DMS_score_bin'].iloc[0]
    except:
        lys_bin_gt = 0
    
    
    # For the baselines, if the value is not available, set it to -inf (highly unlikely)
    
    try:
        glu_esif = baselines[(baselines['pos'] == position) & (baselines['mut'] == 'E')][model_name].iloc[0]
    except:
        glu_esif = -np.inf
    
    try:
        gln_esif = baselines[(baselines['pos'] == position) & (baselines['mut'] == 'Q')][model_name].iloc[0]
    except:
        gln_esif = -np.inf
    
    try:
        lys_esif = baselines[(baselines['pos'] == position) & (baselines['mut'] == 'K')][model_name].iloc[0]
    except:
        lys_esif = -np.inf
    
    # If the wild type is one of them, set the predicted value to 0, ground truth bin value to 1 and ground truth value to -inf
    
    if df_gt[(df_gt['pos'] == position)]['aa'].iloc[0] == 'E':
        glu_gt = 0
        glu_esif = 0
        glu_bin_gt = 1
    if df_gt[(df_gt['pos'] == position)]['aa'].iloc[0] == 'Q':
        gln_gt = 0
        gln_esif = 0
        gln_bin_gt = 1
    if df_gt[(df_gt['pos'] == position)]['aa'].iloc[0] == 'K':
        lys_gt = 0
        lys_esif = 0 
        lys_bin_gt = 1
    
    # from IPython import embed; embed()
    # [glu_bin_gt, gln_bin_gt, lys_bin_gt]
    
    try:
        ndcg_score_dict[position] = calc_ndcg(np.array([glu_gt, gln_gt, lys_gt]), np.array([glu, gln, lys]), top=1, quantile=False)
        
    except:
        ndcg_score_dict[position] = np.nan
        
    try:
        ndcg_score_dict_esmif[position] = calc_ndcg(np.array([glu_gt, gln_gt, lys_gt]), np.array([glu_esif, gln_esif, lys_esif]), top=1, quantile=False)
    except:
        ndcg_score_dict_esmif[position] = np.nan
    
    
    ""
    try:
        topkrecall_dict[position] = calc_toprecall(np.array([glu_gt, gln_gt, lys_gt]), np.array([glu, gln, lys]))
    except:
        topkrecall_dict[position] = np.nan
    
    try:
        topkrecall_dict_esmif[position] = calc_toprecall(np.array([glu_gt, gln_gt, lys_gt]), np.array([glu_esif, gln_esif, lys_esif]))
    except:
        topkrecall_dict_esmif[position] = np.nan
    
    try:
        roc_auc_score_dict[position] = roc_auc_score(y_true=[glu_bin_gt, gln_bin_gt, lys_bin_gt], y_score=[glu, gln, lys])
    except:
        roc_auc_score_dict[position] = np.nan
    try:
        roc_auc_score_dict_esmif[position] = roc_auc_score(y_true=[glu_bin_gt, gln_bin_gt, lys_bin_gt], y_score=[glu_esif, gln_esif, lys_esif])
    except:
        roc_auc_score_dict_esmif[position] = np.nan
    try:
        mcc_score_dict[position] = matthews_corrcoef([glu_bin_gt, gln_bin_gt, lys_bin_gt], [glu, gln, lys])
    except:
        mcc_score_dict[position] = np.nan
    
    try:
        mcc_score_dict_esmif[position] = matthews_corrcoef([glu_bin_gt, gln_bin_gt, lys_bin_gt], [glu_esif, gln_esif, lys_esif])
    except:
        mcc_score_dict_esmif[position] = np.nan
        
    # from IPython import embed; embed()
    
    spearman_corr[position] = spearmanr([glu, gln, lys], [glu_gt, gln_gt, lys_gt]).statistic
    spearman_corr_esmif[position] = spearmanr([glu_esif, gln_esif, lys_esif], [glu_gt, gln_gt, lys_gt]).statistic

# convert spearman_corr to a list, with key as index
spearman_corr_list = [] 
spearman_corr_esmif_list = []
ndcg_list = []
ndcg_esmif_list = []
topkrecall_list = []
topkrecall_esmif_list = []
pos_list = list(spearman_corr.keys())
pos_list.sort()

for corr in pos_list:
    spearman_corr_list.append(spearman_corr[corr])  
    spearman_corr_esmif_list.append(spearman_corr_esmif[corr])
    ndcg_list.append(ndcg_score_dict[corr])
    ndcg_esmif_list.append(ndcg_score_dict_esmif[corr])
    topkrecall_list.append(topkrecall_dict[corr])
    topkrecall_esmif_list.append(topkrecall_dict_esmif[corr])

ndcg_mean = np.nanmean(np.array(ndcg_list))
ndcg_esmif_mean = np.nanmean(np.array(ndcg_esmif_list))
topkrecall_mean = np.nanmean(np.array(topkrecall_list))
topkrecall_esmif_mean = np.nanmean(np.array(topkrecall_esmif_list))
from IPython import embed; embed()
# roc_auc_score(y_true=[glu_bin_gt, gln_bin_gt, lys_bin_gt], y_score=[glu_esif, gln_esif, lys_esif])