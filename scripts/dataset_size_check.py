import os
import sys
import wandb
import torch
from tqdm import tqdm
sys.path.append('.')
sys.path.append('..')
import json
from uaag.utils import load_data, load_model
import pickle
root_pdb_path = "/home/qcx679/hantang/UAAG2/data/full_graph/data_2"
    
# root_pdb_path_test = "/home/qcx679/hantang/UAAG2/data/full_graph/data"
# root_pdb_path = root_pdb_path_test

pdb_list = os.listdir(root_pdb_path)
pdb_list = [os.path.join(root_pdb_path, pdb) for pdb in pdb_list]

root_naa_path = "/home/qcx679/hantang/UAAG2/data/full_graph/naa"
naa_list = os.listdir(root_naa_path)
naa_list = [os.path.join(root_naa_path, naa) for naa in naa_list]

pdbbind_path = "/home/qcx679/hantang/UAAG2/data/full_graph/pdbbind/pdbbind_data.pt"

naa_path = "/home/qcx679/hantang/UAAG2/data/full_graph/benchmarks/2roc.pt"
# combine three parts of the data
data = []
# with open('data/aa_graph.json', 'rb') as json_file:
#     AA_GRAPH_DICT = json.load(json_file)
#     json_file.close()

with open("/home/qcx679/hantang/UAAG2/data/full_graph/statistic.pkl", 'rb') as f:
    data_info = pickle.load(f)
    
#     print(len(data_file))
from IPython import embed; embed()
# data_file = torch.load(naa_path)
# from IPython import embed; embed()
# for pdb in pdb_list:
#     data.append(pdb)

# for naa in naa_list:
#     data.append(naa)

# data.append(pdbbind_path)

# for file in tqdm(data):
#     print(f"Loading {file} \n")
#     data_file = torch.load(file)

    # data.extend(data_file)