import os
import sys
import wandb
import torch
from tqdm import tqdm
sys.path.append('.')
sys.path.append('..')

dataset_path = "/home/qcx679/hantang/UAAG2/data/full_graph/benchmarks/5ly1.pt"

naa_data = torch.load(dataset_path)


filtered_data = []

for graph in naa_data:
    if graph.compound_id.split('_')[-2] == "E":
        filtered_data.append(graph)
        
id_list = []        
for graph in filtered_data:
    id_list.append(int(graph.compound_id.split('_')[-3]))
from IPython import embed; embed()