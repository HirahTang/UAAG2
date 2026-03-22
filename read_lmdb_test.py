
from rdkit import Chem

import torch
import torch.distributed as dist
        
from torch.utils.data import Subset, DistributedSampler
from torch_geometric.data import Dataset, DataLoader, Data

from torch_geometric.data.lightning import LightningDataset

# from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
from uaag.data.abstract_dataset import (
    AbstractDataModule,
)

from torch_geometric.utils import sort_edge_index

from uaag.utils import visualize_mol
import lmdb
import pytorch_lightning as pl
import pickle

data_path = "/scratch/project_465002574/unaagi_whole_v1.lmdb"
idx = 0

env = lmdb.open(
    data_path,
    readonly=True,
    lock=False,
    subdir=False,
    readahead=False,
    meminit=False,
)
    
with env.begin() as txn:
    key = f"{idx:08}".encode("ascii")
    byteflow = txn.get(key)
graph_data = pickle.loads(byteflow)
assert isinstance(graph_data, Data), f"Expected torch_geometric.data.Data, got {type(graph)}"
env.close()

print(graph_data)
print(graph_data.id)
print(graph_data.ids)