import lmdb
import pickle
import torch
from tqdm import tqdm

test_data_path = "/home/qcx679/hantang/UAAG2/data/full_graph/full_graph_debug.pt"

data = torch.load(test_data_path)
import os
from IPython import embed


# Load data
# data = torch.load("your_data.pt")  # data should be a list or dict of samples
# assert isinstance(data, (list, tuple)), "Expected a list-like dataset"

# LMDB setup
lmdb_path = "/datasets/biochem/unaagi/debug_test_2.lmdb"
embed()
# if os.path.exists(lmdb_path):
#     print(f"Removing existing LMDB at {lmdb_path}")
#     import shutil
#     shutil.rmtree(lmdb_path)

env = lmdb.open(
    lmdb_path,
    map_size=1 << 40,       # 1 TB
    subdir=False,
    readonly=False,
    meminit=False,
    map_async=True,
)

with env.begin(write=True) as txn:
    for idx, sample in tqdm(enumerate(data), total=len(data)):
        key = f"{idx:08}".encode("ascii")
        value = pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL)
        txn.put(key, value)
    
    # Save length info
    txn.put(b"__len__", pickle.dumps(len(data)))

# Flush to disk
env.sync()
env.close()
