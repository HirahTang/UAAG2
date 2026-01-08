import lmdb
import pickle
import torch
from tqdm import tqdm
import os
import glob


root_pdb_path = "/home/qcx679/hantang/UAAG2/data/full_graph/data_2"
    
# root_pdb_path_test = "/home/qcx679/hantang/UAAG2/data/full_graph/data"
# root_pdb_path = root_pdb_path_test

pdb_list = os.listdir(root_pdb_path)
pdb_list = [os.path.join(root_pdb_path, pdb) for pdb in pdb_list]

root_naa_path = "/home/qcx679/hantang/UAAG2/data/full_graph/naa"
naa_list = os.listdir(root_naa_path)
naa_list = [os.path.join(root_naa_path, naa) for naa in naa_list]

pdbbind_path = "/home/qcx679/hantang/UAAG2/data/full_graph/pdbbind/pdbbind_data.pt"

    
data = []
    
for pdb in pdb_list:
    data.append(pdb)
# data = ['/home/qcx679/hantang/UAAG2/data/full_graph/full_graph_debug.pt']
for naa in naa_list:
    data.append(naa)

data.append(pdbbind_path)

lmdb_path = "/datasets/biochem/unaagi/unaagi_whole_v1.lmdb"

env = lmdb.open(
    lmdb_path,
    map_size=1 << 40,
    subdir=False,
    readonly=False,
    meminit=False,
    map_async=True,
)

total_count = 0
metadata = {}
txn = env.begin(write=True)

for pt_file in data:
    print(f"Loading {pt_file}...", flush=True)
    source_name = os.path.basename(pt_file)
    print(f"Source name: {source_name}")
    data_list = torch.load(pt_file, map_location="cpu")  # Should be a list of torch_geometric.data.Data
    print(f"Loaded {len(data_list)} samples from {source_name}", flush=True)
    for i, sample in tqdm(enumerate(data_list)):
        key = f"{total_count:08}".encode("ascii")
        sample.source_name = source_name
        metadata[key] = source_name
        value = pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL)
        txn.put(key, value)
        if (i+1) % 10000 == 0:
            txn.commit()
            txn = env.begin(write=True)
        total_count += 1

# Save length
txn.put(b"__len__", pickle.dumps(total_count))
txn.commit()
env.sync()
env.close()
with open("/datasets/biochem/unaagi/unaagi_whole_v1.metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)
print(f"All done. {total_count} graphs saved to {lmdb_path}")