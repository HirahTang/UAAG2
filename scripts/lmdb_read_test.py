import lmdb
import pickle
from torch.utils.data import Dataset
from torch_geometric.data import Data

class LMDBGraphDataset(Dataset):
    def __init__(self, lmdb_path):
        self.env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            subdir=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin() as txn:
            self.length = pickle.loads(txn.get(b"__len__"))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            key = f"{idx:08}".encode("ascii")
            byteflow = txn.get(key)
        graph = pickle.loads(byteflow)
        assert isinstance(graph, Data), f"Expected torch_geometric.data.Data, got {type(graph)}"
        return graph


lmdb_path = "/datasets/biochem/unaagi/debug_test.lmdb"
dataset = LMDBGraphDataset(lmdb_path)
print(f"Dataset length: {len(dataset)}")
metadata_path = "/datasets/biochem/unaagi/unaagi_whole_v1.metadata.pkl"
with open(metadata_path, "rb") as f:
    metadata = pickle.load(f)
print(f"Metadata length: {len(metadata)}")
from IPython import embed
embed()