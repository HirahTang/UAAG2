import sys
from turtle import st

sys.path.append('.')
sys.path.append('..')

import numpy as np
import torch
from uaag.data.uaag_dataset import UAAG2Dataset, UAAG2Dataset_sampling, UAAG2DataModule
import argparse
from uaag.equivariant_diffusion import Trainer
from uaag.utils import load_data, load_model
data_path = [
    '/home/qcx679/hantang/UAAG2/data/full_graph/data/uaag_aa_eqgat_13_24.pt',
    '/home/qcx679/hantang/UAAG2/data/full_graph/data/uaag_aa_eqgat_13_34.pt',
    '/home/qcx679/hantang/UAAG2/data/full_graph/data/uaag_aa_eqgat_9_27.pt',
    # '/home/qcx679/hantang/UAAG2/data/full_graph/pdbbind/pdbbind_data.pt',
]

# dataset = UAAG2Dataset(data_path)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument("--num_degree", type=int, default=5)
parser.add_argument("--load_ckpt_from_pretrained", default=None)
parser.add_argument("--sdim", default=256, type=int)
parser.add_argument("--vdim", default=64, type=int)
parser.add_argument("--use-cross-product", default=False, action="store_true")
parser.add_argument("--edim", default=32, type=int)
parser.add_argument("--cutoff-local", default=7.0, type=float)
parser.add_argument("--vector-aggr", default="mean", type=str)
parser.add_argument("--num-layers", default=7, type=int)
parser.add_argument("--fully-connected", default=True, action="store_true")
parser.add_argument("--local-global-model", default=False, action="store_true")
parser.add_argument("--edge-mp", default=False, action="store_true")
parser.add_argument("--context-mapping", default=False, action="store_true")
parser.add_argument("--num-context-features", default=0, type=int)
parser.add_argument("--bond-prediction", default=False, action="store_true")
parser.add_argument("--property-prediction", default=False, action="store_true")
parser.add_argument(
"--continuous-param", default="data", type=str, choices=["data", "noise"]
)

parser.add_argument("--beta-min", default=1e-4, type=float)
parser.add_argument("--beta-max", default=2e-2, type=float)
parser.add_argument("--timesteps", default=500, type=int)
parser.add_argument(
"--noise-scheduler",
default="cosine",
choices=["linear", "cosine", "quad", "sigmoid", "adaptive", "linear-time"],
)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--num-hybridization", default=4, type=int)
# Model hyperparameters:

# Diffusion hyperparameters:



hparams = parser.parse_args()
train_data, val_data, test_data = load_data(hparams, data_path)
train_data = UAAG2Dataset(train_data)
val_data = UAAG2Dataset(val_data)
test_data = UAAG2Dataset_sampling(test_data)
datamodule = UAAG2DataModule(hparams, train_data, val_data, test_data)
datamodule.setup(stage="fit")
loader = datamodule.train_dataloader(shuffle=False)
iterator = iter(loader)

pdbcase = train_data[2]
pdbbind_case = train_data[-1]

for key in pdbcase.keys():
    try:
        print(key, pdbcase[key].dtype, pdbbind_case[key].dtype)
    except:
        print(key, pdbcase[key], pdbbind_case[key])


ids = torch.tensor(range(len(pdbcase.x)))
is_pocket = 1 - pdbcase.is_ligand + pdbcase.is_backbone

new_ids = ids[is_pocket == 1]

new_id_mapping = {id.item(): i for i, id in enumerate(new_ids)}
map_keys = torch.tensor(list(new_id_mapping.keys()))
map_values = torch.tensor(list(new_id_mapping.values()))

start_in_index = torch.isin(pdbcase.edge_index[0], new_ids)
end_in_index = torch.isin(pdbcase.edge_index[1], new_ids)
edge_mask = start_in_index & end_in_index
edge_index = pdbcase.edge_index[:, edge_mask]

new_edge_index = torch.stack([
    map_values[(edge_index[0].unsqueeze(-1)==map_keys).nonzero(as_tuple=True)[1]],
    map_values[(edge_index[1].unsqueeze(-1)==map_keys).nonzero(as_tuple=True)[1]],
])

new_edge_attr = pdbcase.edge_attr[edge_mask]

from IPython import embed; embed()
# model = Trainer(
#     hparams=hparams.__dict__,
#     dataset_info=dataset.info,
# )

