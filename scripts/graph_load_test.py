import sys

sys.path.append('.')
sys.path.append('..')
import numpy as np
from torch_geometric.data import Batch
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from uaag.e3moldiffusion.coordsatomsbonds import DenoisingEdgeNetwork
from IPython import embed
from tqdm import tqdm
from uaag.data.uaag_dataset import UAAG2DataModule, UAAG2Dataset, Dataset_Info
# path = "/home/qcx679/hantang/UAAG2/data/full_graph/data/uaag_aa_eqgat_1_24.pt"

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--data_path", type=str, default="/home/qcx679/hantang/UAAG2/data/full_graph/data/uaag_aa_eqgat_5_11.pt")
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--num_workers", type=int, default=0)
args = parser.parse_args()


path = [
    "/home/qcx679/hantang/UAAG2/data/full_graph/data/uaag_aa_eqgat_5_11.pt",
    "/home/qcx679/hantang/UAAG2/data/full_graph/data/uaag_aa_eqgat_0_9.pt",
    # "/home/qcx679/hantang/UAAG2/data/full_graph/data/uaag_aa_eqgat_1_37.pt",
    # "/home/qcx679/hantang/UAAG2/data/full_graph/data/uaag_aa_eqgat_1_43.pt",
    # "/home/qcx679/hantang/UAAG2/data/full_graph/data/uaag_aa_eqgat_1_44.pt",
    # "/home/qcx679/hantang/UAAG2/data/full_graph/data/uaag_aa_eqgat_1_42.pt",
    "/home/qcx679/hantang/UAAG2/data/full_graph/naa/AACLBR_data.pt",
    "/home/qcx679/hantang/UAAG2/data/full_graph/naa/L_sidechain_data.pt",
    "/home/qcx679/hantang/UAAG2/data/full_graph/pdbbind/pdbbind_data.pt"
    ]
# # load the data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# data = torch.load(path)
# embed()
model = DenoisingEdgeNetwork(
    num_atom_features=22,
    num_bond_types=5,
    hn_dim=(256, 64),
    cutoff_local=7.0,
    num_layers=7,
    latent_dim=None,
    use_cross_product=False,
    fully_connected=True,
    local_global_model=False,
    recompute_radius_graph=False,
    recompute_edge_attributes=True,
    vector_aggr='mean',
    edge_mp=False,
    use_pos_norm=False,

).to(device)
def split_list(lst, n_splits):
    # Calculate the approximate size of each sublist
    avg_size = len(lst) // n_splits
    remainder = len(lst) % n_splits

    sublists = []
    start = 0
    for i in range(n_splits):
        # Add an extra item to some sublists to account for the remainder
        end = start + avg_size + (1 if i < remainder else 0)
        sublists.append(lst[start:end])
        start = end

    return sublists


# data_care_about = data[20]
# atom_feature = data_care_about.x

Graph_dataset = UAAG2Dataset(path)
datamodule = UAAG2DataModule(args, Graph_dataset)
datamodule.setup(stage='fit')
# loader = DataLoader(Graph_dataset, batch_size=8, shuffle=True)

loader = datamodule.train_dataloader()
iterator = iter(loader)
for i in tqdm(range(1000)):
    try:
        batch = next(iterator)
    except:
        print("Something goes wrong")
        embed()
    batch = batch.to(device)
    t = torch.randint(
                low=1,
                high=1000 + 1,
                size=(32,),
                dtype=torch.long,
                device=batch.x.device,
            )
    temb = t.float() / 1000.0
    temb = temb.clamp(min=1e-3)
    temb = temb.unsqueeze(dim=1)


    atom_features = F.one_hot(batch.x.squeeze().long(), num_classes=8).float()
    charges = F.one_hot(batch.charges.squeeze().long(), num_classes=3).float()
    # charges = batch.charges.unsqueeze(dim=-1).float()
    ring_feat = batch.is_in_ring.unsqueeze(dim=-1).float()
    aromatic_feat = batch.is_aromatic.unsqueeze(dim=-1).float()
    hybridization = F.one_hot(batch.hybridization.squeeze().long(), num_classes=4).float()
    degree = F.one_hot(batch.degree.squeeze().long(), num_classes=5).float()
    print("Concatenating things")
    # embed()
    atom_feats_in_perturbed = torch.cat(
                [
                    atom_features,
                    charges,
                    ring_feat,
                    aromatic_feat,
                    hybridization,
                    degree,
                ],
                dim=-1,
            )
    pos = batch.pos
    edge_index = batch.edge_index
    edge_attr = batch.edge_attr
    edge_attr = F.one_hot(edge_attr.squeeze().long(), num_classes=5).float()
    data_batch = batch.batch
    # edge_batch = batch.edge_batch
    edge_batch = data_batch[edge_index[0]]
    edge_mask = batch.edge_ligand
    pocket_mask = (1 - batch.is_ligand + batch.is_backbone).long()
    batch_ligand = batch.batch[pocket_mask==0]

    embed()

    out = model(
        x=atom_feats_in_perturbed,
        t=temb,
        pos=pos,
        edge_index_local=None,
        edge_index_global=edge_index,
        edge_attr_global=edge_attr,
    #    edge_index_global_lig=edge_index_global_lig,
        batch=data_batch,
        batch_edge_global=edge_batch,
        context=None,
        pocket_mask=pocket_mask.unsqueeze(1),
        edge_mask=edge_mask.long(),
        batch_lig=batch_ligand,
    )
    torch.cuda.empty_cache()
    # Data shape 
    # Data(x=[199], edge_index=[2, 9784], edge_attr=[9784], pos=[199, 3], edge_ligand=[9784], 
        # charges=[199], degree=[199], is_aromatic=[199], is_in_ring=[199], hybridization=[199], 
        # is_ligand=[199], is_backbone=[199], id=[199], ids=[199], compound_id='1HRT_tidy_LEU_53_H_74')

    # embed()