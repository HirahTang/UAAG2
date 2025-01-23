import argparse
import math
from os.path import dirname, exists, join
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from pytorch_lightning.utilities import rank_zero_warn
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from torch_geometric.utils import dense_to_sparse, sort_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sort_edge_index import sort_edge_index
from torch_geometric.utils.subgraph import subgraph
from torch_scatter import scatter_add, scatter_mean
from tqdm import tqdm
# from torch_sparse import coalesces

def load_data(hparams, data_path: list, pdb_list: list) -> Data:
    data = []
    pdb_list = []
    for file in tqdm(data_path):
        print(f"Loading {file} \n")
        data_file = torch.load(file)
        data.extend(data_file)
    
    for file in tqdm(pdb_list):
        print(f"Loading {file} \n")
        pdb_file = torch.load(file)
        pdb_list.extend(pdb_file)
    
    # randomly split data into train, val, test
    np.random.seed(hparams.seed)
    np.random.shuffle(data)
    np.random.shuffle(pdb_list)
    num_data = len(data)
    num_train = math.floor(num_data * hparams.train_size)
    num_val = math.floor(num_data * hparams.val_size)
    num_test = hparams.test_size
    
    train_data = data[:num_train]
    val_data = data[num_train:]
    test_data = pdb_list[:num_test]
    
    # test_data = data[:num_test]
    # val_data = data[num_test:num_test + num_val]
    # train_data = data[num_test + num_val:]
    
    return train_data, val_data, test_data

def create_model(hparams, num_atom_features, num_bond_classes):
    from e3moldiffusion.coordsatomsbonds import DenoisingEdgeNetwork

    model = DenoisingEdgeNetwork(
        hn_dim=(hparams["sdim"], hparams["vdim"]),
        num_layers=hparams["num_layers"],
        latent_dim=None,
        use_cross_product=hparams["use_cross_product"],
        num_atom_features=num_atom_features,
        num_bond_types=num_bond_classes,
        edge_dim=hparams["edim"],
        cutoff_local=hparams["cutoff_local"],
        vector_aggr=hparams["vector_aggr"],
        fully_connected=hparams["fully_connected"],
        local_global_model=hparams["local_global_model"],
        recompute_edge_attributes=True,
        recompute_radius_graph=False,
        edge_mp=hparams["edge_mp"],
        context_mapping=hparams["context_mapping"],
        num_context_features=hparams["num_context_features"],
        bond_prediction=hparams["bond_prediction"],
        property_prediction=hparams["property_prediction"],
        coords_param=hparams["continuous_param"],
        use_pos_norm=hparams["use_pos_norm"],
    )
    return model

def load_model(filepath, num_atom_features, num_bond_classes, device="cpu", **kwargs):
    import re

    ckpt = torch.load(filepath, map_location="cpu")
    args = ckpt["hyper_parameters"]

    args["use_pos_norm"] = True

    model = create_model(args, num_atom_features, num_bond_classes)

    state_dict = ckpt["state_dict"]
    state_dict = {
        re.sub(r"^model\.", "", k): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model")
    }
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if not any(x in k for x in ["prior", "sde", "cat"])
    }
    model.load_state_dict(state_dict)
    return model.to(device)

def zero_mean(x, batch, dim_size: int, dim=0):
    out = x - scatter_mean(x, index=batch, dim=dim, dim_size=dim_size)[batch]
    return out

class Statistics:
    def __init__(
        self,
        num_nodes,
        atom_types,
        bond_types,
        charge_types,
        valencies,
        bond_lengths,
        bond_angles,
        dihedrals=None,
        is_in_ring=None,
        is_aromatic=None,
        hybridization=None,
        degree=None,
        force_norms=None,
    ):
        self.num_nodes = num_nodes
        # print("NUM NODES IN STATISTICS", num_nodes)
        self.atom_types = atom_types
        self.bond_types = bond_types
        self.charge_types = charge_types
        self.valencies = valencies
        self.bond_lengths = bond_lengths
        self.bond_angles = bond_angles
        self.dihedrals = dihedrals
        self.is_in_ring = is_in_ring
        self.is_aromatic = is_aromatic
        self.hybridization = hybridization
        self.degree = degree
        self.force_norms = force_norms