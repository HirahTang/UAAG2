from operator import is_
from os.path import join
from typing import Optional
import sys
import numpy as np
sys.path.append('.')
sys.path.append('..')

import torch

from torch.utils.data import Subset
from torch_geometric.data import Dataset, DataLoader, Data

from torch_geometric.data.lightning import LightningDataset

# from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
from uaag.data.abstract_dataset import (
    AbstractDataModule,
)
import pytorch_lightning as pl
import pickle
# from experiments.data.geom.geom_dataset_adaptive import GeomDrugsDataset
# from experiments.data.utils import train_subset

class UAAG2Dataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        data,
    ):
        super(UAAG2Dataset, self).__init__()
        # self.statistics = Statistic()

        self.data = data
        # self.load_dataset()
        self.charge_emb = {
            -1: 0,
            0: 1,
            1: 2,
        }
    # def load_dataset(self):
    #     for file in tqdm(self.root):
    #         print(f"Loading {file} \n")
    #         data_file = torch.load(file)
    #         self.data.extend(data_file)
    #     self.data
        
    def __len__(self):
        return len(self.data)
    
    def pocket_centering(self, batch):
        # graph_data = self.data[idx]
        pos = batch.pos
        if batch.is_ligand.sum() == len(batch.is_ligand):
            pocket_mean = pos.mean(dim=0)
        else:
            pocket_pos = batch.pos[batch.is_ligand==0]
            pocket_mean = pocket_pos.mean(dim=0)
        pos = pos - pocket_mean
        batch.pos = pos
        return batch
    
    def __getitem__(self, idx):
        
        # TODO zero center the positions by the mean of the pocket atoms
        
        graph_data = self.data[idx]
        graph_data = self.pocket_centering(graph_data)
        #. aligning dtype
        
        # if graph_data.edge_ligand
        
        # check if graph_data.edge_ligand exists
        if not hasattr(graph_data, 'compound_id'):
            graph_data.compound_id = graph_data.componud_id
        if not hasattr(graph_data, 'edge_ligand'):
            graph_data.edge_ligand = torch.ones(graph_data.edge_attr.size(0))
        # if not hasattr(graph_data, 'compound_id'):
        #     graph_data.compound_id = graph_data.componud_id
        if not hasattr(graph_data, 'id'):
            graph_data.id = graph_data.compound_id
        graph_data.x = graph_data.x.float()
        graph_data.pos = graph_data.pos.float()
        graph_data.edge_attr = graph_data.edge_attr.float()
        graph_data.edge_index = graph_data.edge_index.long()
        # from IPython import embed; embed()
        
        charges_np = graph_data.charges.numpy()
        mapped_np = np.vectorize(self.charge_emb.get)(charges_np)
        charges = torch.from_numpy(mapped_np)
        
        
        # graph_data.charges = graph_data.charges.long()
        # graph_data.charges = torch.tensor(self.charge_emb[i] for i in graph_data.charges).float()
        # map the value of charges by {-1: 0, 0: 1, 1: 2}
        
        
        
        graph_data.degree = graph_data.degree.float()
        graph_data.is_aromatic = graph_data.is_aromatic.float()
        graph_data.is_in_ring = graph_data.is_in_ring.float()
        graph_data.hybridization = graph_data.hybridization.float()
        graph_data.is_backbone = graph_data.is_backbone.float()
        graph_data.is_ligand = graph_data.is_ligand.float()
        
        batch_graph_data = Data(
            x=graph_data.x,
            pos=graph_data.pos,
            edge_index=graph_data.edge_index,
            edge_attr=graph_data.edge_attr,
            edge_ligand=torch.tensor(graph_data.edge_ligand).float(),
            charges=charges,
            degree=graph_data.degree,
            is_aromatic=graph_data.is_aromatic,
            is_in_ring=graph_data.is_in_ring,
            hybridization=graph_data.hybridization,
            is_backbone=graph_data.is_backbone,
            is_ligand=graph_data.is_ligand,
            ligand_size=torch.tensor(graph_data.is_ligand.sum() - graph_data.is_backbone.sum()).long(),
            id=graph_data.compound_id,
        )
        
        return batch_graph_data


class UAAG2Dataset_sampling(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        dataset_info,
        fix_size=False,
        sample_size=10,
    ):
        super(UAAG2Dataset_sampling, self).__init__()
        # self.statistics = Statistic()

        self.fix_size = fix_size
        self.sample_size = sample_size
        
        self.data = data
        # self.load_dataset()
        self.charge_emb = {
            -1: 0,
            0: 1,
            1: 2,
        }
        self.dataset_info = dataset_info
        
        atom_types_distribution = dataset_info.atom_types.float()
        bond_types_distribution = dataset_info.bond_types.float()
        charge_types_distribution = dataset_info.charge_types.float()
        is_aromatic_distribution = dataset_info.is_aromatic.float()
        is_ring_distribution = dataset_info.is_ring.float()
        hybridization_distribution = dataset_info.hybridization.float()
        degree_distribution = dataset_info.degree.float()
        
        
        self.register_buffer("atoms_prior", atom_types_distribution.clone())
        self.register_buffer("bonds_prior", bond_types_distribution.clone())
        self.register_buffer("charges_prior", charge_types_distribution.clone())
        self.register_buffer("is_aromatic_prior", is_aromatic_distribution.clone())
        self.register_buffer("is_in_ring_prior", is_ring_distribution.clone())
        self.register_buffer("hybridization_prior", hybridization_distribution.clone())
        self.register_buffer("degree_prior", degree_distribution.clone())
        
    def __len__(self):
        return len(self.data)
    
    def pocket_centering(self, batch):
        # graph_data = self.data[idx]
        pos = batch.pos
        if batch.is_ligand.sum() == len(batch.is_ligand):
            pocket_mean = pos.mean(dim=0)
        else:
            pocket_pos = batch.pos[batch.is_ligand==0]
            pocket_mean = pocket_pos.mean(dim=0)
        pos = pos - pocket_mean
        batch.pos = pos
        return batch
    
    def __getitem__(self, idx):
        graph_data = self.data[idx]
        graph_data = self.pocket_centering(graph_data)
        #. aligning dtype
        
        # if graph_data.edge_ligand
        
        # check if graph_data.edge_ligand exists
        # if not hasattr(graph_data, 'componud_id'):
        #     graph_data.componud_id = graph_data.compound_id
        if not hasattr(graph_data, 'edge_ligand'):
            graph_data.edge_ligand = torch.ones(graph_data.edge_attr.size(0))
        if not hasattr(graph_data, 'compound_id'):
            graph_data.compound_id = graph_data.componud_id
        if not hasattr(graph_data, 'id'):
            graph_data.id = graph_data.compound_id    
        
        graph_data.x = graph_data.x.float()
        graph_data.pos = graph_data.pos.float()
        graph_data.edge_attr = graph_data.edge_attr.float()
        graph_data.edge_index = graph_data.edge_index.long()
        # from IPython import embed; embed()
        
        charges_np = graph_data.charges.numpy()
        mapped_np = np.vectorize(self.charge_emb.get)(charges_np)
        charges = torch.from_numpy(mapped_np)
        
        
        # graph_data.charges = graph_data.charges.long()
        # graph_data.charges = torch.tensor(self.charge_emb[i] for i in graph_data.charges).float()
        # map the value of charges by {-1: 0, 0: 1, 1: 2}
        
        
        
        graph_data.degree = graph_data.degree.float()
        graph_data.is_aromatic = graph_data.is_aromatic.float()
        graph_data.is_in_ring = graph_data.is_in_ring.float()
        graph_data.hybridization = graph_data.hybridization.float()
        graph_data.is_backbone = graph_data.is_backbone.float()
        graph_data.is_ligand = graph_data.is_ligand.float()
        
        batch_graph_data = Data(
            x=graph_data.x,
            pos=graph_data.pos,
            edge_index=graph_data.edge_index,
            edge_attr=graph_data.edge_attr,
            edge_ligand = graph_data.edge_ligand.float(),
            charges=charges,
            degree=graph_data.degree,
            is_aromatic=graph_data.is_aromatic,
            is_in_ring=graph_data.is_in_ring,
            hybridization=graph_data.hybridization,
            is_backbone=graph_data.is_backbone,
            is_ligand=graph_data.is_ligand,
            id=graph_data.compound_id,
        )
        
        # convert batch_graph_data to remove the non-pocket information
        
        is_pocket = 1 - batch_graph_data.is_ligand + batch_graph_data.is_backbone
        is_reconstruct = 1 - is_pocket
        ids = torch.tensor(range(len(batch_graph_data.x)))
        new_ids = ids[is_pocket==1]
        
        map_ids = {int(new_ids[i]): i for i in range(len(new_ids))}
        map_keys = torch.tensor(list(map_ids.keys()))
        map_values = torch.tensor(list(map_ids.values()))
        
        start_in_index = torch.isin(batch_graph_data.edge_index[0], new_ids)
        end_in_index = torch.isin(batch_graph_data.edge_index[1], new_ids)
        edge_mask = start_in_index & end_in_index
        edge_index = batch_graph_data.edge_index[:, edge_mask]
        
        new_edge_index = torch.stack([
            map_values[(edge_index[0].unsqueeze(-1)==map_keys).nonzero(as_tuple=True)[1]],
            map_values[(edge_index[1].unsqueeze(-1)==map_keys).nonzero(as_tuple=True)[1]],
        ])
        new_edge_attr = batch_graph_data.edge_attr[edge_mask]
        new_edge_ligand = batch_graph_data.edge_ligand[edge_mask]
        new_x = batch_graph_data.x[is_pocket==1]
        new_pos = batch_graph_data.pos[is_pocket==1]
        new_charges = batch_graph_data.charges[is_pocket==1]
        new_degree = batch_graph_data.degree[is_pocket==1]
        new_is_aromatic = batch_graph_data.is_aromatic[is_pocket==1]
        new_is_in_ring = batch_graph_data.is_in_ring[is_pocket==1]
        new_hybridization = batch_graph_data.hybridization[is_pocket==1]
        
        # Adding prior noise to the graph
        
        reconstruct_size = is_reconstruct.sum() if not self.fix_size else self.sample_size
        
        sampled_x = torch.multinomial(self.atoms_prior, reconstruct_size, replacement=True)
        sampled_charge = torch.multinomial(self.charges_prior, reconstruct_size, replacement=True)
        sampled_degree = torch.multinomial(self.degree_prior, reconstruct_size, replacement=True)
        sampled_aromatic = torch.multinomial(self.is_aromatic_prior, reconstruct_size, replacement=True)
        sampled_ring = torch.multinomial(self.is_in_ring_prior, reconstruct_size, replacement=True)
        sampled_hybrid = torch.multinomial(self.hybridization_prior, reconstruct_size, replacement=True)
        sampled_pos = torch.randn(reconstruct_size, 3)
        
        
        graph_ligand_removed = Data(
            x=new_x.float(),
            pos=new_pos.float(),
            edge_index=new_edge_index.long(),
            edge_attr=new_edge_attr.float(),
            edge_ligand = new_edge_ligand.float(),
            charges=new_charges.float(),
            degree=new_degree.float(),
            is_aromatic=new_is_aromatic.float(),
            is_in_ring=new_is_in_ring.float(),
            hybridization=new_hybridization.float(),
            is_backbone=graph_data.is_backbone[is_pocket==1].float(),
            is_ligand=graph_data.is_ligand[is_pocket==1].float(),
            ligand_size=torch.tensor(len(batch_graph_data.x[is_pocket==0])).long(),
            id=graph_data.compound_id,
        )

        return graph_ligand_removed
    
class UAAG2DataModule(pl.LightningDataModule):
    def __init__(self, cfg, train_data, val_data, test_data, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.cfg = cfg
        # split into train, val, test with test & valid consisting 2000 samples each
        # self._log_hyperparams = True
        self.pin_memory = True
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        # self.setup(stage='fit')
        
    # def prepare_data(self):
    #     pass
    
    # def load_dataset(self):
    #     for file in tqdm(self.root):
    #         print(f"Loading {file} \n")
    #         data_file = torch.load(file)
    #         self.data.extend(data_file)
        # self.data
    
    
    def setup(self, stage):
        
        # TODO
        # Construct the dictionary & distributions for the dataset
        
        full_length = len(self.train_data) + len(self.val_data) + len(self.test_data)
        

    def train_dataloader(self, shuffle=True):
        dataloader = DataLoader(
            dataset=self.train_data,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
            persistent_workers=False,
        )
        return dataloader
    
    def val_dataloader(self):
        dataloader = DataLoader(
            dataset=self.val_data,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=False,
        )
        return dataloader
    
    def test_dataloader(self):
        dataloader = DataLoader(
            dataset=self.test_data,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=False,
        )
        return dataloader
    
        
class Dataset_Info:
    def __init__(self, info_path):
        self.info_path = info_path
        self.process()
        self.get_decoder()
    def process(self):
        # read from the info_path, a pickle file
        
        with open(self.info_path, 'rb') as f:
            data_info = pickle.load(f)

        
        self.atom_types = []
        
        sum_x = []
        for k in data_info['x'].keys():
            sum_x.append(data_info['x'][k])
        sum_x = torch.tensor(sum_x).sum()
        for key in data_info['x']:
            self.atom_types.append(data_info['x'][key] / sum_x)
            # print(self.atom_types[int(key)], key, data_info['x'][key])
        self.atom_types = torch.tensor(self.atom_types)
        
        self.bond_types = []
        sum_edge = []
        for k in data_info['edge'].keys():
            sum_edge.append(data_info['edge'][k])
        sum_edge = torch.tensor(sum_edge).sum()
        for key in data_info['edge']:
            self.bond_types.append(data_info['edge'][key] / sum_edge)
            # print(self.bond_types[int(key)], key, data_info['edge_index'][key])
        self.bond_types = torch.tensor(self.bond_types)
        
        self.charge_types = []
        sum_charge = []
        for k in data_info['charge'].keys():
            sum_charge.append(data_info['charge'][k])
        sum_charge = torch.tensor(sum_charge).sum()
        for key in data_info['charge']:
            self.charge_types.append(data_info['charge'][key] / sum_charge)
        self.charge_types = torch.tensor(self.charge_types)
            # print(self.charge_types[int(key)], key, data_info['charge'][key])
        
        self.is_aromatic = []
        sum_aromatic = []
        for k in data_info['aro'].keys():
            sum_aromatic.append(data_info['aro'][k])
        sum_aromatic = torch.tensor(sum_aromatic).sum()
        for key in data_info['aro']:
            self.is_aromatic.append(data_info['aro'][key] / sum_aromatic)
        self.is_aromatic = torch.tensor(self.is_aromatic)
            # print(self.is_aromatic[int(key)], key, data_info['aromatic'][key])
            
        self.is_ring = []
        sum_ring = []
        for k in data_info['ring'].keys():
            sum_ring.append(data_info['ring'][k])
        sum_ring = torch.tensor(sum_ring).sum()
        for key in data_info['ring']:
            self.is_ring.append(data_info['ring'][key] / sum_ring)
        self.is_ring = torch.tensor(self.is_ring)
            # print(self.is_ring[int(key)], key, data_info['ring'][key])
            
        self.hybridization = []
        sum_hybrid = []
        for k in data_info['hybrid'].keys():
            sum_hybrid.append(data_info['hybrid'][k])
        sum_hybrid = torch.tensor(sum_hybrid).sum()
        for key in data_info['hybrid']:
            self.hybridization.append(data_info['hybrid'][key] / sum_hybrid)
            # print(self.hybridization[int(key)], key, data_info['hybrid'][key])
        self.hybridization = torch.tensor(self.hybridization)
        
        self.degree = []
        sum_degree = []
        for k in data_info['degree'].keys():
            sum_degree.append(data_info['degree'][k])
        sum_degree = torch.tensor(sum_degree).sum()
        for key in data_info['degree']:
            self.degree.append(data_info['degree'][key] / sum_degree)
        self.degree = torch.tensor(self.degree)
            # print(self.degree[int(key)], key, data_info['degree'][key])
            
    def get_decoder(self):
        atom_encoder = {
            "C": 0,
            "N": 1,
            "O": 2,
            "S": 3,
            "P": 4,
            "Cl": 5,
            "F": 6,
            "Br": 7,
        }
        atom_decoder  = {v: k for k, v in atom_encoder.items()}
        self.atom_decoder = atom_decoder
            
    
        
        
        
        