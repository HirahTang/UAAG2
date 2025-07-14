from operator import is_
from os.path import join
import os
from typing import Optional
import sys
import numpy as np
sys.path.append('.')
sys.path.append('..')

from rdkit import Chem

import torch

from torch.utils.data import Subset
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
# from experiments.data.geom.geom_dataset_adaptive import GeomDrugsDataset
# from experiments.data.utils import train_subset

class UAAG2Dataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        data_path,
        mask_rate: float = 0,
        pocket_noise: bool = False,
        noise_scale: float = 0.1,
        params=None,
    ):
        super(UAAG2Dataset, self).__init__()
        # self.statistics = Statistic()
        self.params = params
        self.pocket_noise = pocket_noise
        if self.pocket_noise:
            self.noise_scale = noise_scale
        
        self.env = lmdb.open(
            data_path,
            readonly=True,
            lock=False,
            subdir=False,
            readahead=False,
            meminit=False,
        )
        
        with self.env.begin() as txn:
            self.length = pickle.loads(txn.get(b"__len__"))
        # self.load_dataset()
        self.charge_emb = {
            -1: 0,
            0: 1,
            1: 2,
            2: 3,
        }
        self.mask_rate = mask_rate
    # def load_dataset(self):
    #     for file in tqdm(self.root):
    #         print(f"Loading {file} \n")
    #         data_file = torch.load(file)
    #         self.data.extend(data_file)
    #     self.data
        
    def __len__(self):
        return self.length
    
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
        
        with self.env.begin() as txn:
            key = f"{idx:08}".encode("ascii")
            byteflow = txn.get(key)
        graph_data = pickle.loads(byteflow)
        assert isinstance(graph_data, Data), f"Expected torch_geometric.data.Data, got {type(graph)}"
        # TODO zero center the positions by the mean of the pocket atoms
        
        # graph_data = self.data[idx]
        graph_data = self.pocket_centering(graph_data)
        CoM = graph_data.pos[graph_data.is_ligand==1].mean(dim=0)
        #. aligning dtype
        
        # if graph_data.edge_ligand
        
        # if graph_data.source_name not in ["pdbbind_data.pt", "AACLBR.pt", "L_sidechain_data.pt"]:
        #     # padding atom by atom types
        #     # ToDO
        # else:
            # randomly adding by atom types
        
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
        graph_data.charges = charges.float()
        graph_data.virtual_nodes = torch.zeros(graph_data.x.size(0))
        reconstruct_mask = graph_data.is_ligand - graph_data.is_backbone
        new_backbone = graph_data.is_backbone[reconstruct_mask==1]
        # randomly change contents in new_backbone to 1 by the prob of 0.5
        
        # randomly mask part of the backbone by the mask_rate, 0 - reconstruct everything, 1 - reconstruct nothing
        new_backbone = torch.bernoulli(torch.ones_like(new_backbone) * self.mask_rate).float()
        
        graph_data.is_backbone[reconstruct_mask==1] = new_backbone
        
        # if graph_data.source_name not in ["pdbbind_data.pt", "AACLBR.pt", "L_sidechain_data.pt"]:
        #         # get the count of C, N, O, S in the reconstruction part of the ligand
        #         # get count of 0, 1, 2, 3, from graph_data.x[reconstruct_mask == 1]
        #         atom_count = torch.zeros(8)
        #         for i in graph_data.x[reconstruct_mask==1]:
        #             atom_count[int(i)] += 1
        # from IPython import embed; embed()
        if self.pocket_noise:
            
            # Introduce gaussian pocket noise here
            # from IPython import embed; embed()
            gaussian_pocket_noise = torch.randn_like(graph_data.pos[reconstruct_mask==1]) * self.noise_scale
            graph_data.pos[reconstruct_mask==1] += gaussian_pocket_noise
        
        reconstruct_size = (graph_data.is_ligand.sum() - graph_data.is_backbone.sum()).item()
        
        # from IPython import embed; embed()
        if self.params.virtual_node:
            if graph_data.source_name not in ["pdbbind_data.pt", "AACLBR.pt", "L_sidechain_data.pt"]:
                # get the count of C, N, O, S in the reconstruction part of the ligand
                # get count of 0, 1, 2, 3, from graph_data.x[reconstruct_mask == 1]
                atom_count = torch.zeros(8)
                for i in graph_data.x[reconstruct_mask==1]:
                    atom_count[int(i)] += 1
                
                desered_atom_count = torch.tensor([9, 2, 3, 1, 0, 0, 0, 0])
                
                if atom_count.sum() > desered_atom_count.sum():
                    sample_n = np.random.randint(1, self.params.max_virtual_nodes)
                    virtual_x = torch.zeros(sample_n)
                else:
                    virtual_atom_count = (desered_atom_count - atom_count).int()
                    sample_n = int(virtual_atom_count.sum().item())
                    # atom_count = torch.tensor([atom_count[0], atom_count[1], atom
                    # from IPython import embed; embed()
                    virtual_x = torch.zeros(sample_n)
                    # add the atom types to the virtual_x
                    virtual_x[:virtual_atom_count[0]] = 0  # C
                    virtual_x[virtual_atom_count[0]:virtual_atom_count[0]+virtual_atom_count[1]] = 1  # N
                    virtual_x[virtual_atom_count[0]+virtual_atom_count[1]:virtual_atom_count[0]+virtual_atom_count[1]+virtual_atom_count[2]] = 2  # O
                    virtual_x[virtual_atom_count[0]+virtual_atom_count[1]+virtual_atom_count[2]:virtual_atom_count[0]+virtual_atom_count[1]+virtual_atom_count[2]+virtual_atom_count[3]] = 3
                    
                    virtual_pos = torch.stack([CoM] * sample_n)
                
                
                
            # adding random n of virtual nodes by the maximum max-virtual-node
            # if reconstruct_size < self.params.max_virtual_nodes:
            #     sample_n = int(self.params.max_virtual_nodes - reconstruct_size)
            # else:
            else:
                sample_n = np.random.randint(1, self.params.max_virtual_nodes)
                virtual_x = torch.zeros(sample_n)
            # virtual pos is a tensor of shape (sample_n, 3) with CoM * 8
            
            virtual_pos = torch.stack([CoM] * sample_n)
            # add gaussian noise to the virtual positions
            # gaussian_noise = torch.randn_like(virtual_pos)
            # virtual_pos += gaussian_noise
            
            virtual_charges = torch.ones(sample_n) * 3
            virtual_degree = torch.ones(sample_n) * 5
            virtual_is_aromatic = torch.ones(sample_n) * 2
            virtual_is_in_ring = torch.ones(sample_n) * 2
            virtual_hybridization = torch.ones(sample_n) * 4
            virtual_nodes = torch.ones(sample_n)
            # append virtual_x to graph_data.x
            graph_data.x = torch.cat([graph_data.x, virtual_x])
            graph_data.pos = torch.cat([graph_data.pos, virtual_pos])
            graph_data.charges = torch.cat([graph_data.charges, virtual_charges])
            graph_data.degree = torch.cat([graph_data.degree, virtual_degree])
            graph_data.is_aromatic = torch.cat([graph_data.is_aromatic, virtual_is_aromatic])
            graph_data.is_in_ring = torch.cat([graph_data.is_in_ring, virtual_is_in_ring])
            graph_data.hybridization = torch.cat([graph_data.hybridization, virtual_hybridization])
            graph_data.is_backbone = torch.cat([graph_data.is_backbone, torch.zeros(sample_n)])
            graph_data.is_ligand = torch.cat([graph_data.is_ligand, torch.ones(sample_n)])
            graph_data.virtual_nodes = torch.cat([graph_data.virtual_nodes, virtual_nodes])
            virtual_new_id = torch.tensor(range(len(graph_data.x)))[-sample_n:]
            virtual_existed = torch.tensor(range(len(graph_data.x)))[:-sample_n]
            grid1, grid2 = torch.meshgrid(virtual_new_id, virtual_existed)
            grid1 = grid1.flatten()
            grid2 = grid2.flatten()
            # create the new edge index as a bidirectional graph
            new_edge_index = torch.stack([grid1, grid2])
            new_edge_index_reverse = torch.stack([grid2, grid1])
            new_edge_index = torch.cat([new_edge_index, new_edge_index_reverse], dim=1)

            edge_index_new = torch.cat([graph_data.edge_index, new_edge_index], dim=1)
            edge_attr_new = torch.cat([graph_data.edge_attr, torch.zeros(new_edge_index.size(1))])
            edge_ligand_new = torch.cat([torch.tensor(graph_data.edge_ligand), torch.zeros(new_edge_index.size(1))])
            
            grid1, grid2 = torch.meshgrid(virtual_new_id, virtual_new_id)
            mask = grid1 != grid2
            grid1 = grid1[mask]
            grid2 = grid2[mask]
            grid1 = grid1.flatten()
            grid2 = grid2.flatten()
            new_ligand_edge_index = torch.stack([grid1, grid2])
            edge_index_new = torch.cat([edge_index_new, new_ligand_edge_index], dim=1)
            edge_attr_new = torch.cat([edge_attr_new, torch.zeros(new_ligand_edge_index.size(1))])
            edge_ligand_new = torch.cat([edge_ligand_new, torch.ones(new_ligand_edge_index.size(1))]).float()
            
            graph_data.edge_index = edge_index_new
            graph_data.edge_attr = edge_attr_new
            graph_data.edge_ligand = edge_ligand_new
            
            
            # from IPython import embed; embed()
        
        
            # from IPython import embed; embed()
            
        graph_data.degree = graph_data.degree.float()
        graph_data.is_aromatic = graph_data.is_aromatic.float()
        graph_data.is_in_ring = graph_data.is_in_ring.float()
        graph_data.hybridization = graph_data.hybridization.float()
        graph_data.is_backbone = graph_data.is_backbone.float()
        graph_data.is_ligand = graph_data.is_ligand.float()
        # from IPython import embed; embed()
        batch_graph_data = Data(
            x=graph_data.x,
            pos=graph_data.pos,
            edge_index=graph_data.edge_index,
            edge_attr=graph_data.edge_attr,
            edge_ligand=torch.tensor(graph_data.edge_ligand).float(),
            charges=graph_data.charges,
            degree=graph_data.degree,
            is_aromatic=graph_data.is_aromatic,
            is_in_ring=graph_data.is_in_ring,
            hybridization=graph_data.hybridization,
            is_backbone=graph_data.is_backbone,
            is_ligand=graph_data.is_ligand,
            virtual_nodes=graph_data.virtual_nodes,
            ligand_size=torch.tensor(graph_data.is_ligand.sum() - graph_data.is_backbone.sum()).long(),
            id=graph_data.compound_id,
        )
        
        return batch_graph_data


class UAAG2Dataset_sampling(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        hparams,
        save_path,
        dataset_info,
        sample_size=15,
        sample_length=1000,
    ):
        super(UAAG2Dataset_sampling, self).__init__()
        # self.statistics = Statistic()
        self.save_dir = os.path.join(hparams.save_dir, f'run{hparams.id}', f"{save_path}")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
    
        
        self.sample_size = sample_size
        self.sample_length = sample_length
        self.data = data
        # self.load_dataset()
        self.charge_emb = {
            -1: 0,
            0: 1,
            1: 2,
            2: 3,
        }
        
        self.dataset_info = dataset_info
        self.atom_decoder = self.dataset_info.atom_decoder
        self.data = self.preprocess(data)
        self.data.virtual_nodes = torch.zeros(graph_data.x.size(0))
        
        ligand_pos_true = self.data.pos[self.data.is_ligand==1].cpu().detach()
        ligand_atom_true = [self.atom_decoder[int(a)] for a in self.data.x[self.data.is_ligand==1]]
        
        true_molblock = visualize_mol((ligand_pos_true, ligand_atom_true), val_check=False)
        # save the true ligand molblock
        
        
        
        with open(os.path.join(self.save_dir, 'ligand_true.mol'), 'w') as f:
            f.write(true_molblock)
        
        pocket_pos_true = self.data.pos[self.data.is_ligand==0].cpu().detach()
        pocket_atom_true = [self.atom_decoder[int(a)] for a in self.data.x[self.data.is_ligand==0]]
        pocket_molblock = visualize_mol((pocket_pos_true, pocket_atom_true), val_check=False)
        with open(os.path.join(self.save_dir, 'pocket_true.mol'), 'w') as f:
            f.write(pocket_molblock)
        
        
    def __len__(self):
        return self.sample_length
    
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
    
    def preprocess(self, data):
        graph_data = self.pocket_centering(data)
        if not hasattr(graph_data, 'compound_id'):
            graph_data.compound_id = graph_data.componud_id
        if not hasattr(graph_data, 'edge_ligand'):
            graph_data.edge_ligand = torch.ones(graph_data.edge_attr.size(0))
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
            ids=torch.tensor(range(len(graph_data.x))),
        )
        
        return batch_graph_data
    


    def __getitem__(self, idx):
        
        reconstruct_mask = self.data.is_ligand - self.data.is_backbone
        x = self.data.x[reconstruct_mask==0]
        pos = self.data.pos[reconstruct_mask==0]
        charges = self.data.charges[reconstruct_mask==0]
        degree = self.data.degree[reconstruct_mask==0]
        is_aromatic = self.data.is_aromatic[reconstruct_mask==0]
        is_in_ring = self.data.is_in_ring[reconstruct_mask==0]
        hybridization = self.data.hybridization[reconstruct_mask==0]
        is_ligand = self.data.is_ligand[reconstruct_mask==0]
        is_backbone = self.data.is_backbone[reconstruct_mask==0]
        ids = self.data.ids[reconstruct_mask==0]
        
        virtual_nodes = self.data.virtual_nodes[reconstruct_mask==0]
        # remove the information of current edges connect to ligands
        
        edge_mask = torch.isin(self.data.edge_index[0], ids) & torch.isin(self.data.edge_index[1], ids)
        edge_index = self.data.edge_index[:, edge_mask]
        edge_attr = self.data.edge_attr[edge_mask]
        edge_ligand = torch.tensor(self.data.edge_ligand)[edge_mask]
        
        # recreate the ligand idx to create a new edge indexes
        
        map_ids = {int(ids[i]): i for i in range(len(ids))}
        
        new_edge_index = torch.empty_like(edge_index)
        for i in range(edge_index.size(1)):
            new_edge_index[0, i] = map_ids[int(edge_index[0, i])]
            new_edge_index[1, i] = map_ids[int(edge_index[1, i])]
        edge_index = new_edge_index
        
        # edge_index = torch.stack([
        #     new_ids[(edge_index[0].unsqueeze(-1)==ids).nonzero(as_tuple=True)[1]],
        #     new_ids[(edge_index[1].unsqueeze(-1)==ids).nonzero(as_tuple=True)[1]],
        # ])
        
        # Add new nodes based on the assigned sample size
        # print("Inside the get item function")
        # from IPython import embed; embed()
        x_new = [0] * 9 + [1] * 2 + [2] * 3 + [3] * 1 
        x_new = torch.cat([x, torch.tensor(x_new)])
        pos_new = torch.cat([pos, torch.randn(self.sample_size, 3)])
        charges_new = torch.cat([charges, torch.multinomial(self.dataset_info.charge_types, self.sample_size, replacement=True)])
        degree_new = torch.cat([degree, torch.multinomial(self.dataset_info.degree, self.sample_size, replacement=True)])
        is_aromatic_new = torch.cat([is_aromatic, torch.multinomial(self.dataset_info.is_aromatic, self.sample_size, replacement=True)])
        is_in_ring_new = torch.cat([is_in_ring, torch.multinomial(self.dataset_info.is_ring, self.sample_size, replacement=True)])
        hybridization_new = torch.cat([hybridization, torch.multinomial(self.dataset_info.hybridization, self.sample_size, replacement=True)])
        is_ligand_new = torch.cat([is_ligand, torch.ones(self.sample_size)])
        is_backbone_new = torch.cat([is_backbone, torch.zeros(self.sample_size)])
        virtual_nodes_new = torch.cat(virtual_nodes, [torch.ones(self.sample_size)])
        # Add new edges, firstly interaction edge between ligand and pocket (edge_ligand=0)
        # Then adding the edges inside ligands (edge_ligand=1)
        
        ids_new = torch.tensor(range(len(x_new)))
        ids_new_node = ids_new[-self.sample_size:]
        ids_existed = ids_new[:-self.sample_size]
        
        # adding a new full connected graph of new nodes to existed graph
        
        grid1, grid2 = torch.meshgrid(ids_new_node, ids_existed)
        grid1 = grid1.flatten()
        grid2 = grid2.flatten()
        # create the new edge index as a bidirectional graph
        new_edge_index = torch.stack([grid1, grid2])
        new_edge_index_reverse = torch.stack([grid2, grid1])
        new_edge_index = torch.cat([new_edge_index, new_edge_index_reverse], dim=1)
        
        # print("New interaction edges")
        # from IPython import embed; embed()
        
        # Adding interaction edge information
        edge_index_new = torch.cat([edge_index, new_edge_index], dim=1)
        edge_attr_new = torch.cat([edge_attr, torch.zeros(new_edge_index.size(1))])
        edge_ligand_new = torch.cat([edge_ligand, torch.zeros(new_edge_index.size(1))])
        
        # Adding edge information inside the new nodes
        
        grid1, grid2 = torch.meshgrid(ids_new_node, ids_new_node)
        mask = grid1 != grid2
        grid1 = grid1[mask]
        grid2 = grid2[mask]
        grid1 = grid1.flatten()
        grid2 = grid2.flatten()
        new_ligand_edge_index = torch.stack([grid1, grid2])
        
        # print("New edges inside ligand")
        # from IPython import embed; embed()
        
        edge_index_new = torch.cat([edge_index_new, new_ligand_edge_index], dim=1)
        edge_attr_new = torch.cat([edge_attr_new, torch.zeros(new_ligand_edge_index.size(1))])
        edge_ligand_new = torch.cat([edge_ligand_new, torch.ones(new_ligand_edge_index.size(1))]).float()
   
        
        
        output_graph = Data(
            x=x_new,
            pos=pos_new,
            edge_index=edge_index_new,
            edge_attr=edge_attr_new,
            edge_ligand=edge_ligand_new,
            charges=charges_new,
            degree=degree_new,
            is_aromatic=is_aromatic_new,
            is_in_ring=is_in_ring_new,
            hybridization=hybridization_new,
            is_backbone=is_backbone_new,
            virtual_nodes=virtual_nodes_new,
            is_ligand=is_ligand_new,
            ids=ids_new,
            id=self.data.id,
        )
        
        return output_graph

      
    
class UAAG2DataModule(pl.LightningDataModule):
    def __init__(self, cfg, train_data, val_data, test_data, sampler, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.cfg = cfg
        # split into train, val, test with test & valid consisting 2000 samples each
        # self._log_hyperparams = True
        self.pin_memory = True
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.sampler = sampler
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
            shuffle=False,
            persistent_workers=False,
            drop_last=True,
            sampler=self.sampler,
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
            drop_last=True,
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
    def __init__(self, hparams, info_path):
        self.hparams = hparams
        self.info_path = info_path
        self.process()
        self.get_decoder()
    def process(self):
        # read from the info_path, a pickle file
        
        with open(self.info_path, 'rb') as f:
            data_info = pickle.load(f)

        
        self.atom_types = []
        
        sum_x = []
        if self.hparams.virtual_node:
            # add another value of 0 to data_info['x']
            data_info['charge'][2] = 0
            data_info['aro'][2] = 0
            data_info['degree'][5] = 0
            data_info['hybrid'][4] = 0
            data_info['ring'][2] = 0
            
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
            "Br": 7
        }
        atom_decoder  = {v: k for k, v in atom_encoder.items()}
        self.atom_decoder = atom_decoder
            
        bond_encoder = {
            Chem.BondType.SINGLE: 1,
            Chem.BondType.DOUBLE: 2,
            Chem.BondType.AROMATIC: 3,
            Chem.BondType.TRIPLE: 4,
        }
        bond_decoder = {v: k for k, v in bond_encoder.items()}
        self.bond_decoder = bond_decoder
        
        
        