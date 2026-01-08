import json

import pdb
import sys
import os

sys.path.append('.')
import pickle

import numpy as np
from torch_geometric.data import Batch
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from IPython import embed
from Bio.PDB import PDBParser
from Bio.PDB import Selection
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from tqdm import tqdm
from Bio.PDB import PDBParser
from torch_geometric.utils import dense_to_sparse, sort_edge_index
# import Counter
from collections import Counter
def data_inspect(path):
    
    datalist = torch.load(path)
    x_dict_all = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 0,
    }
    
    edge_dict_all = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
    }
    
    charge_dict_all = {
        -1: 0,
        0: 0,
        1: 0,
    }
    
    aro_dict_all = {
        0: 0,
        1: 0,
    }
    
    degree_dict_all = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
    }
    
    hybrid_dict_all = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
    }
    
    ring_dict_all = {
        0: 0,
        1: 0,
    }
    
    
    
    for graph in datalist:
        
        data_x = graph.x.tolist()
        data_x = [int(i) for i in data_x]
        
        x_dict = Counter(data_x)
        
        data_edge = graph.edge_attr.tolist()
        data_edge = [int(i) for i in data_edge]
        
        edge_dict = Counter(data_edge)
        
        data_charge = graph.charges.tolist()
        data_charge = [int(i) for i in data_charge]
        
        charge_dict = Counter(data_charge)
        
        data_aro = graph.is_aromatic.tolist()
        data_aro = [int(i) for i in data_aro]
        
        aro_dict = Counter(data_aro)
        
        data_degree = graph.degree.tolist()
        data_degree = [int(i) for i in data_degree]
        
        degree_dict = Counter(data_degree)
        
        data_ring = graph.is_in_ring.tolist()
        data_ring = [int(i) for i in data_ring]
        
        ring_dict = Counter(data_ring)
        
        data_hybrid = graph.hybridization.tolist()
        data_hybrid = [int(i) for i in data_hybrid]
        
        hybrid_dict = Counter(data_hybrid)
        
        x_dict_all = {key: x_dict_all[key] + x_dict.get(key, 0) for key in x_dict_all}
        edge_dict_all = {key: edge_dict_all[key] + edge_dict.get(key, 0) for key in edge_dict_all}
        charge_dict_all = {key: charge_dict_all[key] + charge_dict.get(key, 0) for key in charge_dict_all}
        aro_dict_all = {key: aro_dict_all[key] + aro_dict.get(key, 0) for key in aro_dict_all}
        degree_dict_all = {key: degree_dict_all[key] + degree_dict.get(key, 0) for key in degree_dict_all}
        hybrid_dict_all = {key: hybrid_dict_all[key] + hybrid_dict.get(key, 0) for key in hybrid_dict_all}
        ring_dict_all = {key: ring_dict_all[key] + ring_dict.get(key, 0) for key in ring_dict_all}
        
    return x_dict_all, edge_dict_all, charge_dict_all, aro_dict_all, degree_dict_all, hybrid_dict_all, ring_dict_all
    
    
def main():
    pdb_path = "/home/qcx679/hantang/UAAG2/data/full_graph/data"
    pdbbind_path = "/home/qcx679/hantang/UAAG2/data/full_graph/pdbbind"
    single_naa = "/home/qcx679/hantang/UAAG2/data/full_graph/naa"
    final_x_dict = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 0,
    }
    
    final_edge_dict = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
    }
    
    final_charge_dict = {
        -1: 0,
        0: 0,
        1: 0,
    }
    
    final_aro_dict = {
        0: 0,
        1: 0,
    }
    
    final_degree_dict = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
    }
    
    final_hybrid_dict = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
    }
    
    final_ring_dict = {
        0: 0,
        1: 0,
    }
    
    print("Loop over PDBs")
    pdb_dir = os.listdir(pdb_path)
    for pdb_pt in tqdm(pdb_dir):
        full_path = os.path.join(pdb_path, pdb_pt)
        x_dict, edge_dict, charge_dict, aro_dict, \
            degree_dict, hybrid_dict, ring_dict = data_inspect(full_path)
        
        # update the final dict
        final_x_dict = {key: final_x_dict[key] + x_dict.get(key, 0) for key in final_x_dict}
        final_edge_dict = {key: final_edge_dict[key] + edge_dict.get(key, 0) for key in final_edge_dict}
        final_charge_dict = {key: final_charge_dict[key] + charge_dict.get(key, 0) for key in final_charge_dict}
        final_aro_dict = {key: final_aro_dict[key] + aro_dict.get(key, 0) for key in final_aro_dict}
        final_degree_dict = {key: final_degree_dict[key] + degree_dict.get(key, 0) for key in final_degree_dict}
        final_hybrid_dict = {key: final_hybrid_dict[key] + hybrid_dict.get(key, 0) for key in final_hybrid_dict}
        final_ring_dict = {key: final_ring_dict[key] + ring_dict.get(key, 0) for key in final_ring_dict}
    
    print("Loop over PDBbind")
    pdbbind_dir = os.listdir(pdbbind_path)
    for pdb_pt in tqdm(pdbbind_dir):
        full_path = os.path.join(pdbbind_path, pdb_pt)
        x_dict, edge_dict, charge_dict, aro_dict, \
            degree_dict, hybrid_dict, ring_dict = data_inspect(full_path)
        
        # update the final dict
        final_x_dict = {key: final_x_dict[key] + x_dict.get(key, 0) for key in final_x_dict}
        final_edge_dict = {key: final_edge_dict[key] + edge_dict.get(key, 0) for key in final_edge_dict}
        final_charge_dict = {key: final_charge_dict[key] + charge_dict.get(key, 0) for key in final_charge_dict}
        final_aro_dict = {key: final_aro_dict[key] + aro_dict.get(key, 0) for key in final_aro_dict}
        final_degree_dict = {key: final_degree_dict[key] + degree_dict.get(key, 0) for key in final_degree_dict}
        final_hybrid_dict = {key: final_hybrid_dict[key] + hybrid_dict.get(key, 0) for key in final_hybrid_dict}
        final_ring_dict = {key: final_ring_dict[key] + ring_dict.get(key, 0) for key in final_ring_dict}
    
    print("Loop over single NAA")
    naa_dir = os.listdir(single_naa)
    for pdb_pt in tqdm(naa_dir):
        full_path = os.path.join(single_naa, pdb_pt)
        x_dict, edge_dict, charge_dict, aro_dict, \
            degree_dict, hybrid_dict, ring_dict = data_inspect(full_path)
        
        # update the final dict
        final_x_dict = {key: final_x_dict[key] + x_dict.get(key, 0) for key in final_x_dict}
        final_edge_dict = {key: final_edge_dict[key] + edge_dict.get(key, 0) for key in final_edge_dict}
        final_charge_dict = {key: final_charge_dict[key] + charge_dict.get(key, 0) for key in final_charge_dict}
        final_aro_dict = {key: final_aro_dict[key] + aro_dict.get(key, 0) for key in final_aro_dict}
        final_degree_dict = {key: final_degree_dict[key] + degree_dict.get(key, 0) for key in final_degree_dict}
        final_hybrid_dict = {key: final_hybrid_dict[key] + hybrid_dict.get(key, 0) for key in final_hybrid_dict}
        final_ring_dict = {key: final_ring_dict[key] + ring_dict.get(key, 0) for key in final_ring_dict}
    
    statistic_dict = {
        "x": final_x_dict,
        "edge": final_edge_dict,
        "charge": final_charge_dict,
        "aro": final_aro_dict,
        "degree": final_degree_dict,
        "hybrid": final_hybrid_dict,
        "ring": final_ring_dict,
    }
    # save to pickle
    with open("/home/qcx679/hantang/UAAG2/data/full_graph/statistic.pkl", "wb") as f:
        pickle.dump(statistic_dict, f)
    # from IPython import embed; embed()

if __name__ == '__main__':
    main()