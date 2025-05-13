import json
import networkx as nx
import networkx.algorithms.isomorphism as iso
from rdkit import Chem
import sys
from networkx.readwrite import json_graph
sys.path.append('.')
sys.path.append('..')
from uaag.utils import aa_check, mol_to_graph
with open('data/aa_graph.json', 'rb') as json_file:
    AA_GRAPH_DICT = json.load(json_file)
    json_file.close()

aoc_mol = "/home/qcx679/hantang/UAAG2/data/2018_PNAS_DeltaG/Aoc.mol"
aoc_mol = Chem.MolFromMolFile(aoc_mol)
G = mol_to_graph(aoc_mol)
node_link_dict_aoc = json_graph.node_link_data(G)

AA_GRAPH_DICT['Aoc'] = node_link_dict_aoc

tme_mol = "/home/qcx679/hantang/UAAG2/data/2018_PNAS_DeltaG/Tme.mol"
tme_mol = Chem.MolFromMolFile(tme_mol)
G = mol_to_graph(tme_mol)
node_link_dict_tme = json_graph.node_link_data(G)
AA_GRAPH_DICT['Tme'] = node_link_dict_tme

hsm_mol = "/home/qcx679/hantang/UAAG2/data/2018_PNAS_DeltaG/hSM.mol"
hsm_mol = Chem.MolFromMolFile(hsm_mol)
G = mol_to_graph(hsm_mol)
node_link_dict_hsm = json_graph.node_link_data(G)
AA_GRAPH_DICT['hSM'] = node_link_dict_hsm

tbu_mol = "/home/qcx679/hantang/UAAG2/data/2018_PNAS_DeltaG/tBu.mol"
tbu_mol = Chem.MolFromMolFile(tbu_mol)
G = mol_to_graph(tbu_mol)
node_link_dict_tbu = json_graph.node_link_data(G)
AA_GRAPH_DICT['tBu'] = node_link_dict_tbu

cpa_mol = "/home/qcx679/hantang/UAAG2/data/2018_PNAS_DeltaG/Cpa.mol"
cpa_mol = Chem.MolFromMolFile(cpa_mol)
G = mol_to_graph(cpa_mol)
node_link_dict_cpa = json_graph.node_link_data(G)
AA_GRAPH_DICT['Cpa'] = node_link_dict_cpa

aib_mol = "/home/qcx679/hantang/UAAG2/data/2018_PNAS_DeltaG/Aib.mol"
aib_mol = Chem.MolFromMolFile(aib_mol)
G = mol_to_graph(aib_mol)
node_link_dict_aib = json_graph.node_link_data(G)
AA_GRAPH_DICT['Aib'] = node_link_dict_aib

meg_mol = "/home/qcx679/hantang/UAAG2/data/2018_PNAS_DeltaG/MeG.mol"
meg_mol = Chem.MolFromMolFile(meg_mol)
G = mol_to_graph(meg_mol)
node_link_dict_meg = json_graph.node_link_data(G)
AA_GRAPH_DICT['MeG'] = node_link_dict_meg

mea_mol = "/home/qcx679/hantang/UAAG2/data/2018_PNAS_DeltaG/MeA.mol"
mea_mol = Chem.MolFromMolFile(mea_mol)
G = mol_to_graph(mea_mol)
node_link_dict_mea = json_graph.node_link_data(G)
AA_GRAPH_DICT['MeA'] = node_link_dict_mea

meb_mol = "/home/qcx679/hantang/UAAG2/data/2018_PNAS_DeltaG/MeB.mol"
meb_mol = Chem.MolFromMolFile(meb_mol)
G = mol_to_graph(meb_mol)
node_link_dict_meb = json_graph.node_link_data(G)
AA_GRAPH_DICT['MeB'] = node_link_dict_meb

mef_mol = "/home/qcx679/hantang/UAAG2/data/2018_PNAS_DeltaG/MeF.mol"
mef_mol = Chem.MolFromMolFile(mef_mol)
G = mol_to_graph(mef_mol)
node_link_dict_mef = json_graph.node_link_data(G)
AA_GRAPH_DICT['MeF'] = node_link_dict_mef

_2th_mol = "/home/qcx679/hantang/UAAG2/data/2018_PNAS_DeltaG/2th.mol"
_2th_mol = Chem.MolFromMolFile(_2th_mol)
G = mol_to_graph(_2th_mol)
node_link_dict_2th = json_graph.node_link_data(G)
AA_GRAPH_DICT['2th'] = node_link_dict_2th

_3th_mol = "/home/qcx679/hantang/UAAG2/data/2018_PNAS_DeltaG/3th.mol"
_3th_mol = Chem.MolFromMolFile(_3th_mol)
G = mol_to_graph(_3th_mol)
node_link_dict_3th = json_graph.node_link_data(G)
AA_GRAPH_DICT['3th'] = node_link_dict_3th

yme_mol = "/home/qcx679/hantang/UAAG2/data/2018_PNAS_DeltaG/YMe.mol"
yme_mol = Chem.MolFromMolFile(yme_mol)
G = mol_to_graph(yme_mol)
node_link_dict_yme = json_graph.node_link_data(G)
AA_GRAPH_DICT['YMe'] = node_link_dict_yme

_2np = "/home/qcx679/hantang/UAAG2/data/2018_PNAS_DeltaG/2Np.mol"
_2np = Chem.MolFromMolFile(_2np)
G = mol_to_graph(_2np)
node_link_dict_2np = json_graph.node_link_data(G)
AA_GRAPH_DICT['2Np'] = node_link_dict_2np

bzt = "/home/qcx679/hantang/UAAG2/data/2018_PNAS_DeltaG/Bzt.mol"
bzt = Chem.MolFromMolFile(bzt)
G = mol_to_graph(bzt)
node_link_dict_bzt = json_graph.node_link_data(G)
AA_GRAPH_DICT['Bzt'] = node_link_dict_bzt


with open('data/aa_graph.json', 'w') as json_file:
    json.dump(AA_GRAPH_DICT, json_file)
    json_file.close()



from IPython import embed; embed()
