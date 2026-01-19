import torch
import numpy as np
import networkx as nx
import pymetis
from torch_geometric.datasets import Planetoid, CitationFull, Coauthor
from torch_geometric.data import Data
from data_update import *
import os
import os.path as osp

ROOT = '/kaggle/working/Thesis/data' if 'kaggle' in os.getcwd() else osp.dirname(__file__)
DATA_NAME = ['Cora', 'CiteSeer', 'PubMed']

def get_data(name):
    global data
    if name in DATA_NAME:
        dataset = Planetoid(root=ROOT, name=name)
        data = dataset[0]
    elif name == "DBLP":
        dataset = CitationFull(root=ROOT, name=name)
        data = dataset[0]
    elif name in ['CS', 'Physics']:
        dataset = Coauthor(root=ROOT, name=name)
        data = dataset[0]
    elif name == 'LastFM':
        data = lastfm_to_Tensor()
    elif name == 'FaceBook_Large':
        data = facebook_large_to_Tensor()
    elif name == 'FaceBook':
        data = facebook_to_Tensor()

    train_num = int(len(data.x) * 0.9)
    all_train_mask = [i <= train_num for i in range(len(data.x))]
    all_test_mask  = [not x for x in all_train_mask]

    data.train_mask = torch.tensor(all_train_mask, dtype=torch.bool)
    data.test_mask  = torch.tensor(all_test_mask, dtype=torch.bool)
    return data

def load_partition_data(num_clients: int, data: Data, isImpaired=False):
    G = nx.Graph()
    G.add_nodes_from(range(data.x.shape[0]))
    edges = np.array(data.edge_index.cpu().T, dtype=int)
    G.add_edges_from(edges)

    adjacency = [list(G.neighbors(n)) for n in range(len(G))]
    n_cuts, parts = pymetis.part_graph(num_clients, adjacency=adjacency)

    nodes_list = [[] for _ in range(num_clients)]
    for node_id, part_id in enumerate(parts):
        nodes_list[part_id].append(node_id)

    return nodes_list
