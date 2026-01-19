"""
get_data() : 获取SubGraph设置下的图数据
load_partition_data() : 获取切分后的图数据
"""

import torch
import numpy as np
import networkx as nx
import pymetis
from torch_geometric.datasets import Planetoid, CitationFull, Coauthor
from torch_geometric.data import Data
from data_update import *
import os
import os.path as osp

# Automatically set ROOT for Kaggle / Colab / local
ROOT = '/kaggle/working/FC-FedGCN/data' if 'kaggle' in os.getcwd() else osp.dirname(__file__)
DATA_NAME = ['Cora', 'CiteSeer', 'PubMed']

def get_data(name):
    """
    Subgraph setting 下的数据集, 返回 PyG 数据对象

    Args:
        name (str): 数据集名称

    Returns:
        data (torch_geometric.data.Data)
    """
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

    # 90% train / 10% test split
    train_num = int(len(data.x) * 0.9)
    all_train_mask = [i <= train_num for i in range(len(data.x))]
    all_test_mask  = [not x for x in all_train_mask]

    data.train_mask = torch.tensor(all_train_mask, dtype=torch.bool)
    data.test_mask  = torch.tensor(all_test_mask, dtype=torch.bool)
    return data

def load_partition_data(num_clients: int, data: Data, isImpaired=False):
    """
    将整图切分为多个子图 (客户端划分)

    Args:
        num_clients (int): 客户端数量
        data (Data): PyG 图数据
        isImpaired (bool): 是否考虑损失的link (可扩展)

    Returns:
        nodes_list (list[list[int]]): 每个客户端的节点列表
    """
    # 转换 PyG Data 为 NetworkX 图
    G = nx.Graph()
    G.add_nodes_from(range(data.x.shape[0]))
    edges = np.array(data.edge_index.T, dtype=int)
    G.add_edges_from(edges)

    # 转换为 adjacency list 给 pymetis
    adjacency = [list(G.neighbors(n)) for n in range(len(G))]

    # 使用 pymetis 划分
    n_cuts, parts = pymetis.part_graph(num_clients, adjacency=adjacency)

    # 构建每个客户端节点列表
    nodes_list = [[] for _ in range(num_clients)]
    for node_id, part_id in enumerate(parts):
        nodes_list[part_id].append(node_id)

    return nodes_list
