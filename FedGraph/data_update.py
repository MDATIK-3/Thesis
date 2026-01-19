import csv
import json
import torch
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
from torch_sparse import coalesce
from torch_geometric.data import Data
import os

ROOT = '/kaggle/working/data' if 'kaggle' in os.getcwd() else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

def file_convert(args):
    src = f"{ROOT}/concept/{args.dataset}_edge.csv"
    dst = f"{ROOT}/concept/{args.dataset}_edge.txt"
    with open(src, 'r') as a, open(dst, 'w') as f:
        reader = csv.reader(a)
        for row in reader:
            f.write('\t'.join(row) + '\n')

def read_csv(filepath):
    with open(filepath, 'r') as file:
        return list(csv.reader(file))

def PowerSetsInsert(data):
    res = data[0]
    for i in range(1, len(data)):
        res = [val for val in res if val in data[i]]
    return res

def load_lastfm():
    edges = pd.read_csv(f"{ROOT}/lastfm_asia/lastfm_asia_edges.csv")
    G = nx.from_pandas_edgelist(edges, 'node_1', 'node_2')
    with open(f"{ROOT}/lastfm_asia/lastfm_asia_features.json", 'r') as f:
        feats = json.load(f)

    nodes = sorted(G.nodes())
    df_node = pd.DataFrame(nodes)

    features = list(feats.values())
    max_feat = max(max(v) for v in features)
    feat01 = np.zeros((len(features), max_feat+1), dtype=int)
    for i, vals in enumerate(features):
        for v in vals:
            feat01[i][v] = 1

    label = pd.read_csv(f"{ROOT}/lastfm_asia/lastfm_asia_target.csv").iloc[:, -1]
    df_label = label

    node_subjects = pd.concat([df_node, pd.DataFrame(feat01), df_label], axis=1)
    return G, node_subjects

def facebook_to_Tensor():
    edges = pd.read_csv(f"{ROOT}/facebook_large/facebook_edges.csv")
    G = nx.from_pandas_edgelist(edges, 'id_1', 'id_2')

    with open(f"{ROOT}/facebook_large/facebook_features.json", 'r') as f:
        feats = json.load(f)

    nodes = sorted(G.nodes())
    features = list(feats.values())
    max_feat = max(max(v) for v in features)

    feat01 = np.zeros((len(features), max_feat+1), dtype=int)
    for i, vals in enumerate(features):
        for v in vals:
            feat01[i][v] = 1

    label = pd.read_csv(f"{ROOT}/facebook_large/facebook_target.csv").iloc[:, -1].to_numpy()

    x = torch.from_numpy(feat01).float()
    y = torch.from_numpy(label).long()

    edge_index = torch.tensor(np.array(G.edges()), dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index, y=y)
    return data

def lastfm_to_Tensor():
    edges = pd.read_csv(f"{ROOT}/lastfm_asia/lastfm_asia_edges.csv").to_numpy()
    with open(f"{ROOT}/lastfm_asia/lastfm_asia_features.json", 'r') as f:
        feats = json.load(f)

    features = list(feats.values())
    max_feat = max(max(vals) for vals in features)
    feat01 = np.zeros((len(features), max_feat+1), dtype=int)
    for i, vals in enumerate(features):
        for v in vals:
            feat01[i][v] = 1

    label = pd.read_csv(f"{ROOT}/lastfm_asia/lastfm_asia_target.csv").iloc[:, -1].to_numpy()

    x = torch.tensor(feat01, dtype=torch.float)
    y = torch.tensor(label, dtype=torch.long)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index, y=y)

def reddit_to_Tensor():
    reddit = np.load(f"{ROOT}/reddit_2/reddit_data.npz")
    adj = sp.load_npz(f"{ROOT}/reddit_2/reddit_graph.npz")

    x = torch.tensor(reddit['feature'], dtype=torch.float)
    y = torch.tensor(reddit['label'], dtype=torch.long)
    row = torch.tensor(adj.row, dtype=torch.long)
    col = torch.tensor(adj.col, dtype=torch.long)

    edge_index = torch.stack([row, col], dim=0)
    edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))
    return Data(x=x, edge_index=edge_index, y=y)
