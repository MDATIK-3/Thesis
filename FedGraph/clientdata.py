import torch
from typing import List, Dict
import sys
import os

KAGGLE_PATH = '/kaggle/working/Thesis' if 'kaggle' in os.getcwd() else os.path.dirname(os.path.abspath(__file__))
sys.path.append(KAGGLE_PATH)

class ClientData():
    def __init__(self, node_list: List, node_feature: Dict, node_label: Dict, send_info: Dict, receive_info: Dict,
                 node_neighbor, edge_index, process_id, **kwargs) -> None:
        self.node_list = node_list
        self.node_feature = node_feature
        self.send_info = send_info
        self.receive_info = receive_info
        self.node_neighbor = node_neighbor
        self.process_id = process_id
        self.node_label = node_label
        self.edge_index = edge_index
        self.node_index_2_client_index, self.x = self.get_x()
        self.y = self.get_y()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_y(self):
        y = None
        nodes = self.node_list
        node_label = self.node_label
        for node in nodes:
            if y is None:
                y = node_label[node].unsqueeze(dim=0)
            else:
                y = torch.cat((y, node_label[node].unsqueeze(dim=0)), dim=0)
        return y

    def get_x(self):
        x = None
        nodes = self.node_list
        node_feature = self.node_feature
        node_index_2_client_index = {}
        id = 0
        for node in nodes:
            if x is None:
                x = node_feature[node].unsqueeze(0)
            else:
                x = torch.cat((x, node_feature[node].unsqueeze(0)), dim=0)
            node_index_2_client_index[node] = id
            id += 1
        return node_index_2_client_index, x
