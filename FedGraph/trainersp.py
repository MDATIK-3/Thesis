#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import logging
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score

# Kaggle-safe device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TrainerSP:
    def __init__(self, args, model, data=None):
        self.args = args
        self.model = model
        self.data = data

    def set_model(self, model_params):
        self.model.load_state_dict(model_params)

    def set_data(self, clientdata):
        self.data = clientdata

    def train(self):
        """Node classification (if ever used)"""
        model = self.model.to(device)
        data = self.data.to(device)

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=self.args.lr,
                                     weight_decay=5e-4)
        criterion = torch.nn.NLLLoss()

        for e in range(self.args.epoch):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

        return model.state_dict(), data.num_nodes

    def trainLP(self):
        """Link prediction training"""
        model = self.model.to(device)
        data = self.data.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        criterion = torch.nn.BCEWithLogitsLoss()

        for e in range(self.args.epoch):
            model.train()
            optimizer.zero_grad()

            z = model.encode(data)

            # negative sample per epoch
            neg_edge_index = negative_sampling(
                edge_index=data.edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=data.edge_label_index.size(1),
                method='sparse'
            )

            edge_label_index = torch.cat(
                [data.edge_label_index, neg_edge_index], dim=-1
            )

            edge_label = torch.cat([
                data.edge_label,
                torch.zeros(neg_edge_index.size(1), device=device)
            ])

            out = model.decode(z, edge_label_index).view(-1)
            loss = criterion(out, edge_label)
            loss.backward()
            optimizer.step()

        return model.state_dict(), data.num_nodes

    @torch.no_grad()
    def evaluate_auc(self, data):
        """Optional: used on val/test by the server"""
        self.model.eval()
        data = data.to(device)

        z = self.model.encode(data)
        out = self.model.decode(z, data.edge_label_index).sigmoid().cpu()
        y = data.edge_label.cpu()

        return roc_auc_score(y, out)
