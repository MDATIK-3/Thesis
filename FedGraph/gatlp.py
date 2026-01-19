from torch_geometric.nn import GATConv
import torch.nn.functional as F
import torch
from torch_geometric.data import Data

class GATLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv1 = GATConv(in_dim, 8, heads=8, concat=True, dropout=0.6)
        self.conv2 = GATConv(8 * 8, out_dim, dropout=0.6)

    def encode(self, data):
        x = self.conv1(data.x, data.edge_index).relu()
        x = F.dropout(x, training=self.training)
        return self.conv2(x, data.edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z, edge_label_index):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    def forward(self, data: Data):
        x = self.conv1(data.x, data.edge_index)
        x = F.relu(x)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)
