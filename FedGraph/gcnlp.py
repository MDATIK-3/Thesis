import torch
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T
from data_loader import get_data

class GCNLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, data):
        x = self.conv1(data.x, data.edge_index).relu()
        return self.conv2(x, data.edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

def train_link_predictor(model, train_data, val_data, optimizer, criterion, n_epochs=100):
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data)

        neg_edge_index = negative_sampling(
            edge_index=train_data.edge_index,
            num_nodes=train_data.num_nodes,
            num_neg_samples=train_data.edge_label_index.size(1),
            method='sparse'
        )

        edge_label_index = torch.cat([train_data.edge_label_index, neg_edge_index], dim=-1)
        edge_label = torch.cat([
            train_data.edge_label,
            train_data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)

        out = model.decode(z, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()

        val_auc = eval_link_predictor(model, val_data)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}, Loss {loss:.3f}, Val AUC {val_auc:.3f}")

    return model

@torch.no_grad()
def eval_link_predictor(model, data):
    model.eval()
    z = model.encode(data)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = get_data('Cora').to(device)
    model = GCNLP(data.num_node_features, 128, 64).to(device)

    split = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=1.0
    )

    train_data, val_data, test_data = split(data)
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_link_predictor(model, train_data, val_data, optimizer, criterion, 100)
    test_auc = eval_link_predictor(model, test_data)
    print(f"Test AUC: {test_auc:.3f}")
