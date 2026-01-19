import torch
import sys

sys.path.append('/home/lames/code/Gcode')
from clientdata import ClientData
from torch_geometric.data import Data

root = '/home/lames/code/data'
from torch_geometric.utils import to_dense_adj, subgraph, k_hop_subgraph


def get_adj(edge_index, add_loop_self=True):
    """
    Compute normalized adjacency (symmetric Laplacian)
    """
    N = int(edge_index.max().item()) + 1
    edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)  # make symmetric

    if add_loop_self:
        loop_index = torch.arange(N, device=edge_index.device)
        loop_index = torch.stack([loop_index, loop_index], dim=0)
        edge_index = torch.cat([edge_index, loop_index], dim=1)

    adj = to_dense_adj(edge_index)[0].to(device)
    deg = adj.sum(dim=1)
    D_sqrt_inv = torch.diag(torch.sqrt(1.0 / deg))
    return D_sqrt_inv @ adj @ D_sqrt_inv

def get_client_data(data: Data, process_id, node_lists) -> ClientData:
    nodes = node_lists[process_id - 1]

    node_feature = {}
    node_label = {}
    train_mask = []
    x = data.x
    y = data.y
    for node in nodes:
        node_feature[node] = x[node]
        node_label[node] = y[node]
        train_mask.append(data.train_mask[node])

    train_mask = torch.Tensor(train_mask).bool()

    node2client_dict = {}
    for i in range(len(node_lists)):
        for node in node_lists[i]:
            node2client_dict[node] = i

    edge_index = data.edge_index

    edge_dict = {i: [] for i in range(data.num_nodes)}
    for i in range(len(edge_index[0])):
        a = edge_index[0][i].item()
        b = edge_index[1][i].item()
        edge_dict[a].append(b)
        edge_dict[b].append(a)

    receive_info = []
    send_info = []
    for node in nodes:
        for n in edge_dict[node]:
            if n not in nodes:
                s = [node, node2client_dict[n] + 1]
                r = [n, node2client_dict[n] + 1]
                send_info.append(s)
                receive_info.append(r)

    node_neighbor = {node: edge_dict[node] for node in nodes}

    sub = subgraph(subset=torch.Tensor(nodes), edge_index=edge_index)
    edge_index_new = sub[0]

    for i in range(len(edge_index_new[0])):
        a = edge_index_new[0][i].item()
        b = edge_index_new[1][i].item()
        edge_index_new[0][i] = torch.tensor(node2client_dict[a])
        edge_index_new[1][i] = torch.tensor(node2client_dict[b])

    clientdata = ClientData(
        node_list=nodes,
        node_feature=node_feature,
        node_label=node_label,
        send_info=send_info,
        receive_info=receive_info,
        node_neighbor=node_neighbor,
        edge_index=edge_index_new,
        process_id=process_id,
        train_mask=train_mask
    )

    print('send_node:  {}  receive_node: {}'.format(len(send_info), len(receive_info)))
    return clientdata
def get_client_data_without_link(data: Data, process_id, node_lists) -> Data:
    nodes = node_lists[process_id - 1]
    edge_index = data.edge_index

    node2idx = {nodes[i]: i for i in range(len(nodes))}
    train_mask = [data.train_mask[node] for node in nodes]

    edge_index_new = subgraph(nodes, edge_index)[0]

    x = torch.stack([data.x[node] for node in nodes], dim=0)
    y = torch.stack([data.y[node] for node in nodes], dim=0)

    for i in range(edge_index_new.size(1)):
        edge_index_new[0][i] = torch.tensor(node2idx[edge_index_new[0][i].item()])
        edge_index_new[1][i] = torch.tensor(node2idx[edge_index_new[1][i].item()])

    train_mask = torch.tensor(train_mask, dtype=torch.bool)

    return Data(x=x, edge_index=edge_index_new, y=y, train_mask=train_mask)

def get_client_data_with_extend(data: Data, process_id, node_lists) -> Data:
    nodes = node_lists[process_id - 1]
    edge_index = data.edge_index

    subset, edge_index_new, _, _ = k_hop_subgraph(nodes, 2, edge_index, relabel_nodes=False)
    nodes = subset.tolist()
    
    node2idx = {nodes[i]: i for i in range(len(nodes))}
    train_mask = [data.train_mask[n] and n in node_lists[process_id - 1] for n in nodes]

    x = torch.stack([data.x[n] for n in nodes], dim=0)
    y = torch.stack([data.y[n] for n in nodes], dim=0)

    for i in range(edge_index_new.size(1)):
        edge_index_new[0][i] = torch.tensor(node2idx[edge_index_new[0][i].item()])
        edge_index_new[1][i] = torch.tensor(node2idx[edge_index_new[1][i].item()])

    train_mask = torch.tensor(train_mask, dtype=torch.bool)

    return Data(x=x, edge_index=edge_index_new, y=y, train_mask=train_mask)

def get_Laplace(data: Data, node_lists):
    Laplace = [{} for _ in range(len(node_lists))]
    L = get_adj(data.edge_index)

    edge_dict = {i: [] for i in range(data.num_nodes)}
    for i in range(edge_index.size(1)):
        a = edge_index[0][i].item()
        b = edge_index[1][i].item()
        edge_dict[a].append(b)
        edge_dict[b].append(a)

    for i, nodes in enumerate(node_lists):
        for node in nodes:
            Laplace[i][node] = {neighbor: L[node][neighbor].item() for neighbor in edge_dict[node]}
            Laplace[i][node][node] = L[node][node].item()

    return Laplace


def get_data_lp_withoutlink(data, train_data, process_id, nodes_lists):
    nodes = nodes_lists[process_id - 1]
    edge_index = data.edge_index
    graph = subgraph(nodes, edge_index)
    node2idx = {nodes[i]: i for i in range(len(nodes))}
    edge_index_new = graph[0]

    x = torch.stack([data.x[node] for node in nodes], dim=0)
    y = torch.stack([data.y[node] for node in nodes], dim=0)

    edge_label_index = train_data.edge_label_index
    edge_dict = {(edge_label_index[0][i].item(), edge_label_index[1][i].item()): 1
                 for i in range(edge_label_index.size(1))}
    edge_dict.update({(b, a): 1 for (a, b) in edge_dict.keys()})

    new_label_index1, new_label_index2 = [], []
    for i in range(edge_index_new.size(1)):
        a, b = edge_index_new[0][i].item(), edge_index_new[1][i].item()
        new_a, new_b = node2idx[a], node2idx[b]
        if edge_dict.get((a, b), 0) == 1:
            new_label_index1.append(new_a)
            new_label_index2.append(new_b)
            edge_dict[(a, b)] = edge_dict[(b, a)] = 0
        edge_index_new[0][i] = torch.tensor(new_a)
        edge_index_new[1][i] = torch.tensor(new_b)

    new_edge_label_index = torch.tensor([new_label_index1, new_label_index2], dtype=torch.long)
    edge_label = torch.ones(len(new_label_index1), dtype=torch.float)

    return Data(x=x, edge_index=edge_index_new, y=y,
                edge_label=edge_label, edge_label_index=new_edge_label_index)

def get_data_lp_withextend(data, train_data, process_id, nodes_lists):
    """
    data: 测试数据集
    train_data: 训练数据集
    """
    nodes = nodes_lists[process_id - 1]
    edge_index = data.edge_index
    # graph = subgraph(nodes, edge_index) # 此时没有重新编序号

    subset, edge_index_new, mapping, edge_mask = k_hop_subgraph(
        nodes, 2, edge_index, relabel_nodes=False)

    node2idx = {}  # 新旧节点的对应
    subset = list(subset)
    for i in range(len(subset)):
        subset[i] = subset[i].item()
    nodes = list(subset)

    for i in range(len(nodes)):
        node2idx[nodes[i]] = i

    # print(node2idx)

    x = None
    y = None
    # 重建 x y
    for node in nodes:
        if x == None:
            x = data.x[node].unsqueeze(0)
        else:
            x = torch.cat((x, data.x[node].unsqueeze(0)), dim=0)

        if y == None:
            y = data.y[node].unsqueeze(0)
        else:
            y = torch.cat((y, data.y[node].unsqueeze(0)), dim=0)

    edge_label_index = train_data.edge_label_index
    new_lebel_index1 = []
    new_label_index2 = []
    mydict = {}
    for i in range(len(edge_label_index[0])):
        a = edge_label_index[0][i].item()
        b = edge_label_index[1][i].item()
        mydict[(a, b)] = 1
        mydict[(b, a)] = 1

    for i in range(len(edge_index_new[0])):
        a = edge_index_new[0][i].item()
        b = edge_index_new[1][i].item()
        new_a = node2idx[a]
        new_b = node2idx[b]
        # if mydict[(a, b)] == 1 or mydict[(b, a)] == 1:
        if ((a, b) in mydict.keys() and mydict[(a, b)] == 1) or ((b, a) in mydict.keys() and mydict[(b, a)] == 1):
            new_lebel_index1.append(new_a)
            new_label_index2.append(new_b)
            mydict[(a, b)] = 0
            mydict[(b, a)] = 0
        edge_index_new[0][i] = torch.tensor(new_a)
        edge_index_new[1][i] = torch.tensor(new_b)

    new_edge_label_index = [new_lebel_index1, new_label_index2]
    new_edge_label_index = torch.Tensor(new_edge_label_index).long()

    length = len(new_edge_label_index[0])

    edge_label = [1] * length
    edge_label = torch.Tensor(edge_label)

    clientdata = Data(x=x, edge_index=edge_index_new, y=y,
                      edge_label=edge_label, edge_label_index=new_edge_label_index)

    return clientdata


if __name__ == '__main__':
    
    A = {}
    B = {}

    A[0] = torch.zeros(10)
    A[1] = torch.zeros(10)
    A[2] = torch.zeros(10)

    B[0] = [i for i in range(10)]
    B[1] = [i for i in range(10)]

    C = {}

    print(sys.getsizeof(A))
    print(sys.getsizeof(B))
    print(sys.getsizeof(C))


