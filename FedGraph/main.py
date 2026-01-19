import sys, time, torch, numpy as np
from torch_geometric.transforms import RandomLinkSplit as T_RandomLinkSplit
from sklearn.metrics import roc_auc_score

sys.path.append('/kaggle/working/Thesis')  # Adjust to your repo path
from data_loader import get_data, load_partition_data
from Arguments import Arguments
from client import clientSP
from trainersp import TrainerSP
from gcnlp import GCNLP

import argparse

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--clientnum', type=int)
    parser.add_argument('-r', '--round', type=int)
    parser.add_argument('-e', '--epoch', type=int)
    parser.add_argument('-d', '--name', type=str)
    x = parser.parse_args()
    args = {
        "client_num": x.clientnum,
        "comm_round": x.round,
        "epoch": x.epoch,
        "lr": 0.01,
        "worker_num": x.clientnum,
        "name": x.name
    }
    return Arguments(args)

if __name__ == '__main__':
    start_time = time.time()
    args = init_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data = get_data(args.name)

    split = T_RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=1.0
    )
    train_data, val_data, test_data = split(data)
    data.to(device)
    train_data.to(device)
    test_data.to(device)

    _, node_lists = load_partition_data(args.client_num, data)

    client_num_list = [len(nodes) for nodes in node_lists]
    total_nodes = sum(client_num_list)

    server_model = GCNLP(data.num_node_features, 128, 64).to(device)
    model = GCNLP(data.num_node_features, 128, 64).to(device)
    trainer = TrainerSP(args=args, model=model, data=None)
    client = clientSP(args=args, trainer=trainer)

    auc_list = []

    for r in range(args.comm_round):
        local_model = server_model.state_dict()
        all_model = None

        for i in range(args.client_num):
            clientdata = get_data_lp_withoutlink(data, train_data, i + 1, node_lists)
            client.trainer.set_data(clientdata)
            client.trainer.set_model(local_model)
            model_params, _ = client.train()
            if all_model is None:
                all_model = model_params
            else:
                for key in all_model.keys():
                    all_model[key] += model_params[key] * client_num_list[i] / total_nodes

        server_model.load_state_dict(all_model)
        server_model.eval()

        z = model.encode(test_data)
        out = model.decode(z, test_data.edge_label_index).view(-1).sigmoid()
        auc = roc_auc_score(test_data.edge_label.detach().cpu().numpy(),
                            out.detach().cpu().numpy())
        auc_list.append(auc)

    max_auc = max(auc_list)

    run_time = time.time() - start_time
    with open('output.txt', 'a') as f:
        f.write(f"{args.client_num} {args.epoch} {args.comm_round} {max_auc} {args.name} {run_time}\n")
