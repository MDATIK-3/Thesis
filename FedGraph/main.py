import sys, time, torch, os
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import roc_auc_score

# ========== PATH FIX FOR KAGGLE ==========
KAGGLE_PATH = '/kaggle/working/Thesis' if 'kaggle' in os.getcwd() else os.path.dirname(os.path.abspath(__file__))
sys.path.append(KAGGLE_PATH)

from data_loader import get_data, load_partition_data
from Arguments import Arguments
from client import clientSP
from trainersp import TrainerSP
from gcnlp import GCNLP
from data_update import get_data_lp_withoutlink  # IMPORTANT: ensure exists

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

    # ========== LOAD DATA ==========
    data = get_data(args.name)

    split = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=1.0
    )

    train_data, val_data, test_data = split(data)

    data = data.to(device)
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    # ========== GRAPH PARTITION ==========
    _, node_lists = load_partition_data(args.client_num, data)
    client_num_list = [len(nodes) for nodes in node_lists]
    total_nodes = sum(client_num_list)

    # ========== SERVER MODEL ==========
    server_model = GCNLP(data.num_node_features, 128, 64).to(device)

    # ========== CLIENT ==========
    model = GCNLP(data.num_node_features, 128, 64).to(device)
    trainer = TrainerSP(args=args, model=model, data=None)
    client = clientSP(args=args, trainer=trainer)

    auc_list = []

    # ========== FEDERATED TRAINING LOOP ==========
    for r in range(args.comm_round):
        local_model = server_model.state_dict()
        agg_model = None

        for i in range(args.client_num):
            client_data = get_data_lp_withoutlink(data, train_data, i + 1, node_lists)
            client.trainer.set_data(client_data)
            client.trainer.set_model(local_model)

            model_params, _ = client.train()

            weight = client_num_list[i] / total_nodes
            if agg_model is None:
                agg_model = {k: model_params[k] * weight for k in model_params}
            else:
                for k in agg_model:
                    agg_model[k] += model_params[k] * weight

        server_model.load_state_dict(agg_model)
        server_model.eval()

        with torch.no_grad():
            z = server_model.encode(test_data)
            out = server_model.decode(z, test_data.edge_label_index).sigmoid().view(-1)
            auc = roc_auc_score(test_data.edge_label.cpu().numpy(), out.cpu().numpy())
            auc_list.append(auc)

    max_auc = max(auc_list)
    run_time = time.time() - start_time

    with open('output.txt', 'a') as f:
        f.write(f"{args.client_num} {args.epoch} {args.comm_round} {max_auc:.4f} {args.name} {run_time:.2f}\n")
