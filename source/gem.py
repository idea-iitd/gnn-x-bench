import torch
import argparse
import random
import numpy as np
import os
import time

import data_utils
from tqdm import tqdm
import torch.nn.functional as F
from torch import nn, optim
import sys

from torch_geometric.data import Data

from methods.GEM.gae.model import GCNModelVAE3
from methods.GEM.gae.optimizer import loss_function as gae_loss

"""
This is an adaptation of GEM code from: https://github.com/wanyu-lin/ICML2021-Gem/blob/main/explainer_gae_graph.py
"""

parser = argparse.ArgumentParser()
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='Mutagenicity',
                    choices=['Mutagenicity', 'Proteins', 'Mutag', 'IMDB-B', 'AIDS', 'NCI1', 'Tree-of-Life', 'Graph-SST2', 'DD', 'REDDIT-B'],
                    help="Dataset name")
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--gnn_run', type=int, default=1)
parser.add_argument('--explainer_run', type=int, default=1)
parser.add_argument('--gnn_type', type=str, default='gcn', choices=['gcn', 'gat', 'gin', 'sage'])
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--robustness', type=str, default='na', choices=['topology_random', 'topology_adversarial', 'feature', 'na'], help="na by default means we do not run for perturbed data")
parser.add_argument('--top_k', type=int, default=25)
parser.add_argument('--weighted', action='store_true')
parser.add_argument('--gae3', action='store_true')
parser.add_argument('--loss', type=str, default='mse')
parser.add_argument('--early_stop', action='store_true')
parser.add_argument('--train_on_positive_label', action='store_true')
parser.add_argument('--lr', type=float, default=0.01)

parser.add_argument('--exclude_non_label', action='store_true')
parser.add_argument('--label_feat', action='store_true')
parser.add_argument('--degree_feat', action='store_true')
parser.add_argument('--neigh_degree_feat', type=int, default=0, help='Number of neighbors\' degree.')
parser.add_argument('--normalize_feat', action='store_true')
parser.add_argument('--explain_class', type=int, default=None, help='Number of training epochs.')

# we allow disconnected graphs
# --weighted --gae3 --early_stop --normalize_feat --train_on_positive_label

args = parser.parse_args()
algo_conf = {
    "max_grad_norm": 1,
    "num_minibatch": 10
}

# Logging.
result_folder = f'data/{args.dataset}/gem/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
dataset = data_utils.load_dataset(args.dataset)
splits, indices = data_utils.split_data(dataset)
train_indices, val_indices, test_indices = indices

torch.manual_seed(args.explainer_run)
torch.cuda.manual_seed(args.explainer_run)
np.random.seed(args.explainer_run)
random.seed(args.explainer_run)

distillation_folder = os.path.join(result_folder, f'distillation_{args.gnn_type}_{args.gnn_run}')

distillations = []

for i in range(len(dataset)):
    distillation = torch.load(os.path.join(distillation_folder, f'graph_gt_{i}.pt'))
    if distillation['adj_y'] is None:  # could not find a ground truth
        # remove from splits
        if i in train_indices:
            train_indices.remove(i)
        if i in val_indices:
            val_indices.remove(i)
        if i in test_indices:
            test_indices.remove(i)
    distillations.append(distillation)

feat_dim = distillations[0]['features'].shape[-1]
preds = np.stack([distillations[i]['pred'][0] for i in range(len(dataset))])
preds = np.argmax(preds, axis=1)
labels = np.stack([distillations[i]['label'][0] for i in range(len(dataset))])

# Only train on samples with correct prediction
train_indices = np.array(train_indices)[preds[train_indices] == labels[train_indices]]

model = GCNModelVAE3(feat_dim, args.hidden1, args.hidden2, args.dropout).to(device)

if args.dataset == 'Graph-SST2':
    optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.05)
else:
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


class GraphSampler(torch.utils.data.Dataset):
    """ Sample graphs and nodes in graph
    """

    def __init__(
            self,
            graph_idxs,
            distillation_datas,
    ):
        self.graph_idxs = graph_idxs
        self.graph_data = [load_graph(distillation_datas[graph_idx], graph_idx) for graph_idx in graph_idxs]

    def __len__(self):
        return len(self.graph_idxs)

    def __getitem__(self, idx):
        return self.graph_data[idx]


def load_graph(distillation_data, graph_idx):
    """Returns the neighborhood of a given ndoe."""
    # data = torch.load("distillation/%s/graph_idx_%d.ckpt" % (args.distillation, graph_idx))
    data = distillation_data
    sub_adj = torch.from_numpy(np.int64(data['adj'] > 0)).float()
    adj_norm = preprocess_graph(sub_adj.numpy())
    sub_feat = torch.from_numpy(data['features'])
    if args.normalize_feat:
        sub_feat = F.normalize(sub_feat, p=2, dim=1)
    if args.degree_feat:
        degree_feat = torch.sum(sub_adj, dim=0).unsqueeze(1)
        sub_feat = torch.cat((sub_feat, degree_feat), dim=1)
    if args.neigh_degree_feat > 0:
        degree_feat = torch.sum(sub_adj, dim=0)
        neigh_degree = degree_feat.repeat(100, 1) * sub_adj
        v, _ = neigh_degree.sort(axis=1, descending=True)
        sub_feat = torch.cat((sub_feat, v[:, :args.neigh_degree_feat]), dim=1)
    sub_label = torch.from_numpy(data['label'])
    if args.weighted:
        sub_loss_diff = data['adj_y']
    else:
        sub_loss_diff = np.int64(data['adj_y'] > 0)
    sub_loss_diff = torch.from_numpy(sub_loss_diff).float()
    adj_label = sub_loss_diff + np.eye(sub_loss_diff.shape[0])
    n_nodes = sub_loss_diff.shape[0]
    pos_weight = float(sub_loss_diff.shape[0] * sub_loss_diff.shape[0] - sub_loss_diff.sum()) / sub_loss_diff.sum()
    pos_weight = torch.from_numpy(np.array(pos_weight))
    norm = torch.tensor(sub_loss_diff.shape[0] * sub_loss_diff.shape[0] / float((sub_loss_diff.shape[0] * sub_loss_diff.shape[0] - sub_loss_diff.sum()) * 2))
    return {
        "graph_idx": graph_idx,
        "sub_adj": sub_adj,
        "adj_norm": adj_norm.float(),
        "sub_feat": sub_feat.float(),
        "sub_label": sub_label.float(),
        "sub_loss_diff": sub_loss_diff.float(),
        "adj_label": adj_label.float(),
        "n_nodes": n_nodes,
        "pos_weight": pos_weight,
        "norm": norm
    }


def get_edges(adj_dict, edge_dict, node, hop, edges=set(), visited=set()):
    for neighbor in adj_dict[node]:
        edges.add(edge_dict[node, neighbor])
        visited.add(neighbor)
    if hop <= 1:
        return edges, visited
    for neighbor in adj_dict[node]:
        edges, visited = get_edges(adj_dict, edge_dict, neighbor, hop - 1, edges, visited)
    return edges, visited


def preprocess_graph(adj):
    adj_ = adj + np.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = np.diag(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    return torch.from_numpy(adj_normalized).float()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# MSE
def mse(x, mu, logvar, data):
    return F.mse_loss(x.view(x.shape[0], -1), data['adj_label'].view(x.shape[0], -1).to(device))


# GVAE
def gaeloss(x, mu, logvar, data):
    return gae_loss(preds=x, labels=data['adj_label'].to(device),
                    mu=mu, logvar=logvar, n_nodes=data['n_nodes'].to(device),
                    norm=data['norm'].to(device), pos_weight=data['pos_weight'].to(device))


def eval_model(dataset):
    with torch.no_grad():
        losses = []
        for data in dataset:
            recovered, mu, logvar = model(data['sub_feat'].to(device), data['adj_norm'].to(device))
            loss = criterion(recovered, mu, logvar, data)
            losses += [loss.view(-1)]
    return (torch.cat(losses)).mean().item()


if args.loss == 'mse':
    criterion = mse
elif args.loss == 'gae':
    criterion = gaeloss
else:
    raise ("Loss function %s is not implemented" % args.loss)

best_explainer_model_path = os.path.join(result_folder, f'best_model_base_{args.gnn_type}_run_{args.gnn_run}_explainer_run_{args.explainer_run}.pt')
args.best_explainer_model_path = best_explainer_model_path
explanations_path = os.path.join(result_folder, f'explanations_{args.gnn_type}_run_{args.explainer_run}.pt')

if args.robustness == 'na':
    start_epoch = 1
    train_graphs = GraphSampler(train_indices, distillations)
    train_dataset = torch.utils.data.DataLoader(
        train_graphs,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    val_graphs = GraphSampler(val_indices, distillations)
    val_dataset = torch.utils.data.DataLoader(
        val_graphs,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    test_graphs = GraphSampler(test_indices, distillations)
    test_dataset = torch.utils.data.DataLoader(
        test_graphs,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model.train()
    start_time = time.time()
    best_loss = 10000
    patience = 15
    current_patience = 0
    for epoch in tqdm(range(start_epoch, args.epochs + 1)):
        train_losses = []
        for batch_idx, data in enumerate(train_dataset):
            optimizer.zero_grad()
            recovered, mu, logvar = model(data['sub_feat'].to(device), data['adj_norm'].to(device))
            loss = criterion(recovered, mu, logvar, data)
            loss.mean().backward()
            nn.utils.clip_grad_norm_(model.parameters(), algo_conf['max_grad_norm'])
            optimizer.step()
            train_losses += [loss.view(-1)]
            sys.stdout.flush()

        train_loss = (torch.cat(train_losses)).mean().item()
        val_loss = eval_model(val_dataset)
        if args.early_stop and val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_explainer_model_path)
            current_patience = 0
        else:
            current_patience += 1
            if current_patience > patience:
                break

    checkpoint = torch.load(best_explainer_model_path, map_location=device)
    model.load_state_dict(checkpoint)

    model.eval()

    with torch.no_grad():
        explanations = []
        for idx in tqdm(range(len(dataset))):
            graph_pyg = dataset[idx]
            distillation = distillations[idx]
            if distillation['adj_y'] is None:
                explanation = Data(x=graph_pyg.x.clone(), edge_index=graph_pyg.edge_index.clone(), edge_weight=torch.ones(graph_pyg.edge_index.shape[1]))
            else:
                data = load_graph(distillation, idx)
                sub_adj = data['sub_adj']
                adj_norm = data['adj_norm']
                sub_feat = data['sub_feat']
                sub_loss_diff = data['sub_loss_diff']
                recovered, mu, logvar = model(sub_feat.unsqueeze(0).to(device), adj_norm.unsqueeze(0).to(device))
                recovered = recovered.squeeze(0)
                graph_pyg = dataset[data['graph_idx']]
                explanation = Data(x=graph_pyg.x.clone(), edge_index=graph_pyg.edge_index.clone(), edge_weight=recovered[graph_pyg.edge_index[0], graph_pyg.edge_index[1]].detach().cpu().clone())
            explanations.append(explanation)
        torch.save(explanations, explanations_path)
elif args.robustness == 'topology_random':
    checkpoint = torch.load(best_explainer_model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    with torch.no_grad():
        for noise in [1, 2, 3, 4, 5]:
            distillation_noise_path = os.path.join(result_folder, f'distillation_{args.gnn_type}_{args.gnn_run}_noise_{noise}')
            distillations = [torch.load(os.path.join(distillation_noise_path, f'graph_gt_{i}.pt')) for i in range(len(dataset))]
            explanations_path = os.path.join(result_folder, f'explanations_{args.gnn_type}_run_{args.explainer_run}_noise_{noise}.pt')
            explanations = []
            noisy_dataset = data_utils.load_dataset(data_utils.get_noisy_dataset_name(dataset_name=args.dataset, noise=noise))
            for idx in tqdm(range(len(noisy_dataset))):
                distillation = distillations[idx]
                if distillation['adj_y'] is None:
                    explanation = Data(x=graph_pyg.x.clone(), edge_index=graph_pyg.edge_index.clone(), edge_weight=torch.ones(graph_pyg.edge_index.shape[1]))
                else:
                    data = load_graph(distillation, idx)
                    sub_adj = data['sub_adj']
                    adj_norm = data['adj_norm']
                    sub_feat = data['sub_feat']
                    sub_loss_diff = data['sub_loss_diff']
                    recovered, mu, logvar = model(sub_feat.unsqueeze(0).to(device), adj_norm.unsqueeze(0).to(device))
                    recovered = recovered.squeeze(0)
                    graph_pyg = noisy_dataset[data['graph_idx']]
                    explanation = Data(x=graph_pyg.x.clone(), edge_index=graph_pyg.edge_index.clone(), edge_weight=recovered[graph_pyg.edge_index[0], graph_pyg.edge_index[1]].detach().cpu().clone())
                explanations.append(explanation)
            torch.save(explanations, explanations_path)
