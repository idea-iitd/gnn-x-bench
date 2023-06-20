import torch
import argparse
import random
import numpy as np
import os
import networkx as nx

import data_utils
from tqdm import tqdm

from gnn_trainer import GNNTrainer
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, to_dense_adj

"""
This is an adaptation of GEM code from: https://github.com/wanyu-lin/ICML2021-Gem/blob/main/generate_ground_truth_graph_classification.py
"""

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Mutagenicity',
                    choices=['Mutagenicity', 'Proteins', 'Mutag', 'IMDB-B', 'AIDS', 'NCI1', 'Tree-of-Life', 'Graph-SST2', 'DD', 'REDDIT-B'],
                    help="Dataset name")
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--gnn_run', type=int, default=1)
parser.add_argument('--explainer_run', type=int, default=1)
parser.add_argument('--gnn_type', type=str, default='gcn', choices=['gcn', 'gat', 'gin', 'sage'])
parser.add_argument('--robustness', action='store_true')
parser.add_argument('--top_k', type=int, default=25)

# we allow disconnected graphs

args = parser.parse_args()

# Logging.
result_folder = f'data/{args.dataset}/gem/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
dataset = data_utils.load_dataset(args.dataset)

torch.manual_seed(args.explainer_run)
torch.cuda.manual_seed(args.explainer_run)
np.random.seed(args.explainer_run)
random.seed(args.explainer_run)

trainer = GNNTrainer(dataset_name=args.dataset, gnn_type=args.gnn_type, task='basegnn', device=args.device)
model = trainer.load(args.gnn_run)
model.eval()


# generates ground truth for graphs
def graph_labeling(G):
    for node in G:
        G.nodes[node]['string'] = 1
    old_strings = tuple([G.nodes[node]['string'] for node in G])
    for iter_num in range(100):
        for node in G:
            string = sorted([G.nodes[neigh]['string'] for neigh in G.neighbors(node)])
            G.nodes[node]['concat_string'] = tuple([G.nodes[node]['string']] + string)
        d = nx.get_node_attributes(G, 'concat_string')
        nodes, strings = zip(*{k: d[k] for k in sorted(d, key=d.get)}.items())
        map_string = dict([[string, i + 1] for i, string in enumerate(sorted(set(strings)))])
        for node in nodes:
            G.nodes[node]['string'] = map_string[G.nodes[node]['concat_string']]
        new_strings = tuple([G.nodes[node]['string'] for node in G])
        if old_strings == new_strings:
            break
        else:
            old_strings = new_strings
    return G


def generate_gt(device, model):
    ce = torch.nn.CrossEntropyLoss(reduction='none')
    distillation_folder = os.path.join(result_folder, 'distillation')
    if not os.path.exists(distillation_folder):
        os.mkdir(distillation_folder)

    def change_graph_to_feat_adj(graph):
        feat = graph.x.float()
        if graph.edge_index.shape[1] == 0:
            adj = torch.zeros((graph.num_nodes, graph.num_nodes))
        else:
            adj = to_dense_adj(graph.edge_index, max_num_nodes=graph.num_nodes).float()
        return feat, adj

    def change_feat_adj_to_graph(feat, adj):
        graph = Data(x=feat, edge_index=torch.stack(torch.where(adj != 0)[1:]))
        return graph

    # for graph_idx in range(cg_dict["adj"].shape[0]):
    def run(graph, graph_idx, top_k):
        preds = model(graph.to(device))[-1]
        loss = ce(preds, graph.y.to(device))
        G = to_networkx(graph, to_undirected=True)
        top_k = max(3, top_k)
        top_k = min(top_k, len(G.edges))
        sorted_edges = sorted(G.edges)
        masked_loss = []
        edge_dict = np.zeros((len(G.nodes), len(G.nodes)), dtype=np.int64)
        feat, adj = change_graph_to_feat_adj(graph)
        if len(G.edges) > 0:

            for edge_idx, (x, y) in enumerate(sorted_edges):
                edge_dict[x, y] = edge_idx
                edge_dict[y, x] = edge_idx
                masked_adj = adj.clone()
                masked_adj[0, x, y] = 0
                masked_adj[0, y, x] = 0
                masked_graph = change_feat_adj_to_graph(feat, masked_adj)
                with torch.no_grad():
                    m_preds = model(masked_graph.to(device))[-1]
                m_loss = ce(m_preds, graph.y.to(device))
                masked_loss += [m_loss]
                G[x][y]['weight'] = (m_loss - loss).item()

            masked_loss = torch.stack(masked_loss)
            loss_diff = (masked_loss - loss).squeeze(-1)

            best_loss = loss.detach()
            masked_loss = []
            weights = loss_diff.detach().cpu().numpy()
            sub_G = G.copy()
            sorted_weight_idxs = np.argsort(weights)
            highest_weights = sum(weights)
            extracted_adj = np.zeros(adj.shape[1:])
            for idx, sorted_idx in enumerate(sorted_weight_idxs):
                sub_G.remove_edge(*sorted_edges[sorted_idx])
                masked_adj = torch.tensor(nx.to_numpy_matrix(sub_G, weight=None)).unsqueeze(0).float()
                masked_graph = change_feat_adj_to_graph(feat, masked_adj)
                with torch.no_grad():
                    m_preds = model(masked_graph.to(device))[-1]
                m_loss = ce(m_preds, graph.y.to(device))
                x, y = sorted_edges[sorted_idx]
                masked_loss += [m_loss]
                if m_loss > best_loss:
                    extracted_adj[x, y] = (m_loss - best_loss).item()
                    sub_G.add_edge(*sorted_edges[sorted_idx])
                else:
                    best_loss = m_loss
            masked_loss = torch.stack(masked_loss)
            loss_diff = (masked_loss - best_loss).squeeze(-1)

            G2 = nx.from_numpy_array(extracted_adj)
            d = nx.get_edge_attributes(G2, 'weight')

            if d and top_k is not None:
                edges, weights = zip(*{k: d[k] for k in sorted(d, key=d.get)}.items())
                weights = torch.tensor(weights)
                sub_G = G2.copy()
                sorted_weight_idxs = np.argsort(weights)
                highest_weights = sum(weights)
                for idx, sorted_idx in enumerate(sorted_weight_idxs):
                    sub_G.remove_edge(*edges[sorted_idx])
                    if sub_G.number_of_edges() < top_k:
                        sub_G.add_edge(*edges[sorted_idx])
                G3 = nx.Graph()
                G3.add_nodes_from(list(G2.nodes))
                G3.add_weighted_edges_from([[*e, d[e]] for e in sub_G.edges])
                extracted_adj = nx.to_numpy_matrix(G3)

            G = graph_labeling(G)
            graph_label = np.array([G.nodes[node]['string'] for node in G])
            save_dict = {
                'adj': adj.squeeze(0).cpu().detach().numpy(),
                "adj_y": extracted_adj,
                "mapping": np.asarray(list(G.nodes)),
                "label": graph.y.detach().cpu().numpy(),
                'features': feat.cpu().detach().numpy(),
                "graph_label": graph_label,
                'idx': graph_idx,
                "pred": preds.detach().cpu().numpy(),
            }
            assert highest_weights >= sum(weights)
        else:
            save_dict = {
                'adj': adj.squeeze(0).cpu().detach().numpy(),
                "adj_y": None,
                "mapping": np.asarray(list(G.nodes)),
                "label": graph.y.detach().cpu().numpy(),
                'features': feat.cpu().detach().numpy(),
                "graph_label": None,
                'idx': graph_idx,
                "pred": preds.detach().cpu().numpy(),
            }
        torch.save(save_dict, os.path.join(distillation_folder, f'graph_gt_{graph_idx}.pt'))

    for idx in tqdm(range(len(dataset))):
        run(dataset[idx], idx, args.top_k)


generate_gt(device, model)
