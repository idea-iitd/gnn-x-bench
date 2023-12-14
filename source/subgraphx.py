from methods.SubGraphX.subgraphx import SubgraphX

import torch
import argparse
import random
import numpy as np
import os
from tqdm import tqdm

import data_utils
from gnn_trainer import GNNTrainer
from torch_geometric.data import Data
import torch_geometric.utils.subgraph as subgraph_func

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Mutagenicity',
                    choices=['Mutagenicity', 'Proteins', 'Mutag', 'IMDB-B', 'AIDS', 'NCI1', 'Tree-of-Life', 'Graph-SST2', 'DD', 'REDDIT-B'],
                    help="Dataset name")
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--gnn_run', type=int, default=1)
parser.add_argument('--explainer_run', type=int, default=1)
parser.add_argument('--gnn_type', type=str, default='gcn', choices=['gcn', 'gat', 'gin', 'sage'])
parser.add_argument('--explain_test_only', action='store_true')  # for scalability
parser.add_argument('--robustness', type=str, default='na', choices=['topology_random', 'topology_adversarial', 'feature', 'na'], help="na by default means we do not run for perturbed data")

args = parser.parse_args()

# Logging.
result_folder = f'data/{args.dataset}/subgraphx/'
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

best_explainer_model_path = os.path.join(result_folder, f'best_model_base_{args.gnn_type}_run_{args.gnn_run}_explainer_run_{args.explainer_run}.pt')
args.best_explainer_model_path = best_explainer_model_path
if not args.explain_test_only:
    explanations_path = os.path.join(result_folder, f'explanations_{args.gnn_type}_run_{args.explainer_run}.pt')
else:
    explanations_path = os.path.join(result_folder, f'explanations_{args.gnn_type}_run_{args.explainer_run}_test.pt')

trainer = GNNTrainer(dataset_name=args.dataset, gnn_type=args.gnn_type, task='basegnn', device=args.device)
model = trainer.load(args.gnn_run)
model.eval()


def get_explanations_from_subgraphx_results(explanation, graph):
    edge_map = {}
    edge_weights = torch.zeros(graph.edge_index.shape[1])
    for i, (a, b) in enumerate(graph.edge_index.T):
        edge_map[(a.item(), b.item())] = i

    for ex in explanation[:20]:  # get top 20 explanations and generate continuous edge weights
        nodes = ex['coalition']
        if nodes is not None and len(nodes) > 0 and len(set(nodes) - set(graph.edge_index.unique().tolist())) == 0:
            explanation_graph = subgraph_func(nodes, graph.edge_index)[0]
            for a, b in explanation_graph.T:
                edge_weights[edge_map[(a.item(), b.item())]] += 1.0

    edge_weights = edge_weights / edge_weights.sum()
    return edge_weights


if args.explain_test_only:
    data_indices = test_indices
else:
    data_indices = range(len(dataset))

if args.robustness == 'na':
    explanations = []
    for index in tqdm(data_indices):
        graph = dataset[index]
        subgraphx = SubgraphX(model=model, num_classes=2, device=args.device)
        _, explanation, related_preds = subgraphx(graph.x.to(device), graph.edge_index.to(device), graph.y.to(device))
        explanation_weights = get_explanations_from_subgraphx_results(explanation[0], graph)

        explanation_ = Data(
            edge_index=graph.edge_index.clone(),
            x=graph.x.clone(),
            y=graph.y.clone(),
            edge_weight=explanation_weights.clone()
        )
        explanations.append(explanation_)
    torch.save(explanations, explanations_path)
elif args.robustness == 'topology_random':
    for noise in [1, 2, 3, 4, 5]:
        if not args.explain_test_only:
            explanations_path = os.path.join(result_folder, f'explanations_{args.gnn_type}_run_{args.explainer_run}_noise_{noise}.pt')
        else:
            explanations_path = os.path.join(result_folder, f'explanations_{args.gnn_type}_run_{args.explainer_run}_noise_{noise}_test.pt')
        explanations = []
        noisy_dataset = data_utils.load_dataset(data_utils.get_noisy_dataset_name(dataset_name=args.dataset, noise=noise))
        for index in tqdm(data_indices):
            graph = noisy_dataset[index]
            subgraphx = SubgraphX(model=model, num_classes=2, device=args.device)
            _, explanation, related_preds = subgraphx(graph.x.to(device), graph.edge_index.to(device), graph.y.to(device))
            explanation_weights = get_explanations_from_subgraphx_results(explanation[0], graph)

            explanation_ = Data(
                edge_index=graph.edge_index.clone(),
                x=graph.x.clone(),
                y=graph.y.clone(),
                edge_weight=explanation_weights.clone()
            )
            explanations.append(explanation_)
        torch.save(explanations, explanations_path)
