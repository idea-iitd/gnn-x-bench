from methods.PGExplainer.explainers.GNNExplainer import GNNExplainer

import torch
import argparse
import random
import numpy as np
import os
from tqdm import tqdm

import data_utils
from gnn_trainer import GNNTrainer
from torch_geometric.data import Data

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--dataset', type=str, default='Mutagenicity',
                    choices=['Mutagenicity', 'Proteins', 'Mutag', 'IMDB-B', 'AIDS', 'NCI1', 'Tree-of-Life', 'Graph-SST2', 'DD', 'REDDIT-B'],
                    help="Dataset name")
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--gnn_run', type=int, default=1)
parser.add_argument('--explainer_run', type=int, default=1)
parser.add_argument('--gnn_type', type=str, default='gcn', choices=['gcn', 'gat', 'gin', 'sage'])
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--robustness', type=str, default='na', choices=['topology_random', 'topology_adversarial', 'feature', 'na'], help="na by default means we do not run for perturbed data")

args = parser.parse_args()

# Logging.
result_folder = f'data/{args.dataset}/gnnexplainer/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
dataset = data_utils.load_dataset(args.dataset)
splits, indices = data_utils.split_data(dataset)

torch.manual_seed(args.explainer_run)
torch.cuda.manual_seed(args.explainer_run)
np.random.seed(args.explainer_run)
random.seed(args.explainer_run)

explanations_path = os.path.join(result_folder, f'explanations_{args.gnn_type}_run_{args.explainer_run}.pt')

trainer = GNNTrainer(dataset_name=args.dataset, gnn_type=args.gnn_type, task='basegnn', device=args.device)
model = trainer.load(args.gnn_run)
model.eval()

if args.robustness == 'na':
    data_indices = range(len(dataset))
    explanations = []
    for index in tqdm(data_indices):
        graph = dataset[index]
        explainer = GNNExplainer(model, dataset, task='graph', device=device, epochs=args.epochs)
        explanation = explainer.explain(index)
        explanation_graph = Data(
            edge_index=graph.edge_index.clone(),
            x=graph.x.clone(),
            y=graph.y.clone(),
            edge_weight=explanation.detach().clone()
        )
        explanations.append(explanation_graph)
    torch.save(explanations, explanations_path)
elif args.robustness == 'topology_random':
    for noise in [1, 2, 3, 4, 5]:
        explanations_path = os.path.join(result_folder, f'explanations_{args.gnn_type}_run_{args.explainer_run}_noise_{noise}.pt')
        explanations = []
        noisy_dataset = data_utils.load_dataset(data_utils.get_noisy_dataset_name(dataset_name=args.dataset, noise=noise))
        data_indices = range(len(dataset))
        for index in tqdm(data_indices):
            noisy_graph = noisy_dataset[index]
            explainer = GNNExplainer(model, noisy_dataset, task='graph', device=device, epochs=args.epochs)
            explanation = explainer.explain(index)
            explanation_graph = Data(
                edge_index=noisy_graph.edge_index.clone(),
                x=noisy_graph.x.clone(),
                y=noisy_graph.y.clone(),
                edge_weight=explanation.detach().clone()
            )
            explanations.append(explanation_graph)
        torch.save(explanations, explanations_path)
elif args.robustness == 'feature':
    for noise in [10, 20, 30, 40, 50]:
        explanations_path = os.path.join(result_folder, f'explanations_{args.gnn_type}_run_{args.explainer_run}_feature_noise_{noise}.pt')
        explanations = []
        noisy_dataset = data_utils.load_dataset(data_utils.get_noisy_dataset_name(dataset_name=args.dataset, noise=noise))
        data_indices = range(len(dataset))
        for index in tqdm(data_indices):
            noisy_graph = noisy_dataset[index]
            explainer = GNNExplainer(model, noisy_dataset, task='graph', device=device, epochs=args.epochs)
            explanation = explainer.explain(index)
            explanation_graph = Data(
                edge_index=noisy_graph.edge_index.clone(),
                x=noisy_graph.x.clone(),
                y=noisy_graph.y.clone(),
                edge_weight=explanation.detach().clone()
            )
            explanations.append(explanation_graph)
        torch.save(explanations, explanations_path)
elif args.robustness == 'topology_adversarial':
    for flip_count in [1, 2, 3, 4, 5]:
        explanations_path = os.path.join(result_folder, f'explanations_{args.gnn_type}_run_{args.explainer_run}_topology_adversarial_{flip_count}.pt')
        explanations = []
        noisy_dataset = data_utils.load_dataset(data_utils.get_topology_adversarial_attack_dataset_name(dataset_name=args.dataset, flip_count=flip_count))
        data_indices = range(len(dataset))
        for index in tqdm(data_indices):
            noisy_graph = noisy_dataset[index]
            explainer = GNNExplainer(model, noisy_dataset, task='graph', device=device, epochs=args.epochs)
            explanation = explainer.explain(index)
            explanation_graph = Data(
                edge_index=noisy_graph.edge_index.clone(),
                x=noisy_graph.x.clone(),
                y=noisy_graph.y.clone(),
                edge_weight=explanation.detach().clone()
            )
            explanations.append(explanation_graph)
        torch.save(explanations, explanations_path)
else:
    raise NotImplementedError()
