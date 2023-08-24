import torch
import argparse
import random
import numpy as np
import os

import data_utils
from gnn_trainer import GNNTrainer
from torch_geometric.data import Data
from methods.PGExplainer.explainers.PGExplainer import PGExplainer

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
# 64 is used for social datasets (IMDB) and 16 or 32 for bio datasest (MUTAG, PTC, PROTEINS).
parser.add_argument('--hidden_units', type=int, default=64)
parser.add_argument('--dataset', type=str, default='Mutagenicity',
                    choices=['Mutagenicity', 'Proteins', 'Mutag', 'IMDB-B', 'AIDS', 'NCI1', 'Tree-of-Life', 'Graph-SST2', 'DD', 'REDDIT-B', 'ogbg_molhiv'],
                    help="Dataset name")
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--gnn_run', type=int, default=1)
parser.add_argument('--explainer_run', type=int, default=1)
parser.add_argument('--gnn_type', type=str, default='gcn', choices=['gcn', 'gat', 'gin', 'sage'])
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--robustness', action='store_true')

args = parser.parse_args()

# Logging.
result_folder = f'data/{args.dataset}/pgexplainer/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
dataset = data_utils.load_dataset(args.dataset)
splits, indices = data_utils.split_data(dataset)

torch.manual_seed(args.explainer_run)
torch.cuda.manual_seed(args.explainer_run)
np.random.seed(args.explainer_run)
random.seed(args.explainer_run)

best_explainer_model_path = os.path.join(result_folder, f'best_model_base_{args.gnn_type}_run_{args.gnn_run}_explainer_run_{args.explainer_run}.pt')
args.best_explainer_model_path = best_explainer_model_path
explanations_path = os.path.join(result_folder, f'explanations_{args.gnn_type}_run_{args.explainer_run}.pt')
args.method = 'classification'

if args.dataset in ['IMDB-B', 'REDDIT-B']:
    args.hidden_units = 64
else:
    args.hidden_units = 32

if args.dataset in ['Graph-SST2', 'ogbg_molhiv']:
    lr = 0.001 * 0.05  # smaller lr for large dataset
else:
    lr = 0.001


trainer = GNNTrainer(dataset_name=args.dataset, gnn_type=args.gnn_type, task='basegnn', device=args.device)
model = trainer.load(args.gnn_run)
model.eval()

node_embeddings, graph_embeddings, outs = trainer.load_gnn_outputs(args.gnn_run)

train_indices = indices[0]
val_indices = indices[1]

explainer = PGExplainer(model, dataset, node_embeddings, task='graph', device=device, save_folder=result_folder, args=args, reg_coefs=(0.00001, 0.0), lr=lr)

if not args.robustness:
    explainer.prepare(train_indices=train_indices, val_indices=val_indices, start_training=True)
    explanation_graphs = []
    for i in range(len(dataset)):
        graph = dataset[i]
        explanation = explainer.explain(i)
        explanation_graphs.append(Data(
            edge_index=graph.edge_index.clone(),
            x=graph.x.clone(),
            y=graph.y.clone(),
            edge_weight=explanation.detach().cpu().clone()
        ))
    torch.save(explanation_graphs, explanations_path)
else:
    explainer.prepare(train_indices=train_indices, val_indices=val_indices, start_training=False)
    explainer.explainer_model.load_state_dict(torch.load(args.best_explainer_model_path, map_location=device))
    for noise in [1, 2, 3, 4, 5]:
        explanations_path = os.path.join(result_folder, f'explanations_{args.gnn_type}_run_{args.explainer_run}_noise_{noise}.pt')
        explanation_graphs = []
        noisy_dataset = data_utils.load_dataset(data_utils.get_noisy_dataset_name(dataset_name=args.dataset, noise=noise))
        for i in range(len(dataset)):
            noisy_graph = noisy_dataset[i].to(device)
            explanation = explainer.explain_graph(noisy_graph)
            explanation_graphs.append(Data(
                edge_index=noisy_graph.edge_index.clone(),
                x=noisy_graph.x.clone(),
                y=noisy_graph.y.clone(),
                edge_weight=explanation.detach().cpu().clone()
            ))
        torch.save(explanation_graphs, explanations_path)




