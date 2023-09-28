import torch
import argparse
import random
import numpy as np
import os

from torch_geometric.loader import DataLoader
import data_utils
from gnn_trainer import GNNTrainer
from torch_geometric.data import Data
from methods.TAGE.tagexplainer import TAGExplainer, MLPExplainer
from methods.TAGE.downstream import train_MLP, MLP
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Mutagenicity',
                    choices=['Mutagenicity', 'Proteins', 'Mutag', 'IMDB-B', 'AIDS', 'NCI1', 'Tree-of-Life', 'Graph-SST2', 'DD', 'REDDIT-B', 'ogbg_molhiv'],
                    help="Dataset name")
parser.add_argument('--random_seed', type=int, default=0)

parser.add_argument('--batch_size', type=int, default=128, help='Batch size. Default is 128.')
parser.add_argument('--epochs', type=int, default=1, help='Number of epochs. Default is 1.')

parser.add_argument('--device', type=str, default='0')
parser.add_argument('--gnn_run', type=int, default=1)
parser.add_argument('--explainer_run', type=int, default=1)
parser.add_argument('--gnn_type', type=str, default='gcn', choices=['gcn', 'gat', 'gin', 'sage'], help='GNN layer type to use.')
parser.add_argument('--robustness', type=str, default='na', choices=['topology_random', 'topology_adversarial', 'feature', 'na'], help="na by default means we do not run for perturbed data")
parser.add_argument('--stage', type=int, default=2, help='Stage to run. Default is 2. 1 is embedding explainer, 2 is embedding explainer+downstream training.')

args = parser.parse_args()

# Logging.
result_folder = f'data/{args.dataset}/tagexplainer_{args.stage}/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
dataset = data_utils.load_dataset(args.dataset)
splits, indices = data_utils.split_data(dataset)
train_set, valid_set, test_set = splits

torch.manual_seed(args.explainer_run)
torch.cuda.manual_seed(args.explainer_run)
np.random.seed(args.explainer_run)
random.seed(args.explainer_run)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=0)

best_explainer_model_path = os.path.join(result_folder, f'best_model_base_{args.gnn_type}_run_{args.gnn_run}_explainer_run_{args.explainer_run}.pt')
args.best_explainer_model_path = best_explainer_model_path
best_downstream_mlp_model_path = os.path.join(result_folder, f'best_model_base_{args.gnn_type}_run_{args.gnn_run}_explainer_run_{args.explainer_run}_downstream.pt')
args.best_downstream_mlp_model_path = best_downstream_mlp_model_path
explanations_path = os.path.join(result_folder, f'explanations_{args.gnn_type}_run_{args.explainer_run}.pt')
if args.dataset in ['Tree-of-Life']:
    args.method = 'regression'
else:
    args.method = 'classification'

if args.dataset in ['Graph-SST2']:
    # Graph-SST2 is a large dataset, so we use a smaller learning rate.
    lr = 0.001 * 0.05
else:
    lr = 0.001

trainer = GNNTrainer(dataset_name=args.dataset, gnn_type=args.gnn_type, task='basegnn', device=args.device)
model = trainer.load(args.gnn_run)
model.eval()

embedding_explainer = TAGExplainer(model, trainer.dim, device=device, grad_scale=0.2, coff_size=0.05, coff_ent=0.002, loss_type='JSE')
if args.stage == 2:
    mlp = train_MLP(model, trainer.dim, device, train_loader, valid_loader, save_to=args.best_downstream_mlp_model_path)
    mlp_explainer = MLPExplainer(mlp, device=device)

if args.robustness == 'na':
    embedding_explainer.train_explainer_graph(train_loader, epochs=args.epochs, lr=lr)
    torch.save(embedding_explainer.explainer.state_dict(), args.best_explainer_model_path)
    explanation_graphs = []
    embedding_explainer.eval()
    for i in tqdm(range(len(dataset))):
        graph = dataset[i].to(device)
        if args.stage == 1:
            with torch.no_grad():
                node_embed, _, _ = model(graph)
                _, _, explanation = embedding_explainer.explain(graph, node_embed, training=False)
        elif args.stage == 2:
            explanation = embedding_explainer(graph, mlp_explainer)
        else:
            raise NotImplementedError
        explanation_graphs.append(Data(
            edge_index=graph.edge_index.clone(),
            x=graph.x.clone(),
            y=graph.y.clone(),
            edge_weight=explanation.detach().clone()
        ))
    torch.save(explanation_graphs, explanations_path)
elif args.robustness == 'topology_random':
    # load trained explainers
    embedding_explainer.explainer.load_state_dict(torch.load(args.best_explainer_model_path, map_location=device))
    embedding_explainer.eval()
    if args.stage == 2:
        mlp = MLP(2, trainer.dim, trainer.dim).to(device)
        mlp.load_state_dict(torch.load(args.best_downstream_mlp_model_path, map_location=device))
        mlp_explainer = MLPExplainer(mlp, device=device)
        mlp_explainer.eval()
    for noise in [1, 2, 3, 4, 5]:
        explanations_path = os.path.join(result_folder, f'explanations_{args.gnn_type}_run_{args.explainer_run}_noise_{noise}.pt')
        explanation_graphs = []
        noisy_dataset = data_utils.load_dataset(data_utils.get_noisy_dataset_name(dataset_name=args.dataset, noise=noise))
        for i in tqdm(range(len(dataset))):
            noisy_graph = noisy_dataset[i].to(device)
            if args.stage == 1:
                with torch.no_grad():
                    node_embed, _, _ = model(noisy_graph)
                    _, _, explanation = embedding_explainer.explain(noisy_graph, node_embed, training=False)
            elif args.stage == 2:
                explanation = embedding_explainer(noisy_graph, mlp_explainer)
            else:
                raise NotImplementedError
            explanation_graphs.append(Data(
                edge_index=noisy_graph.edge_index.clone(),
                x=noisy_graph.x.clone(),
                y=noisy_graph.y.clone(),
                edge_weight=explanation.detach().clone()
            ))
        torch.save(explanation_graphs, explanations_path)
elif args.robustness == 'feature':
    # load trained explainers
    embedding_explainer.explainer.load_state_dict(torch.load(args.best_explainer_model_path, map_location=device))
    embedding_explainer.eval()
    if args.stage == 2:
        mlp = MLP(2, trainer.dim, trainer.dim).to(device)
        mlp.load_state_dict(torch.load(args.best_downstream_mlp_model_path, map_location=device))
        mlp_explainer = MLPExplainer(mlp, device=device)
        mlp_explainer.eval()
    for noise in [10, 20, 30, 40, 50]:
        explanations_path = os.path.join(result_folder, f'explanations_{args.gnn_type}_run_{args.explainer_run}_feature_noise_{noise}.pt')
        explanation_graphs = []
        noisy_dataset = data_utils.load_dataset(data_utils.get_noisy_dataset_name(dataset_name=args.dataset, noise=noise))
        for i in tqdm(range(len(dataset))):
            noisy_graph = noisy_dataset[i].to(device)
            if args.stage == 1:
                with torch.no_grad():
                    node_embed, _, _ = model(noisy_graph)
                    _, _, explanation = embedding_explainer.explain(noisy_graph, node_embed, training=False)
            elif args.stage == 2:
                explanation = embedding_explainer(noisy_graph, mlp_explainer)
            else:
                raise NotImplementedError
            explanation_graphs.append(Data(
                edge_index=noisy_graph.edge_index.clone(),
                x=noisy_graph.x.clone(),
                y=noisy_graph.y.clone(),
                edge_weight=explanation.detach().clone()
            ))
        torch.save(explanation_graphs, explanations_path)
elif args.robustness == 'topology_adversarial':
    # load trained explainers
    embedding_explainer.explainer.load_state_dict(torch.load(args.best_explainer_model_path, map_location=device))
    embedding_explainer.eval()
    if args.stage == 2:
        mlp = MLP(2, trainer.dim, trainer.dim).to(device)
        mlp.load_state_dict(torch.load(args.best_downstream_mlp_model_path, map_location=device))
        mlp_explainer = MLPExplainer(mlp, device=device)
        mlp_explainer.eval()
    for flip_count in [1, 2, 3, 4, 5]:
        explanations_path = os.path.join(result_folder, f'explanations_{args.gnn_type}_run_{args.explainer_run}_topology_adversarial_{flip_count}.pt')
        explanation_graphs = []
        noisy_dataset = data_utils.load_dataset(data_utils.get_topology_adversarial_attack_dataset_name(dataset_name=args.dataset, flip_count=flip_count))
        for i in tqdm(range(len(dataset))):
            noisy_graph = noisy_dataset[i].to(device)
            if args.stage == 1:
                with torch.no_grad():
                    node_embed, _, _ = model(noisy_graph)
                    _, _, explanation = embedding_explainer.explain(noisy_graph, node_embed, training=False)
            elif args.stage == 2:
                explanation = embedding_explainer(noisy_graph, mlp_explainer)
            else:
                raise NotImplementedError
            explanation_graphs.append(Data(
                edge_index=noisy_graph.edge_index.clone(),
                x=noisy_graph.x.clone(),
                y=noisy_graph.y.clone(),
                edge_weight=explanation.detach().clone()
            ))
        torch.save(explanation_graphs, explanations_path)
else:
    raise NotImplementedError
