import torch
import argparse
import random
import numpy as np
import os
from tqdm import tqdm
import math

import data_utils
from gnn_trainer import GNNTrainer
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

"""
This is an adaptation of CFF code from: https://github.com/chrisjtan/gnn_cff
"""


class ExplainModelGraph(torch.nn.Module):

    def __init__(self, graph, device):
        super(ExplainModelGraph, self).__init__()
        self.graph = graph
        self.num_nodes = graph.num_nodes
        self.device = device

        self.adj_mask = self.construct_adj_mask()

    def forward(self, base_model, masked_adj=None):
        if masked_adj is None:
            masked_adj = self.get_masked_adj()
        pred1 = base_model(self.graph, masked_adj)[-1][0][0].exp()  # Probability of using explainatory subgraph to be original class 0, should be maximized.
        pred2 = base_model(self.graph, 1 - masked_adj)[-1][0][0].exp()  # Probability of using the residual graph to be original class 0, should be minimized.
        return pred1, pred2

    def loss(self, pred1, pred2, gam, lam, alp):
        relu = torch.nn.ReLU()
        bpr1 = relu(gam + 0.5 - pred1)  # factual
        bpr2 = relu(gam + pred2 - 0.5)  # counterfactual
        masked_adj = self.get_masked_adj()
        L1 = torch.linalg.norm(masked_adj, ord=1)
        loss = L1 + lam * (alp * bpr1 + (1 - alp) * bpr2)
        return bpr1, bpr2, L1, loss

    def construct_adj_mask(self):
        mask = torch.nn.Parameter(torch.FloatTensor(self.num_nodes, self.num_nodes))
        std = torch.nn.init.calculate_gain("relu") * math.sqrt(
            2.0 / (self.num_nodes + self.num_nodes)
        )
        with torch.no_grad():
            mask.normal_(1.0, std)
        return mask

    def get_masked_adj(self):
        sym_mask = torch.sigmoid(self.adj_mask)
        sym_mask = (sym_mask + sym_mask.t()) / 2
        masked_adj = sym_mask[tuple(self.graph.edge_index)]

        return masked_adj


class GraphExplainerEdge(torch.nn.Module):

    def __init__(self, base_model, G_dataset, args, device):

        super(GraphExplainerEdge, self).__init__()
        self.base_model = base_model
        self.G_dataset = G_dataset
        self.args = args
        self.device = device

    def explain_dataset(self):

        cfs = []
        exps = []
        total_sufficiency = 0
        total_necessity = 0
        total_size = 0
        for g in tqdm(self.G_dataset, desc='Graph'):
            g = g.to(self.device)
            masked_adj, binarized_mask, sufficiency, necessity, size = self.explain(g)
            cf = Data(edge_index=g.edge_index.clone().cpu(),
                      edge_weight=1 - masked_adj.clone().cpu(),
                      pred=torch.tensor(necessity),
                      num_nodes=g.num_nodes,
                      x=g.x.clone().cpu())
            cfs.append(cf)
            total_sufficiency += sufficiency
            total_necessity += necessity
            total_size += size
            exp = Data(edge_index=g.edge_index.clone().cpu(),
                       edge_weight=masked_adj.clone().cpu(),
                       y=g.y.clone(),
                       num_nodes=g.num_nodes,
                       x=g.x.clone().cpu())
            exps.append(exp)

        return exps, cfs, total_sufficiency / len(cfs), total_necessity / len(cfs), total_size / len(cfs)

    def explain(self, g):
        explainer = ExplainModelGraph(
            graph=g,
            device=self.device
        ).to(self.device)

        # train explainer
        optimizer = torch.optim.Adam(explainer.parameters(), lr=self.args.lr, weight_decay=0)
        explainer.train()
        for epoch in range(1, self.args.epochs + 1):
            optimizer.zero_grad()
            pred1, pred2 = explainer(self.base_model)

            bpr1, bpr2, l1, loss = explainer.loss(
                pred1, pred2, self.args.gam, self.args.lam, self.args.alp)

            loss.backward()
            optimizer.step()

        # Get explanation and evaluation.
        explainer.eval()
        masked_adj = explainer.get_masked_adj()
        binarized_mask = (masked_adj > self.args.mask_thresh).to(torch.float32)
        pred1, pred2 = explainer(self.base_model, binarized_mask)
        sufficiency = int(pred1 > 0.5)  # Whether explanatory subgraph is original.
        necessity = int(pred2 <= 0.5)  # Whether residual graph is counterfactual.
        size = len(masked_adj[masked_adj > self.args.mask_thresh]) / g.num_edges
        return masked_adj, binarized_mask, sufficiency, necessity, size


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Mutagenicity',
                    choices=['Mutagenicity', 'Proteins', 'Mutag', 'IMDB-B', 'AIDS', 'NCI1', 'Tree-of-Life', 'Graph-SST2', 'DD', 'REDDIT-B'],
                    help="Dataset name")
parser.add_argument('--lam', type=int, default=20,
                    help='Lambda hyperparameter. Need to be tuned to make 1-norm loss and contrastive loss into the same scale. Default is 1000. ')
parser.add_argument("--lr", type=float, default=0.05, help="learning rate")
parser.add_argument("--epochs", type=int, default=500, help="number of the training epochs")
# TODO: change alp to 0.0 for counterfactual
parser.add_argument("--alp", dest="alp", type=float, default=1.0, help="hyper param control factual and counterfactual, 1 is totally factual")
parser.add_argument("--gam", dest="gam", type=float, default=0.5, help="margin value for bpr loss")
parser.add_argument("--mask_thresh", dest="mask_thresh", type=float, default=.5, help="threshold to convert relaxed adj matrix to binary")
parser.add_argument('--device', type=str, default="0")
parser.add_argument('--gnn_run', type=int, default=1)
parser.add_argument('--explainer_run', type=int, default=1)
parser.add_argument('--gnn_type', type=str, default='gcn', choices=['gcn', 'gat', 'gin', 'sage'])
parser.add_argument('--robustness', action='store_true')

args = parser.parse_args()

# Logging.
result_folder = f'data/{args.dataset}/cff_{args.alp}/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
dataset = data_utils.load_dataset(args.dataset)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
splits, indices = data_utils.split_data(dataset)

torch.manual_seed(args.explainer_run)
torch.cuda.manual_seed(args.explainer_run)
np.random.seed(args.explainer_run)
random.seed(args.explainer_run)

explanations_path = os.path.join(result_folder, f'explanations_{args.gnn_type}_run_{args.explainer_run}.pt')
counterfactual_path = os.path.join(result_folder, f'counterfactuals_{args.gnn_type}_run_{args.explainer_run}.pt')
args.method = 'classification'

trainer = GNNTrainer(dataset_name=args.dataset, gnn_type=args.gnn_type, task='basegnn', device=args.device)
model = trainer.load(args.gnn_run)
model.eval()

node_embeddings, graph_embeddings, outs = trainer.load_gnn_outputs(args.gnn_run)

if not args.robustness:
    explainer = GraphExplainerEdge(
        base_model=model,
        G_dataset=dataloader,
        args=args,
        device=device,
    )
    exps, cfs, sufficiency, necessity, average_size = explainer.explain_dataset()
    torch.save(exps, explanations_path)
    torch.save(cfs, counterfactual_path)
else:
    for noise in [1, 2, 3, 4, 5]:
        explanations_path = os.path.join(result_folder, f'explanations_{args.gnn_type}_run_{args.explainer_run}_noise_{noise}.pt')
        counterfactual_path = os.path.join(result_folder, f'counterfactuals_{args.gnn_type}_run_{args.explainer_run}_noise_{noise}.pt')
        noisy_dataset = data_utils.load_dataset(data_utils.get_noisy_dataset_name(dataset_name=args.dataset, noise=noise))
        dataloader = DataLoader(noisy_dataset, batch_size=1, shuffle=False)
        explainer = GraphExplainerEdge(
            base_model=model,
            G_dataset=dataloader,
            args=args,
            device=device,
        )
        exps, cfs, sufficiency, necessity, average_size = explainer.explain_dataset()
        torch.save(exps, explanations_path)
        torch.save(cfs, counterfactual_path)
