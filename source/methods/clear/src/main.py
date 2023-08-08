"""CLEAR explainer."""
import argparse
import math
import os
import pickle
import random
import sys
import time
import typing
from glob import glob

import numpy as np
import scipy.io as scio
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse, to_dense_adj, to_undirected
from torchvision.utils import save_image

sys.path += ["../", "../../../"]
import data_preprocessing as dpp
import models
import plot
import utils
from data_sampler import GraphData
from data_utils import load_dataset, split_data, get_noisy_dataset_name
from gnn_trainer import GNN, GnnSynthetic

font_sz = 28

parser = argparse.ArgumentParser(description='Graph counterfactual explanation generation')
parser.add_argument('--robustness', action='store_true')
# Blackbox
parser.add_argument("--num_layers", type=int, default=3, help="#hidden layers in the blackbox.")
parser.add_argument("--dim", type=int, default=20, help="Black-box's hidden dimension.")
parser.add_argument("--dropout", type=float, default=0., help="Blackbox dropout")
parser.add_argument("--layer", type=str, default="gcn", help="GNN layer to use in the blackbox.")
parser.add_argument("--pool", type=str, default="max", choices=["max", "mean", "sum"],
                    help="Blackbox global pooling")
# CLEAR
parser.add_argument('--device', default="cpu", help='Supply GPU number like 0, 1 for cuda, else cpu.')
parser.add_argument('--batch_size', type=int, default=500, metavar='N',
                    help='input batch size for training (default: 500)')
parser.add_argument('--num_workers', type=int, default=0, metavar='N')
parser.add_argument('--epochs', type=int, default=2000, metavar='N',
                    help='number of epochs to train (default: 2000)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--trn_rate', type=float, default=0.6, help='training data ratio')
parser.add_argument('--tst_rate', type=float, default=0.2, help='test data ratio')

parser.add_argument('--lamda', type=float, default=200, help='weight for CFE loss')
parser.add_argument('--kl_weight', type=float, default=1.0, help='weight for KL loss')
parser.add_argument('--disable_u', type=int, default=0, help='disable u in VAE')
parser.add_argument('--dim_z', type=int, default=16, metavar='N', help='dimension of z')
parser.add_argument('--dim_h', type=int, default=16, metavar='N', help='dimension of h')
# parser.add_argument('--dropout', type=float, default=0.1)

parser.add_argument('-d', '--dataset', required=True, help='dataset to use',
                    choices=["Mutagenicity", "Mutag", "Proteins", "AIDS",
                             "NCI1", "DD", "IMDB-B", "Graph-SST2", "REDDIT-B",
                             "syn1", "syn4", "syn5"])
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate for optimizer')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='weight decay')

parser.add_argument('--save_model', action="store_true")
parser.add_argument('--save_result', action="store_true")
parser.add_argument('-e', '--experiment_type', default='train', choices=['train', 'test', 'baseline'],
                    help='train: train CLEAR model; test: load CLEAR from file; baseline: run a baseline')
parser.add_argument('--baseline_type', default='random', choices=['IST', 'random', 'RM'],
                    help='select baseline type: insert, random perturb, or remove edges')

args = parser.parse_args()
if args.dataset.startswith('syn'):
    print("The following datasets only work with gcn: [syn1, syn4, syn5]")
    print("Using layer: gcn")
    args.layer = 'gcn'

# select gpu if available
if args.dataset in ['syn1', 'syn4', 'syn5']:
    args.weights = f"../../../../data/{args.dataset}/basegnn/gcn-max/gcn_3layer_{args.dataset}.pt"
else:
    args.weights = f"../../../../data/{args.dataset}/basegnn/{args.layer}-max/best_model_run_1.pt"
args.CFE_model_path = f'../models_save/{args.dataset}/seed_{args.seed}/{args.layer}'
os.system(f"mkdir -p {args.CFE_model_path}")
if args.device != "cpu":
    args.device = f"cuda:{args.device}"
    device = args.device
else:
    device = "cpu"
print()
print(args)
print()

print('using device: ', device)


# seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != "cpu":
    torch.cuda.manual_seed(args.seed)

def extract_epoch(path: str) -> int:
    # Use split("/") to split the path by the forward slash '/'
    parts = path.split("/")
    # Get the last part of the path
    last_part = parts[-1]
    # Use split("epoch") to split the last part and get the second element of the resulting list
    after_epoch = last_part.split("epoch")[-1]
    # Use split(".") to split the string and get the first element (the number) of the resulting list
    epoch_number = after_epoch.split(".")[0]
    return int(epoch_number)

def add_list_in_dict(key, dict, elem):
    if key not in dict:
        dict[key] = [elem]
    else:
        dict[key].append(elem)
    return dict

def distance_feature(feat_1, feat_2):
    pdist = nn.PairwiseDistance(p=2)
    output = pdist(feat_1, feat_2) /4
    return output

def distance_graph_prob(adj_1, adj_2_prob):
    dist = F.binary_cross_entropy(adj_2_prob, adj_1)
    return dist

def proximity_feature(feat_1, feat_2, type='cos'):
    if type == 'cos':
        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        output = cos(feat_1, feat_2)
        output = torch.mean(output)
    return output

def compute_loss(params):
    model, pred_model, z_mu, z_logvar, adj_permuted, features_permuted, adj_reconst, features_reconst, \
    adj_input, features_input, y_cf, z_u_mu, z_u_logvar, z_mu_cf, z_logvar_cf, target_node = params['model'], params['pred_model'], params['z_mu'], \
        params['z_logvar'], params['adj_permuted'], params['features_permuted'], params['adj_reconst'], params['features_reconst'], \
        params['adj_input'], params['features_input'], params['y_cf'], params['z_u_mu'], params['z_u_logvar'], params['z_mu_cf'], \
        params['z_logvar_cf'], params['target_node']

    # kl loss
    loss_kl = 0.5 * (((z_u_logvar - z_logvar) + ((z_logvar.exp() + (z_mu - z_u_mu).pow(2)) / z_u_logvar.exp())) - 1)
    loss_kl = torch.mean(loss_kl)

    # similarity loss
    size = len(features_permuted)
    dist_x = torch.mean(distance_feature(features_permuted.view(size, -1), features_reconst.view(size, -1)))
    dist_a = distance_graph_prob(adj_permuted, adj_reconst)

    beta = 10

    loss_sim = beta * dist_x + 10 * dist_a

    # CFE loss
    #* clear
    # y_pred = pred_model(features_reconst, adj_reconst)['y_pred']  # n x num_class
    y_pred = []
    for i in range(len(adj_reconst)):
        pyg_graph = get_pyg_graph(features_reconst[i], adj_reconst[i], target_node=target_node[i] if target_node is not None else None)
        pred = pred_model(pyg_graph, pyg_graph.edge_weight)[-1]
        y_pred.append(pred)
    y_pred = torch.cat(y_pred, dim=0)
    loss_cfe = F.nll_loss(F.log_softmax(y_pred, dim=-1), y_cf.view(-1).long())

    # rep loss
    if z_mu_cf is None:
        loss_kl_cf = 0.0
    else:
        loss_kl_cf = 0.5 * (((z_logvar_cf - z_logvar) + ((z_logvar.exp() + (z_mu - z_mu_cf).pow(2)) / z_logvar_cf.exp())) - 1)
        loss_kl_cf = torch.mean(loss_kl_cf)

    loss = 1. * loss_sim + 1 * loss_kl + 1.0 * loss_cfe

    loss_results = {'loss': loss, 'loss_kl': loss_kl, 'loss_sim': loss_sim, 'loss_cfe': loss_cfe, 'loss_kl_cf':loss_kl_cf}
    return loss_results

def train(params):
    epochs, pred_model, model, optimizer, y_cf_all, train_loader, val_loader, test_loader, exp_i, dataset, metrics, variant = \
        params['epochs'], params['pred_model'], params['model'], params['optimizer'], params['y_cf'],\
        params['train_loader'], params['val_loader'], params['test_loader'], params['exp_i'], params['dataset'], params['metrics'], params['variant']
    save_model = params['save_model'] if 'save_model' in params else True
    print("start training!")

    time_begin = time.time()
    best_loss = 100000

    for epoch in range(epochs + 1):
        model.train()

        loss, loss_kl, loss_sim, loss_cfe, loss_kl_cf = 0.0, 0.0, 0.0, 0.0, 0.0
        batch_num = 0
        for batch_idx, data in enumerate(train_loader):
            batch_num += 1

            features = data['features'].float().to(device)
            adj = data['adj'].float().to(device)
            u = data['u'].float().to(device)
            orin_index = data['index']
            y_cf = y_cf_all[orin_index]
            target_node = data['target_node']

            optimizer.zero_grad()

            # forward pass
            model_return = model(features, u, adj, y_cf)

            # z_cf
            z_mu_cf, z_logvar_cf = model.get_represent(model_return['features_reconst'], u, model_return['adj_reconst'], y_cf)

            # compute loss
            loss_params = {'model': model, 'pred_model': pred_model, 'adj_input': adj, 'features_input': features, 'target_node': target_node, 'y_cf': y_cf, 'z_mu_cf': z_mu_cf, 'z_logvar_cf':z_logvar_cf}
            loss_params.update(model_return)

            loss_results = compute_loss(loss_params)
            loss_batch, loss_kl_batch, loss_sim_batch, loss_cfe_batch, loss_kl_batch_cf = loss_results['loss'], loss_results['loss_kl'], loss_results['loss_sim'], loss_results['loss_cfe'], loss_results['loss_kl_cf']
            loss += loss_batch
            loss_kl += loss_kl_batch
            loss_sim += loss_sim_batch
            loss_cfe += loss_cfe_batch
            loss_kl_cf += loss_kl_batch_cf

        # backward propagation
        loss, loss_kl, loss_sim, loss_cfe, loss_kl_cf = loss / batch_num, loss_kl/batch_num, loss_sim/batch_num, loss_cfe/batch_num, loss_kl_cf/batch_num

        alpha = 5

        if epoch < 450:
            ((loss_sim + loss_kl + 0* loss_cfe)/ batch_num).backward()
        else:
            ((loss_sim + loss_kl + alpha * loss_cfe)/ batch_num).backward()
        optimizer.step()

        # evaluate
        # if epoch % 1 == 0:
        if epoch % 100 == 0:
            model.eval()
            eval_params_val = {'model': model, 'data_loader': val_loader, 'pred_model': pred_model, 'y_cf': y_cf_all, 'dataset': dataset, 'metrics': metrics}
            eval_params_tst = {'model': model, 'data_loader': test_loader, 'pred_model': pred_model, 'y_cf': y_cf_all, 'dataset': dataset, 'metrics': metrics}
            eval_results_val = test(eval_params_val)
            eval_results_tst = test(eval_params_tst)
            val_loss, val_loss_kl, val_loss_sim, val_loss_cfe = eval_results_val['loss'], eval_results_val['loss_kl'], eval_results_val['loss_sim'], eval_results_val['loss_cfe']

            metrics_results_val = ""
            metrics_results_tst = ""
            for k in metrics:
                metrics_results_val += f"{k}_val: {eval_results_val[k]:.4f} | "
                metrics_results_tst += f"{k}_tst: {eval_results_tst[k]:.4f} | "

            print(f"[Train] Epoch {epoch}: train_loss: {(loss):.4f} |" +
                  metrics_results_val + metrics_results_tst +
                  f"time: {(time.time() - time_begin):.4f} |")

            # save
            if save_model:
                # if epoch % 1 == 0:
                if epoch % 300 == 0 and epoch > 450:
                    CFE_model_path = f"{args.CFE_model_path}/{variant}_exp{exp_i}_epoch{epoch}.pt"
                    torch.save(model.state_dict(), CFE_model_path)
                    print('saved CFE model in: ', CFE_model_path)
    return

def test(params):
    model, data_loader, pred_model, y_cf_all, dataset, metrics = params['model'], params['data_loader'], params['pred_model'], params['y_cf'], params['dataset'], params['metrics']
    model.eval()
    pred_model.eval()

    eval_results_all = {k: 0.0 for k in metrics}
    size_all = 0
    loss, loss_kl, loss_sim, loss_cfe = 0.0, 0.0, 0.0, 0.0
    batch_num = 0

    # * benchmarking data
    counterfactual_list = []
    for batch_idx, data in enumerate(data_loader):
        batch_num += 1
        batch_size = len(data['labels'])
        size_all += batch_size

        features = data['features'].float().to(device)
        adj = data['adj'].float().to(device)
        u = data['u'].float().to(device)
        labels = data['labels'].float().to(device)
        orin_index = data['index']
        y_cf = y_cf_all[orin_index]
        target_node = data['target_node']

        model_return = model(features, u, adj, y_cf)
        adj_reconst, features_reconst = model_return['adj_reconst'], model_return['features_reconst']

        adj_reconst_binary = torch.bernoulli(adj_reconst)

        counterfactual = dict()
        # y_cf_pred = pred_model(features_reconst, adj_reconst_binary)['y_pred']
        # y_pred = pred_model(features, adj)['y_pred']
        y_cf_pred = []
        y_pred = []
        assert len(adj) == len(adj_reconst_binary), "Different adjacencies."
        for i in range(len(adj_reconst_binary)):
            pyg_graph = get_pyg_graph(features[i], adj[i], target_node=target_node[i] if target_node is not None else None)
            pred = pred_model(pyg_graph, pyg_graph.edge_weight)[-1]
            y_pred.append(pred)

            is_undirected = pyg_graph.is_undirected()
            pyg_graph_cf = get_pyg_graph(features_reconst[i], adj_reconst_binary[i], is_undirected=is_undirected, target_node=target_node[i] if target_node is not None else None)
            pred_cf = pred_model(pyg_graph_cf, pyg_graph_cf.edge_weight)[-1]
            y_cf_pred.append(pred_cf)

            counterfactual = {
                "graph": pyg_graph.clone().detach().cpu(), "graph_cf": pyg_graph_cf.clone().detach().cpu(),
                "pred": pred[0].clone().detach().cpu(), "pred_cf": pred_cf[0].clone().detach().cpu(),
                "label": int(labels[i]),
            }
            counterfactual_list.append(counterfactual)
        y_cf_pred = torch.cat(y_cf_pred, dim=0)
        y_pred = torch.cat(y_pred, dim=0)

        # z_cf
        z_mu_cf, z_logvar_cf = None, None

        # compute loss
        loss_params = {'model': model, 'pred_model': pred_model, 'adj_input': adj, 'features_input': features, 'target_node': target_node, 'y_cf': y_cf, 'z_mu_cf': z_mu_cf, 'z_logvar_cf':z_logvar_cf}
        loss_params.update(model_return)

        loss_results = compute_loss(loss_params)
        loss_batch, loss_kl_batch, loss_sim_batch, loss_cfe_batch = loss_results['loss'], loss_results['loss_kl'], \
                                                                    loss_results['loss_sim'], loss_results['loss_cfe']
        loss += loss_batch
        loss_kl += loss_kl_batch
        loss_sim += loss_sim_batch
        loss_cfe += loss_cfe_batch

        # evaluate metrics
        eval_params = model_return.copy()
        eval_params.update({'y_cf': y_cf, 'metrics': metrics, 'y_cf_pred': y_cf_pred, 'dataset': dataset, 'adj_input': adj, 'features_input': features, 'labels':labels, 'u': u, 'y_pred':y_pred})

        eval_results = evaluate(eval_params)
        for k in metrics:
            eval_results_all[k] += (batch_size * eval_results[k])

    for k in metrics:
        eval_results_all[k] /= size_all

    loss, loss_kl, loss_sim, loss_cfe = loss / batch_num, loss_kl / batch_num, loss_sim / batch_num, loss_cfe / batch_num
    eval_results_all['loss'], eval_results_all['loss_kl'], eval_results_all['loss_sim'], eval_results_all['loss_cfe'] = loss, loss_kl, loss_sim, loss_cfe

    if args.experiment_type == "test":
        repo_home = "../../../../"
        cf_path = f"{repo_home}/data/{args.dataset}/clear"
        os.system(f"mkdir -p {cf_path}")
        cf_path += f"/explanations_{args.layer}_run_{args.seed}{f'_noise_{args.noise}' if args.robustness else ''}.pt"
        torch.save(counterfactual_list, cf_path)
        print(f"Saved cfs at {cf_path}")
    return eval_results_all

def evaluate(params):
    adj_permuted, features_permuted, adj_reconst_prob, features_reconst, metrics, dataset, y_cf, y_cf_pred, labels, u, y_pred = \
        params['adj_permuted'], params['features_permuted'], params['adj_reconst'], \
        params['features_reconst'], params['metrics'], params['dataset'], params['y_cf'], params['y_cf_pred'], params['labels'], params['u'], params['y_pred']

    adj_reconst = torch.bernoulli(adj_reconst_prob)
    eval_results = {}
    if 'causality' in metrics:
        score_causal = evaluate_causality(dataset, adj_permuted, features_permuted, adj_reconst, features_reconst,  y_cf, labels, u)
        eval_results['causality'] = score_causal
    if 'proximity' in metrics or 'proximity_x' in metrics or 'proximity_a' in metrics:
        score_proximity, dist_x, dist_a = evaluate_proximity(dataset, adj_permuted, features_permuted, adj_reconst_prob, adj_reconst, features_reconst)
        eval_results['proximity'] = score_proximity
        eval_results['proximity_x'] = dist_x
        eval_results['proximity_a'] = dist_a
    if 'validity' in metrics:
        score_valid = evaluate_validity(y_cf, y_cf_pred)
        eval_results['validity'] = score_valid
    if 'correct' in metrics:
        score_correct = evaluate_correct(dataset, adj_permuted, features_permuted, adj_reconst, features_reconst, y_cf, labels, y_cf_pred, y_pred)
        eval_results['correct'] = score_correct

    return eval_results

def evaluate_validity(y_cf, y_cf_pred):
    y_cf_pred_binary = F.softmax(y_cf_pred, dim=-1)
    y_cf_pred_binary = y_cf_pred_binary.argmax(dim=1).view(-1,1)
    y_eq = torch.where(y_cf == y_cf_pred_binary, torch.tensor(1.0).to(device), torch.tensor(0.0).to(device))
    score_valid = torch.mean(y_eq)
    return score_valid

def evaluate_causality(dataset, adj_permuted, features_permuted, adj_reconst, features_reconst, y_cf, labels, u):
    score_causal = 0.0
    if dataset == 'synthetic' or dataset == 'imdb_m':
        size = len(features_permuted)
        max_num_nodes = adj_reconst.shape[-1]

        # Constraint
        ave_degree = (torch.sum(adj_permuted.view(size, -1), dim=-1) - max_num_nodes) / (2 * max_num_nodes) # size
        ave_degree_cf = (torch.sum(adj_reconst.view(size, -1), dim=-1) - max_num_nodes) / (2 * max_num_nodes)
        ave_x0 = torch.mean(features_permuted[:, :, 0], dim=-1)  # size
        ave_x0_cf = torch.mean(features_reconst[:, :, 0], dim=-1)  # size

        count_good = torch.where(
            (((ave_degree > ave_degree_cf) & (ave_x0 > ave_x0_cf)) |
             ((ave_degree == ave_degree_cf) & (ave_x0 == ave_x0_cf)) |
             ((ave_degree < ave_degree_cf) & (ave_x0 < ave_x0_cf))), torch.tensor(1.0).to(device), torch.tensor(0.0).to(device))

        score_causal = torch.mean(count_good)

    elif dataset == 'ogbg_molhiv':
        ave_x0 = torch.mean(features_permuted[:, :, 0], dim=-1)  # size
        ave_x0_cf = torch.mean(features_reconst[:, :, 0], dim=-1)  # size
        ave_x1 = torch.mean(features_permuted[:, :, 1], dim=-1)  # size
        ave_x1_cf = torch.mean(features_reconst[:, :, 1], dim=-1)  # size

        count_good = torch.where(
            (((ave_x0 > ave_x0_cf) & (ave_x1 > ave_x1_cf)) |
             ((ave_x0 == ave_x0_cf) & (ave_x1 == ave_x1_cf)) |
             ((ave_x0 < ave_x0_cf) & (ave_x1 < ave_x1_cf))), torch.tensor(1.0).to(device),
            torch.tensor(0.0).to(device))
        score_causal = torch.mean(count_good)

    elif dataset == 'community':
        size = len(features_permuted)
        max_num_nodes = adj_reconst.shape[-1]

        # Constraint
        n0 = int(max_num_nodes/2)
        n1 = max_num_nodes - n0

        ave_degree_0 = (torch.sum(adj_permuted[:, :n0, :n0].reshape(size, -1), dim=-1) - n0) / (2 * n0)  # size
        ave_degree_cf_0 = (torch.sum(adj_reconst[:, :n0, :n0].reshape(size, -1), dim=-1) - n0) / (2 * n0)
        ave_degree_1 = (torch.sum(adj_permuted[:, n0:, n0:].reshape(size, -1), dim=-1) - n1) / (2 * n1)  # size
        ave_degree_cf_1 = (torch.sum(adj_reconst[:, n0:, n0:].reshape(size, -1), dim=-1) - n1) / (2 * n1)

        max_dg = ave_degree_1.max().tile(len(ave_degree_1))
        min_dg = ave_degree_1.min().tile(len(ave_degree_1))

        count_good = torch.where(
            (((ave_degree_0 > ave_degree_cf_0) & (((ave_degree_1 < max_dg) & (ave_degree_1 < ave_degree_cf_1)) | (ave_degree_1 == max_dg))) |
             ((ave_degree_0 == ave_degree_cf_0) & (ave_degree_1 == ave_degree_cf_1)) |
             ((ave_degree_0 < ave_degree_cf_0) & (((ave_degree_1 > min_dg) & (ave_degree_1 > ave_degree_cf_1)) | (ave_degree_1 == min_dg)))), torch.tensor(1.0).to(device),
            torch.tensor(0.0).to(device))

        score_causal = torch.mean(count_good)

    return score_causal

def evaluate_proximity(dataset, adj_permuted, features_permuted, adj_reconst_prob, adj_reconst, features_reconst):
    size = len(features_permuted)
    dist_x = torch.mean(distance_feature(features_permuted.view(size, -1), features_reconst.view(size, -1)))
    dist_a = distance_graph_prob(adj_permuted, adj_reconst_prob)
    score = dist_x + dist_a

    proximity_x = proximity_feature(features_permuted, features_reconst, 'cos')

    acc_a = (adj_permuted == adj_reconst).float().mean()
    return score, proximity_x, acc_a

def evaluate_correct(dataset, adj_permuted, features_permuted, adj_reconst, features_reconst, y_cf, labels, y_cf_pred, y_pred):
    y_cf_pred_binary = F.softmax(y_cf_pred, dim=-1)
    y_cf_pred_binary = y_cf_pred_binary.argmax(dim=1).view(-1, 1)
    y_pred_binary = F.softmax(y_pred, dim=-1)
    y_pred_binary = y_pred_binary.argmax(dim=1).view(-1, 1)

    score = -1.0
    if dataset == 'synthetic' or dataset == 'imdb_m':
        size = len(features_permuted)
        max_num_nodes = adj_reconst.shape[-1]
        ave_degree = (torch.sum(adj_permuted.view(size, -1), dim=-1) - max_num_nodes) / (2 * max_num_nodes)  # size
        ave_degree_cf = (torch.sum(adj_reconst.view(size, -1), dim=-1) - max_num_nodes) / (2 * max_num_nodes)

        count_good = torch.where(
            (((ave_degree > ave_degree_cf) & (labels.view(-1) > y_cf.view(-1))) |
            ((ave_degree < ave_degree_cf) & (labels.view(-1) < y_cf.view(-1)))), torch.tensor(1.0).to(device), torch.tensor(0.0).to(device))

        score = torch.sum(count_good)
        all = (labels.view(-1) != y_cf.view(-1)).sum()
        if all.item() == 0:
            return score / (all+1)
        score = score / all
    elif dataset == 'ogbg_molhiv':
        ave_x1 = torch.mean(features_permuted[:, :, 1], dim=-1)  # size
        ave_x1_cf = torch.mean(features_reconst[:, :, 1], dim=-1)  # size

        count_good = torch.where(
            (((ave_x1 > ave_x1_cf) & (y_pred_binary.view(-1) > y_cf_pred_binary.view(-1))) |
             ((ave_x1 < ave_x1_cf) & (y_pred_binary.view(-1) < y_cf_pred_binary.view(-1)))),
            torch.tensor(1.0).to(device),
            torch.tensor(0.0).to(device))

        score = torch.sum(count_good)
        all = (y_pred_binary.view(-1) != y_cf_pred_binary.view(-1)).sum()
        if all.item() == 0:
            return score / (all + 1)
        score = score / all

    elif dataset == 'community':
        size = len(features_permuted)
        max_num_nodes = adj_reconst.shape[-1]

        n0 = int(max_num_nodes / 2)
        n1 = max_num_nodes - n0
        ave_degree_0 = (torch.sum(adj_permuted[:, :n0, :n0].reshape(size, -1), dim=-1) - n0) / (2 * n0)  # size
        ave_degree_cf_0 = (torch.sum(adj_reconst[:, :n0, :n0].reshape(size, -1), dim=-1) - n0) / (2 * n0)

        count_good = torch.where(
            (((ave_degree_0 > ave_degree_cf_0) & (y_pred_binary.view(-1) > y_cf_pred_binary.view(-1))) |
             ((ave_degree_0 < ave_degree_cf_0) & (y_pred_binary.view(-1) < y_cf_pred_binary.view(-1)))), torch.tensor(1.0).to(device),
            torch.tensor(0.0).to(device))
        # score = torch.mean(count_good)
        score = torch.sum(count_good)
        all = (y_pred_binary.view(-1) != y_cf_pred_binary.view(-1)).sum()
        if all.item() == 0:
            return score / (all + 1)
        score = score / all

    return score

def perturb_graph(adj, type='random', num_rounds=1):
    num_node = adj.shape[0]
    num_entry = num_node * num_node
    adj_cf = adj.clone()
    if type == 'random':
        # randomly add/remove edges for T rounds
        for rd in range(num_rounds):
            [row, col] = np.random.choice(num_node, size=2, replace=False)
            adj_cf[row, col] = 1 - adj[row, col]
            adj_cf[col, row] = adj_cf[row, col]

    elif type == 'IST':
        # randomly add edge
        for rd in range(num_rounds):
            idx_select = (adj_cf == 0).nonzero()  # 0
            if len(idx_select) <= 0:
                continue
            ii = np.random.choice(len(idx_select), size=1, replace=False)
            idx = idx_select[ii].view(-1)
            row, col = idx[0], idx[1]
            adj_cf[row, col] = 1
            adj_cf[col, row] = 1

    elif type == 'RM':
        # randomly remove edge
        for rd in range(num_rounds):
            idx_select = adj_cf.nonzero()  # 1
            if len(idx_select) <= 0:
                continue
            ii = np.random.choice(len(idx_select), size=1, replace=False)
            idx = idx_select[ii].view(-1)
            row, col = idx[0], idx[1]
            adj_cf[row, col] = 0
            adj_cf[col, row] = 0

    return adj_cf

def baseline_cf(dataset, data_loader, metrics, y_cf_all, pred_model, num_rounds = 10, type='random'):
    eval_results_all = {k: 0.0 for k in metrics}
    size_all = 0
    batch_num = 0
    for batch_idx, data in enumerate(data_loader):
        batch_num += 1
        batch_size = len(data['labels'])
        size_all += batch_size

        features = data['features'].float().to(device)
        adj = data['adj'].float().to(device)
        u = data['u'].float().to(device)
        labels = data['labels'].float().to(device)
        orin_index = data['index'].to(device)
        y_cf = y_cf_all[orin_index].to(device)

        adj_reconst = adj.clone()

        noise = torch.normal(mean=0.0, std=1, size=features.shape).to(device)  # add a Gaussian noise to node features
        features_reconst = features + noise

        # perturbation on A
        for i in range(batch_size):
            for t in range(num_rounds):
                adj_reconst[i] = perturb_graph(adj_reconst[i], type, num_rounds=1)  # randomly perturb graph
                # y_cf_pred_i = pred_model(features_reconst[i].unsqueeze(0), adj_reconst[i].unsqueeze(0))['y_pred'].argmax(dim=1).view(-1,1)  # 1 x 1
                y_cf_pred_i = pred_model(get_pyg_graph(features_reconst[i], adj_reconst[i]))[-1].argmax(dim=1).view(-1, 1)
                if y_cf_pred_i.item() == y_cf[i].item():  # Stop when f(G^CF) == Y^CF
                    break

        # prediction model
        # y_cf_pred = pred_model(features_reconst, adj_reconst)['y_pred']
        y_cf_pred = []
        for i in range(len(adj_reconst)):
            pyg_graph = get_pyg_graph(features_reconst[i], adj_reconst[i])
            pred = pred_model(pyg_graph, pyg_graph.edge_weight)[-1]
            y_cf_pred.append(pred)
        y_cf_pred = torch.cat(y_cf_pred, dim=0)

        # y_pred = pred_model(features, adj)['y_pred']
        y_pred = []
        for i in range(len(adj)):
            pyg_graph = get_pyg_graph(features[i], adj[i])
            pred = pred_model(pyg_graph, pyg_graph.edge_weight)[-1]
            y_pred.append(pred)
        y_pred = torch.cat(y_pred, dim=0)

        # evaluate metrics
        eval_params = {}
        eval_params.update(
            {'y_cf': y_cf, 'metrics': metrics, 'y_cf_pred': y_cf_pred, 'dataset': dataset, 'adj_permuted': adj,
             'features_permuted': features, 'adj_reconst': adj_reconst, 'features_reconst': features_reconst, 'labels': labels, 'u': u, 'y_pred':y_pred})

        eval_results = evaluate(eval_params)
        for k in metrics:
            eval_results_all[k] += (batch_size * eval_results[k])

    for k in metrics:
        eval_results_all[k] /= size_all

    return eval_results_all

def run_clear(args, exp_type):
    # data_path_root = '../dataset/'
    # model_path = '../models_save/'
    assert exp_type == 'train' or exp_type == 'test' or exp_type == 'test_small'
    small_test = 20

    # load data
    # data_load = dpp.load_data(data_path_root, args.dataset)
    data_load = get_pyg_dataset_and_convert_to_clear(dataset_name=args.dataset, noise=args.noise)
    idx_train_list, idx_val_list, idx_test_list = data_load['idx_train_list'], data_load['idx_val_list'], data_load['idx_test_list'] #todo Set train and test indices.
    data = data_load['data']
    x_dim = data[0]["features"].shape[1]
    u_unique = np.unique(np.array(data.u_all))
    u_dim = len(u_unique)

    n = len(data)
    max_num_nodes = data.max_num_nodes
    unique_class = np.unique(np.array(data.labels_all))
    num_class = len(unique_class)
    print('n ', n, 'x_dim: ', x_dim, ' max_num_nodes: ', max_num_nodes, ' num_class: ', num_class)

    results_all_exp = {}
    init_params = {'vae_type': 'graphVAE', 'x_dim': x_dim, 'u_dim': u_dim, 'max_num_nodes': max_num_nodes}  # parameters for initialize GraphCFE model

    # load model
    # pred_model = models.Graph_pred_model(x_dim, 32, num_class, max_num_nodes, args.dataset).to(device)
    # pred_model.load_state_dict(torch.load(model_path + f'prediction/weights_graphPred__{args.dataset}' + '.pt'))
    if args.dataset in ['syn1', 'syn4', 'syn5']:
        pred_model = GnnSynthetic(
            nfeat=x_dim,
            nhid=20,
            nout=20,
            nclass=4 if args.dataset == 'syn1' else 2,
            dropout=0.0
        )
    else:
        pred_model = GNN(num_features=x_dim, num_classes=2, num_layers=args.num_layers,
                         dim=args.dim, dropout=args.dropout, layer=args.layer, pool=args.pool)
    pred_model.load_state_dict(torch.load(args.weights, map_location=device))
    pred_model = pred_model.to(device)
    pred_model.eval()

    y_cf = (num_class - 1) - np.array(data.labels_all)
    y_cf = torch.FloatTensor(y_cf).to(device)

    # metrics = ['causality', 'validity', 'proximity_x', 'proximity_a']
    metrics = ['validity', 'proximity_x', 'proximity_a']
    time_spent_all = []

    exp_num = 1
    for exp_i in range(0, exp_num):
        print('============================= Start experiment ', str(exp_i),
              ' =============================================')
        idx_train = idx_train_list[exp_i]
        idx_val = idx_val_list[exp_i]
        idx_test = idx_test_list[exp_i]

        if args.disable_u:
            model = models.GraphCFE(init_params=init_params, args=args)
        else:
            model = models.GraphCFE(init_params=init_params, args=args)


        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # data loader
        train_loader = utils.select_dataloader(data, idx_train, batch_size=args.batch_size,num_workers=args.num_workers)
        val_loader = utils.select_dataloader(data, idx_val, batch_size=args.batch_size, num_workers=args.num_workers)
        test_loader = utils.select_dataloader(data, idx_test, batch_size=args.batch_size, num_workers=args.num_workers)

        if "cuda" in args.device:
            model = model.to(device)

        variant = 'VAE' if args.disable_u else 'CLEAR'
        if exp_type == 'train':
            # train
            train_params = {'epochs': args.epochs, 'model': model, 'pred_model': pred_model, 'optimizer': optimizer,
                            'y_cf': y_cf,
                            'train_loader': train_loader, 'val_loader': val_loader, 'test_loader': test_loader,
                            'exp_i': exp_i,
                            'dataset': args.dataset, 'metrics': metrics, 'save_model': args.save_model, 'variant': variant}
            train(train_params)
        else:
            # test
            # CFE_model_path = model_path + f'weights_graphCFE_{variant}_{args.dataset}_exp' + str(exp_i) +'_epoch'+args.epochs + '.pt'
            try:
                best_path = ''
                highest_epoch = 0
                #pick path of file with max epochs(ideal is 1200)
                for path in glob(f'{args.CFE_model_path}/*'):
                    epoch = extract_epoch(path)
                    if(epoch > highest_epoch):
                        best_path = path
                        highest_epoch = epoch
                CFE_model_path = best_path #glob(f'{args.CFE_model_path}/*')[-1]
            except IndexError:
                raise FileNotFoundError
            model.load_state_dict(torch.load(CFE_model_path))
            print('CFE generator loaded from: ' + CFE_model_path)
            if exp_type == 'test_small':
                test_loader = utils.select_dataloader(data, idx_test[:small_test], batch_size=args.batch_size, num_workers=args.num_workers)

        test_params = {'model': model, 'dataset': args.dataset, 'data_loader': test_loader, 'pred_model': pred_model,
                       'metrics': metrics, 'y_cf': y_cf}

        time_begin = time.time()

        eval_results = test(test_params)

        time_end = time.time()
        time_spent = time_end - time_begin
        time_spent = time_spent / small_test
        time_spent_all.append(time_spent)

        for k in metrics:
            results_all_exp = add_list_in_dict(k, results_all_exp, eval_results[k].detach().cpu().numpy())

        print('=========================== Exp ', str(exp_i), ' Results ==================================')
        for k in eval_results:
            if isinstance(eval_results[k], list):
                print(k, ": ", eval_results[k])
            else:
                print(k, f": {eval_results[k]:.4f}")
        print('time: ', time_spent)

    print('============================= Overall Results =============================================')
    record_exp_result = {}  # save in file
    for k in results_all_exp:
        results_all_exp[k] = np.array(results_all_exp[k])
        print(k, f": mean: {np.mean(results_all_exp[k]):.4f} | std: {np.std(results_all_exp[k]):.4f}")
        record_exp_result[k] = {'mean': np.mean(results_all_exp[k]), 'std': np.std(results_all_exp[k])}

    time_spent_all = np.array(time_spent_all)
    record_exp_result['time'] = {'mean': np.mean(time_spent_all), 'std': np.std(time_spent_all)}

    save_result = args.save_result
    print("====save in file ====")
    print(record_exp_result)
    if save_result:
        exp_save_path = '../exp_results/'
        if args.disable_u:
            exp_save_path = f"{exp_save_path}/CVAE/{args.dataset}/seed_{args.seed}/{args.layer}"
        else:
            exp_save_path = f"{exp_save_path}/CLEAR/{args.dataset}/seed_{args.seed}/{args.layer}"
        os.system(f"mkdir -p {exp_save_path}")
        exp_save_path += "result.pickle"
        with open(exp_save_path, 'wb') as handle:
            pickle.dump(record_exp_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('saved data: ', exp_save_path)
    return

def run_baseline(args, type='random'):
    # data_path_root = '../dataset/'
    # model_path = '../models_save/'
    small_test = 20
    num_rounds = 150

    # data_load = dpp.load_data(data_path_root, args.dataset)
    data_load = get_pyg_dataset_and_convert_to_clear(dataset_name=args.dataset)
    idx_train_list, idx_val_list, idx_test_list = data_load['idx_train_list'], data_load['idx_val_list'], data_load[
        'idx_test_list']
    data = data_load['data']
    x_dim = data[0]["features"].shape[1]

    n = len(data)
    max_num_nodes = data.max_num_nodes
    unique_class = np.unique(np.array(data.labels_all))
    num_class = len(unique_class)
    print('n ', n, 'x_dim: ', x_dim, ' max_num_nodes: ', max_num_nodes, ' num_class: ', num_class)

    results_all_exp = {}
    exp_num = 3  # 10

    # load model
    # pred_model = models.Graph_pred_model(x_dim, 32, num_class, max_num_nodes, args.dataset).to(device)
    # pred_model.load_state_dict(torch.load(model_path + f'prediction/weights_graphPred__{args.dataset}' + '.pt'))
    pred_model = GNN(num_features=x_dim, num_classes=2, num_layers=args.num_layers,
                     dim=args.dim, dropout=args.dropout, layer=args.layer, pool=args.pool)
    pred_model.load_state_dict(torch.load(args.weights))
    pred_model.eval()

    y_cf = (num_class - 1) - np.array(data.labels_all)
    y_cf = torch.FloatTensor(y_cf).to(device)
    metrics = ['causality', 'proximity', 'validity', 'proximity_x', 'proximity_a', 'correct']

    time_spent_all = []

    for exp_i in range(0, exp_num):
        print('============================= Start experiment ', str(exp_i),
              ' =============================================')
        time_begin = time.time()

        idx_test = idx_test_list[exp_i]

        # data loader
        test_loader = utils.select_dataloader(data, idx_test[:small_test], batch_size=args.batch_size, num_workers=args.num_workers)

        # baseline
        eval_results = baseline_cf(args.dataset, test_loader, metrics, y_cf, pred_model, num_rounds=num_rounds, type=type)

        time_end = time.time()
        time_spent = time_end - time_begin
        time_spent = time_spent / small_test
        time_spent_all.append(time_spent)

        for k in metrics:
            results_all_exp = add_list_in_dict(k, results_all_exp, eval_results[k].detach().cpu().numpy())

        print('=========================== Exp ', str(exp_i), ' Results ==================================')
        for k in eval_results:
            if isinstance(eval_results[k], list):
                print(k, ": ", eval_results[k])
            else:
                print(k, f": {eval_results[k]:.4f}")
        print('time: ', time_spent)

    print('============================= Overall Results =============================================')

    for k in results_all_exp:
        results_all_exp[k] = np.array(results_all_exp[k])
        print(k, f": mean: {np.mean(results_all_exp[k]):.4f} | std: {np.std(results_all_exp[k]):.4f}")
    time_spent_all = np.array(time_spent_all)
    print('time', f": mean: {np.mean(time_spent_all):.4f} | std: {np.std(time_spent_all):.4f}")
    return

def get_pyg_graph(clear_features: np.ndarray, clear_adj: np.ndarray, is_undirected: bool = False, target_node: int = None) -> Data:
    x = clear_features
    edge_index, edge_weight = dense_to_sparse(clear_adj)
    if is_undirected:
        edge_index, edge_weight = to_undirected(edge_index=edge_index, edge_attr=edge_weight)
    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight, target_node=target_node)

def get_pyg_dataset_and_convert_to_clear(dataset_name: str, noise:int = None) -> typing.Dict:
    if noise is None:
        dataset = load_dataset(dataset_name, root="../../../../data/")
    else:
        dataset = load_dataset(get_noisy_dataset_name(dataset_name=args.dataset, noise=noise), root="../../../../data/")

    if dataset_name in ['syn1', 'syn4', 'syn5']:
        if dataset_name == 'syn1':
            train_idx = [333, 97, 380, 311, 281, 10, 56, 680, 67, 488, 467, 492, 465, 78, 186, 696, 478, 216, 65, 116, 676, 236, 613, 37, 554, 382, 325, 668, 440, 201, 543, 495, 107, 124, 328, 221, 587, 237, 266, 61, 239, 66, 356, 322, 567, 515, 46, 87, 424, 572, 273, 123, 570, 342, 521, 673, 410, 577, 282, 252, 42, 117, 559, 590, 686, 57, 511, 243, 291, 26, 598, 213, 489, 413, 47, 277, 421, 62, 561, 157, 386, 594, 507, 188, 436, 651, 553, 55, 564, 363, 296, 376, 187, 430, 43, 115, 137, 295, 257, 354, 161, 475, 102, 408, 180, 160, 442, 22, 58, 659, 271, 60, 72, 32, 75, 381, 288, 194, 591, 81, 460, 175, 601, 150, 48, 120, 109, 523, 334, 510, 29, 370, 628, 637, 513, 389, 650, 634, 428, 130, 17, 474, 302, 390, 101, 119, 191, 133, 165, 398, 387, 606, 480, 362, 401, 660, 687, 212, 621, 466, 485, 625, 9, 304, 623, 336, 666, 575, 379, 287, 337, 77, 493, 153, 443, 438, 422, 481, 626, 453, 426, 64, 263, 491, 192, 284, 85, 524, 396, 324, 656, 234, 350, 233, 459, 528, 532, 518, 642, 317, 534, 299, 285, 557, 307, 198, 612, 638, 602, 536, 603, 79, 458, 18, 73, 383, 114, 624, 355, 135, 499, 373, 405, 12, 406, 279, 441, 588, 586, 599, 174, 195, 461, 500, 301, 689, 618, 654, 217, 238, 449, 583, 3, 627, 95, 448, 399, 139, 679, 596, 551, 140, 463, 223, 698, 88, 529, 321, 368, 147, 256, 372, 403, 156, 222, 502, 178, 584, 446, 416, 141, 199, 486, 275, 340, 595, 435, 400, 509, 685, 108, 59, 255, 359, 592, 241, 264, 437, 313, 620, 83, 661, 270, 691, 525, 278, 8, 112, 34, 231, 662, 329, 580, 104, 189, 93, 579, 482, 267, 699, 24, 21, 118, 138, 341, 30, 268, 326, 397, 695, 531, 166, 323, 514, 290, 692, 395, 505, 468, 197, 182, 283, 375, 615, 300, 427, 203, 522, 314, 411, 417, 566, 53, 357, 52, 96, 578, 27, 316, 286, 503, 89, 655, 549, 196, 319, 533, 539, 504, 432, 353, 556, 129, 439, 645, 261, 207, 169, 608, 128, 494, 360, 470, 106, 548, 303, 84, 80, 565, 348, 418, 339, 297, 204, 517, 550, 149, 352, 351, 152, 94, 526, 452, 132, 269, 684, 682, 391, 7, 541, 113, 6, 163, 483, 392, 501, 210, 409, 555, 450, 472, 484, 552, 248, 347, 346, 11, 469, 644, 244, 71, 126, 431, 445, 154, 366, 657, 20, 576, 677, 50, 614, 69, 568, 309, 669, 343, 168, 220, 51, 633, 159, 540, 653, 635, 183, 569, 312, 260, 111, 652, 76, 122, 393, 344, 45, 639, 506, 318, 694, 289, 155, 631, 664, 39, 49, 306, 68, 476, 259, 177, 98, 434, 641, 636, 251, 479, 678, 315, 193, 358, 527, 394, 4, 648, 227, 293, 361, 407, 593, 229, 148, 622, 31, 535, 611, 90, 378, 690, 173, 158, 185, 520, 170, 629, 23, 249, 674, 640, 327, 308, 144, 681, 171, 19, 457, 127, 179, 546, 125, 136, 365, 38, 544, 384, 369, 423, 538, 190, 335, 225, 54, 162, 181, 332, 224, 146, 331, 36, 547, 420, 530, 258, 338, 131, 145, 247, 91, 675, 658, 240, 693, 571, 388, 16, 134, 121, 433, 40, 0]
            test_idx = [563, 415, 605, 320, 597, 345, 367, 310, 496, 671, 451, 464, 607, 562, 498, 456, 688, 419, 537, 425, 616, 672, 487, 560, 414, 604, 643, 617, 385, 508, 542, 471, 667, 558, 462, 412, 364, 349, 429, 609, 581, 402, 589, 444, 574, 630, 512, 697, 600, 490, 582, 670, 649, 610, 497, 573, 330, 647, 447, 516, 619, 305, 545, 404, 477, 377, 585, 455, 371, 646, 374, 519, 454, 665, 663, 632]
        elif dataset_name == 'syn4':
            train_idx = [498, 852, 868, 518, 364, 248, 474, 70, 281, 709, 635, 770, 578, 842, 627, 278, 106, 72, 720, 643, 513, 393, 421, 261, 344, 30, 79, 451, 254, 110, 172, 858, 2, 114, 538, 660, 503, 177, 150, 744, 628, 88, 229, 238, 507, 687, 415, 33, 558, 4, 319, 683, 382, 120, 164, 445, 402, 803, 828, 559, 159, 593, 259, 511, 201, 240, 138, 493, 707, 394, 657, 436, 640, 819, 556, 755, 649, 681, 191, 599, 133, 768, 472, 270, 384, 619, 747, 775, 815, 28, 298, 466, 809, 678, 35, 708, 497, 173, 704, 588, 607, 306, 795, 52, 129, 772, 833, 475, 267, 3, 645, 804, 134, 369, 48, 365, 540, 241, 244, 455, 349, 340, 44, 176, 165, 322, 847, 308, 405, 321, 427, 420, 145, 680, 668, 111, 526, 253, 36, 287, 594, 605, 163, 50, 243, 41, 728, 454, 840, 274, 12, 846, 477, 633, 827, 509, 548, 419, 142, 311, 59, 80, 676, 284, 746, 699, 808, 603, 855, 128, 156, 608, 67, 401, 318, 802, 85, 275, 260, 663, 171, 135, 658, 403, 773, 555, 84, 831, 859, 216, 692, 564, 702, 552, 207, 348, 265, 136, 355, 73, 613, 62, 345, 435, 865, 697, 740, 31, 81, 379, 1, 89, 395, 143, 257, 357, 583, 60, 481, 199, 362, 63, 677, 476, 473, 691, 330, 609, 331, 309, 586, 655, 832, 396, 124, 777, 443, 651, 397, 272, 222, 152, 154, 366, 478, 587, 96, 354, 653, 727, 569, 456, 807, 537, 524, 202, 766, 54, 82, 665, 359, 838, 452, 620, 329, 251, 122, 53, 198, 438, 857, 718, 437, 754, 313, 656, 302, 752, 400, 292, 561, 568, 418, 581, 410, 622, 508, 263, 24, 843, 192, 579, 21, 285, 324, 14, 812, 458, 168, 213, 332, 601, 500, 64, 650, 204, 286, 457, 210, 350, 236, 104, 716, 612, 836, 793, 688, 482, 205, 724, 667, 532, 787, 304, 801, 834, 584, 824, 76, 375, 148, 43, 282, 519, 661, 299, 512, 671, 196, 220, 517, 20, 376, 69, 103, 730, 669, 446, 504, 167, 378, 217, 798, 841, 460, 468, 341, 116, 750, 522, 845, 738, 269, 849, 469, 139, 197, 294, 611, 751, 245, 471, 830, 496, 247, 703, 484, 447, 520, 367, 679, 141, 490, 215, 39, 638, 398, 866, 485, 6, 615, 544, 483, 462, 835, 784, 550, 283, 742, 131, 580, 823, 487, 16, 774, 351, 563, 221, 333, 93, 320, 585, 105, 121, 146, 225, 792, 779, 682, 790, 572, 174, 183, 237, 184, 40, 448, 256, 296, 444, 597, 346, 233, 736, 630, 685, 799, 108, 713, 212, 523, 853, 27, 557, 534, 7, 61, 118, 211, 576, 190, 489, 441, 531, 317, 117, 337, 268, 741, 753, 499, 670, 411, 719, 684, 767, 625, 158, 312, 98, 810, 56, 390, 214, 623, 761, 450, 170, 352, 264, 629, 541, 74, 543, 75, 547, 646, 363, 442, 227, 310, 822, 361, 87, 342, 125, 130, 796, 38, 266, 271, 675, 461, 637, 90, 565, 491, 195, 814, 632, 618, 813, 386, 19, 711, 326, 97, 336, 567, 368, 389, 495, 763, 465, 854, 314, 101, 745, 749, 188, 510, 295, 430, 358, 280, 765, 654, 107, 723, 186, 157, 560, 778, 206, 829, 600, 689, 480, 353, 47, 325, 123, 505, 412, 26, 760, 279, 717, 178, 756, 467, 343, 757, 575, 242, 648, 494, 200, 839, 46, 722, 102, 381, 479, 759, 439, 166, 506, 781, 223, 86, 149, 464, 732, 820, 126, 160, 739, 219, 339, 372, 276, 373, 582, 743, 327, 856, 185, 539, 463, 252, 848, 502, 95, 11, 169, 226, 182, 300, 806, 100, 864, 786, 297, 338, 224, 112, 155, 844, 262, 551, 232, 119, 783, 514, 380, 83, 694, 17, 610, 385, 218, 470, 115, 209, 789, 370, 659, 180, 595, 725, 769, 571, 15, 29, 32, 715, 521, 710, 666, 246, 797, 706, 193, 673, 528, 42, 288, 631, 371, 573, 335, 591, 714, 234, 179, 289, 258, 672, 194, 696, 429, 825, 616, 652, 862, 782, 58, 94, 18, 162, 453, 255, 189, 700, 323, 592, 698, 91, 729, 794, 598, 536, 290, 870, 695]
            test_idx = [811, 589, 686, 604, 861, 726, 771, 624, 762, 731, 577, 764, 542, 817, 674, 574, 758, 785, 867, 553, 621, 530, 602, 529, 606, 821, 590, 818, 837, 863, 642, 735, 748, 525, 690, 705, 644, 712, 533, 549, 535, 737, 516, 566, 515, 545, 701, 734, 546, 791, 721, 554, 851, 634, 780, 664, 626, 869, 641, 816, 636, 614, 860, 693, 826, 570, 527, 617, 647, 562, 596, 805]
        elif dataset_name == 'syn5':
            train_idx = [1073, 669, 664, 837, 662, 1178, 538, 237, 298, 886, 552, 630, 361, 822, 624, 333, 493, 682, 686, 876, 36, 481, 752, 243, 1168, 730, 850, 824, 459, 577, 677, 78, 388, 282, 1056, 378, 865, 875, 642, 386, 477, 592, 1107, 962, 666, 435, 128, 328, 487, 890, 90, 168, 484, 561, 408, 659, 1176, 139, 845, 573, 185, 934, 678, 456, 205, 739, 519, 1122, 47, 747, 448, 794, 1033, 1219, 723, 993, 496, 564, 390, 582, 781, 949, 929, 1079, 1110, 578, 737, 1091, 587, 534, 306, 812, 922, 688, 423, 1099, 562, 395, 75, 802, 387, 902, 429, 479, 777, 1230, 923, 990, 1163, 1031, 268, 364, 422, 461, 1133, 1124, 644, 1203, 648, 41, 848, 202, 379, 228, 1032, 458, 175, 410, 546, 421, 240, 701, 725, 330, 1070, 869, 1197, 368, 191, 986, 676, 277, 1067, 1172, 1027, 212, 1095, 1209, 1045, 898, 560, 672, 936, 46, 544, 311, 199, 1224, 1150, 58, 871, 543, 518, 74, 383, 978, 532, 823, 303, 1167, 773, 892, 211, 604, 265, 841, 971, 1223, 344, 417, 1135, 1013, 358, 258, 256, 650, 914, 1025, 259, 1085, 655, 591, 391, 454, 753, 698, 278, 1114, 1191, 542, 918, 609, 105, 1038, 770, 472, 431, 501, 1165, 899, 893, 693, 248, 374, 436, 252, 627, 61, 1141, 541, 734, 178, 879, 1204, 1202, 766, 957, 1097, 870, 1103, 489, 1156, 85, 130, 239, 708, 226, 121, 728, 377, 568, 339, 106, 673, 1094, 549, 166, 29, 1083, 19, 1096, 511, 940, 1059, 293, 49, 530, 603, 1154, 104, 571, 910, 230, 599, 1158, 785, 409, 778, 1187, 943, 88, 528, 87, 131, 1052, 186, 1222, 499, 371, 263, 699, 647, 687, 1123, 281, 1002, 1060, 1130, 229, 476, 334, 83, 101, 1225, 411, 357, 56, 985, 796, 815, 948, 1020, 376, 1216, 710, 679, 792, 288, 533, 896, 1101, 192, 581, 322, 315, 235, 1064, 917, 1169, 290, 646, 631, 1072, 343, 757, 193, 503, 222, 983, 1062, 295, 329, 665, 103, 365, 814, 466, 67, 768, 805, 52, 1131, 1161, 1200, 1186, 855, 526, 570, 113, 760, 291, 1065, 181, 654, 969, 966, 138, 853, 30, 21, 878, 1057, 470, 989, 789, 478, 523, 825, 551, 23, 34, 208, 832, 920, 342, 1183, 1142, 626, 490, 207, 8, 774, 625, 1174, 930, 196, 513, 690, 1036, 11, 784, 231, 309, 1, 909, 177, 392, 639, 132, 313, 1108, 658, 1035, 559, 731, 441, 656, 931, 958, 279, 556, 1051, 751, 671, 1210, 187, 215, 37, 700, 804, 786, 583, 1005, 645, 1152, 537, 705, 527, 416, 809, 394, 420, 1115, 606, 572, 76, 398, 1043, 469, 1129, 588, 576, 953, 621, 1047, 6, 954, 232, 1082, 349, 553, 612, 1028, 204, 405, 286, 124, 206, 336, 244, 520, 321, 475, 684, 704, 352, 1098, 1220, 369, 586, 404, 439, 903, 147, 1088, 53, 1034, 1026, 535, 1113, 82, 453, 380, 1120, 1117, 1037, 251, 795, 133, 510, 924, 66, 262, 742, 1119, 995, 938, 901, 50, 312, 720, 253, 1017, 1206, 91, 1055, 782, 967, 129, 857, 521, 27, 1068, 162, 941, 457, 594, 430, 126, 724, 874, 44, 793, 1078, 10, 426, 937, 1014, 1151, 987, 155, 213, 447, 157, 38, 916, 445, 540, 136, 835, 238, 641, 24, 712, 114, 350, 446, 820, 77, 643, 9, 788, 1145, 158, 683, 506, 146, 999, 1058, 150, 505, 1166, 736, 276, 1006, 255, 622, 885, 942, 727, 632, 882, 702, 732, 1127, 492, 135, 486, 325, 932, 670, 319, 242, 60, 1159, 830, 109, 468, 660, 451, 94, 585, 507, 1049, 696, 233, 102, 689, 434, 26, 326, 1121, 1198, 182, 980, 372, 595, 566, 810, 858, 1184, 860, 1066, 141, 674, 864, 264, 43, 1016, 866, 1132, 783, 601, 267, 210, 485, 711, 254, 1162, 64, 1157, 611, 3, 960, 926, 915, 54, 190, 1226, 1181, 842, 0, 1100, 250, 415, 596, 1010, 80, 1177, 852, 977, 1087, 629, 363, 959, 300, 840, 1089, 945, 973, 1139, 968, 1171, 184, 767, 144, 1144, 667, 502, 801, 933, 1179, 1048, 509, 1081, 833, 18, 302, 455, 1193, 651, 719, 500, 653, 661, 976, 715, 593, 907, 800, 970, 89, 755, 703, 714, 1039, 42, 84, 273, 1102, 341, 14, 418, 25, 1207, 863, 717, 531, 975, 1023, 1011, 362, 462, 982, 63, 442, 597, 623, 1041, 397, 292, 691, 316, 1112, 432, 557, 271, 200, 1044, 952, 201, 1084, 1021, 119, 7, 721, 310, 198, 1148, 834, 209, 474, 1086, 28, 838, 1208, 620, 494, 685, 716, 1199, 550, 1008, 189, 165, 638, 827, 283, 984, 1170, 862, 347, 389, 602, 366, 880, 413, 183, 57, 176, 297, 844, 1093, 1007, 122, 463, 2, 367, 234, 839, 460, 956, 92, 1061, 1146, 979, 887, 895, 1029, 127, 1182, 947, 1054, 1018, 799, 296, 813, 1195, 904, 163, 713, 274, 575, 574, 1175, 738, 1050, 790, 706, 928, 524, 680, 780, 614, 913, 285, 4, 965, 749, 340, 854, 634, 33, 891, 317, 889, 424, 821, 403, 99, 160, 488, 332, 974, 1063, 51, 861, 65, 170, 981, 908, 1153, 266, 272, 1185, 112, 97, 223, 203, 906, 548, 289, 951, 1140, 98, 598, 224, 817, 360, 668, 1000, 536, 877, 218, 270, 406, 508, 675, 221, 140, 1190, 152, 40, 214, 294, 247, 547, 605, 327, 68, 829, 798, 194, 467, 828, 771, 1116, 911, 756, 164, 414, 1030, 558, 348, 1128, 48, 787, 1217, 517, 217, 22, 71, 707, 761, 859, 62, 628, 775, 1205, 174, 95, 955, 143, 884, 393, 748, 744, 512, 79, 888, 765, 743, 452, 169, 17, 110, 86, 589, 1092, 718, 616, 134, 16, 996, 381, 370, 764, 1009, 925, 482, 188, 988, 304, 991, 1046, 754, 495, 39, 1111, 1149, 399, 619, 1105, 73, 763, 1053, 331, 729, 151, 287, 849, 867, 762, 843, 1001, 1196, 772, 356, 355, 836, 260, 219, 427, 818, 1004, 12, 735, 994, 758, 425, 31, 819, 70, 1218]
            test_idx = [617, 972, 912, 726, 584, 964, 1194, 516, 1164, 797, 569, 900, 663, 950, 856, 1180, 1022, 608, 811, 515, 740, 759, 692, 1126, 539, 1024, 873, 745, 1040, 709, 1173, 935, 1076, 992, 997, 635, 1118, 633, 750, 769, 1042, 567, 1189, 1213, 921, 529, 563, 746, 883, 1192, 1137, 791, 741, 847, 733, 905, 1019, 1228, 1155, 1134, 831, 1074, 851, 555, 607, 590, 525, 1229, 1106, 776, 961, 963, 868, 1069, 897, 1143, 649, 722, 806, 652, 545, 872, 946, 807, 600, 939, 1075, 1015, 1227, 522, 1136, 554, 1138, 636, 919, 1147, 610, 1003, 846, 881, 1160, 1214, 1109, 514, 580, 565, 613, 1125, 640, 894, 637, 927, 681, 695, 1104, 1071, 808, 694, 1211, 697, 1090, 615, 944, 1201, 579, 657, 618, 1080, 816]
        indices = [train_idx, test_idx, test_idx]
    else:
        __, indices = split_data(dataset)

    # * Create a GraphData object.
    # max_num_nodes
    max_num_nodes = max((graph.num_nodes for graph in dataset))

    # adj_all
    adj_all = []
    for graph in dataset:
        adj = to_dense_adj(graph.edge_index).squeeze(0).numpy()
        adj = adj + np.eye(adj.shape[0])
        adj_all.append(adj)

    # feature_all
    feature_all = [graph.x.numpy() for graph in dataset]

    # padded
    PADDED = True

    # labels_all
    labels_all = [np.array([data.y.item()], dtype=int) for data in dataset]

    # u_all
    U_NUM = 10
    u_all = [np.random.choice(U_NUM, size=1).astype(float) for __ in range(len(adj_all))]

    # target nodes. These are utilized when we are using a node classification
    # dataset as a graph classification dataset.
    target_nodes = None
    if dataset_name in ['syn1', 'syn4', 'syn5']:
        target_nodes = []
        for data in dataset:
            target_nodes.append(data.target_node)

    # GraphData
    dataset_clear = GraphData(
        adj_all=adj_all,
        features_all=feature_all,
        u_all=u_all,
        labels_all=labels_all,
        max_num_nodes=max_num_nodes,
        padded=PADDED,
        index=None,
        target_nodes=target_nodes,
    )
    data_load = {
        "data": dataset_clear,
        'idx_train_list': [np.array(indices[0])],
        'idx_val_list': [np.array(indices[1])],
        'idx_test_list': [np.array(indices[2])]
    }
    return data_load

if __name__ == '__main__':
    experiment_type = args.experiment_type
    print('running experiment: ', experiment_type)
    if args.robustness and args.experiment_type == 'test':
        for noise in [1, 2, 3, 4, 5]:
            args.noise = noise
            print(f"\n---------- Noise {args.noise} ----------")
            run_clear(args, args.experiment_type)
    else:
        args.noise = None
        if args.robustness and args.experiment_type == 'train':
            print("Clear is an inductive method. You need not train it on the noisy dataset."\
                  "Once trained on the noiseless dataset it should generalize to the noisy one itself.")
            print("Proceeding with training on the noiseless dataset.")
        run_clear(args, args.experiment_type)
