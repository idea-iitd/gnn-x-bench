import torch
import numpy as np
import random
import os
import argparse
import time
from methods.rcexplainer.rcexplainer_helper import RuleMinerLargeCandiPool, evalSingleRule
from torch_geometric.utils import to_dense_adj
from methods.rcexplainer.rcexplainer_helper import ExplainModule, train_explainer, evaluator_explainer
import data_utils
from tqdm import tqdm
import torch.nn.functional as F
from gnn_trainer import GNNTrainer

from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import dense_to_sparse


def get_rce_format(data, node_embeddings):
    num_nodes = [graph.num_nodes for graph in data]
    max_num_nodes = max(num_nodes)
    label = [graph.y for graph in data]
    feat = []
    adj = []
    node_embs_pads = []

    for i, graph in enumerate(data):
        m = torch.nn.ZeroPad2d((0, 0, 0, max_num_nodes - graph.num_nodes))
        feat.append(m(graph.x))
        if graph.edge_index.shape[1] != 0:
            adj.append(to_dense_adj(graph.edge_index, max_num_nodes=max_num_nodes)[0])
        else:
            adj.append(torch.zeros(max_num_nodes, max_num_nodes))
        node_embs_pads.append(m(node_embeddings[i]))

    adj = torch.stack(adj)
    feat = torch.stack(feat)
    label = torch.LongTensor(label)
    num_nodes = torch.LongTensor(num_nodes)
    node_embs_pads = torch.stack(node_embs_pads)

    return adj, feat, label, num_nodes, node_embs_pads


def extract_rules(model, train_data, preds, embs, device, pool_size=50):
    I = 36
    length = 2
    is_graph_classification = True
    pivots_list = []
    opposite_list = []
    rule_list_all = []
    cover_list_all = []
    check_repeat = np.repeat(False, len(train_data))
    rule_dict_list = []
    idx2rule = {}
    _pred_labels = preds.cpu().numpy()
    num_classes = 2
    iter = 0
    for i in range(len(preds)):
        idx2rule[i] = None
    for i in range(len(preds)):
        rule_label = preds[i]

        if iter > num_classes * length - 1:
            if idx2rule[i] != None:
                continue

        if np.sum(check_repeat) >= len(train_data):
            break

        # Initialize the rule. Need to input the label for the rule and which layer we are using (I)
        rule_miner_train = RuleMinerLargeCandiPool(model, train_data, preds, embs, _pred_labels[i], device, I)

        feature = embs[i].float().unsqueeze(0).to(device)

        # Create candidate pool
        rule_miner_train.CandidatePoolLabel(feature, pool=pool_size)

        # Perform rule extraction
        array_labels = preds.long().cpu().numpy()  # train_data_upt._getArrayLabels()
        inv_classifier, pivots, opposite, initial = rule_miner_train.getInvariantClassifier(i, feature.cpu().numpy(), preds[i].cpu(), array_labels, delta_constr_=0)
        pivots_list.append(pivots)
        opposite_list.append(opposite)

        # saving info for gnnexplainer
        rule_dict = {}
        inv_bbs = inv_classifier._bb.bb

        inv_invariant = inv_classifier._invariant
        boundaries_info = []
        b_count = 0
        assert (len(opposite) == np.sum(inv_invariant))
        for inv_ix in range(inv_invariant.shape[0]):
            if inv_invariant[inv_ix] == False:
                continue
            boundary_dict = {}
            # boundary_dict['basis'] = inv_bbs[:-1,inv_ix]
            boundary_dict['basis'] = inv_bbs[:, inv_ix]
            boundary_dict['label'] = opposite[b_count]
            b_count += 1
            boundaries_info.append(boundary_dict)
        rule_dict['boundary'] = boundaries_info
        rule_dict['label'] = rule_label.cpu().item()
        print("Rules extracted: ", rule_dict)
        rule_dict_list.append(rule_dict)
        # end saving info for gnn-explainer

        # evaluate classifier
        accuracy_train, cover_indi_train = evalSingleRule(inv_classifier, train_data, embs, preds)
        assert (cover_indi_train[i] == True)
        for c_ix in range(cover_indi_train.shape[0]):
            if cover_indi_train[c_ix] == True:
                if is_graph_classification:
                    idx2rule[c_ix] = len(rule_list_all)
                else:
                    if c_ix not in idx2rule:
                        idx2rule[c_ix] = []
                    idx2rule[c_ix].append(len(rule_list_all))

        rule_list_all.append(inv_classifier)
        cover_list_all.append(cover_indi_train)
        for j in range(len(train_data)):
            if cover_indi_train[j] == True:
                check_repeat[j] = True
        iter += 1

    rule_dict_save = {}
    rule_dict_save['rules'] = rule_dict_list
    rule_dict_save['idx2rule'] = idx2rule
    return rule_dict_save


parser = argparse.ArgumentParser()

parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--opt', default='adam')
parser.add_argument('--opt_scheduler', default='none')
parser.add_argument('--use_tuned_parameters', action='store_true')

# TODO: make args.lambda_ 0.0 for counterfactuals
parser.add_argument('--lambda_', default=1.0, type=float, help="The hyperparameter of L_same, (1 - lambda) will be for L_opp")
parser.add_argument('--mu_', default=0.0, type=float, help="The hyperparameter of L_entropy, makes the weights more close to 0 or 1")
parser.add_argument('--beta_', default=0.0000001, type=float, help="The hyperparameter of L_sparsity, makes the explanations more sparse")

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--dataset', type=str, default='Mutagenicity',
                    choices=['Mutagenicity', 'Proteins', 'Mutag', 'IMDB-B', 'AIDS', 'NCI1', 'Tree-of-Life', 'Graph-SST2', 'DD', 'REDDIT-B', 'ogbg_molhiv'],
                    help="Dataset name")
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--gnn_run', type=int, default=1)
parser.add_argument('--explainer_run', type=int, default=1)
parser.add_argument('--gnn_type', type=str, default='gcn', choices=['gcn', 'gat', 'gin', 'sage'])
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--robustness', type=str, default='na', choices=['topology_random', 'topology_adversarial', 'feature', 'na'], help="na by default means we do not run for perturbed data")

args = parser.parse_args()

# Logging.
result_folder = f'data/{args.dataset}/rcexplainer_{args.lambda_}/'
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
counterfactuals_path = os.path.join(result_folder, f'counterfactuals_{args.gnn_type}_run_{args.explainer_run}.pt')
args.method = 'classification'

trainer = GNNTrainer(dataset_name=args.dataset, gnn_type=args.gnn_type, task='basegnn', device=args.device)
model = trainer.load(args.gnn_run)
for param in model.parameters():
    param.requires_grad = False
model.eval()

node_embeddings, graph_embeddings, outs = trainer.load_gnn_outputs(args.gnn_run)
preds = torch.argmax(outs, dim=-1)

train_indices = indices[0]
val_indices = indices[1]

# rule extraction
rule_folder = f'rcexplainer_rules/'
if not os.path.exists(rule_folder):
    os.makedirs(rule_folder)
rule_path = os.path.join(rule_folder, f'rcexplainer_{args.dataset}_{args.gnn_type}_rule_dict_run_{args.explainer_run}.npy')
if os.path.exists(rule_path):
    rule_dict = np.load(rule_path, allow_pickle=True).item()
else:
    rule_dict = extract_rules(model, dataset, preds, graph_embeddings, device, pool_size=min([100, (preds == 1).sum().item(), (preds == 0).sum().item()]))
    np.save(rule_path, rule_dict)

# setting seed again because of rule extraction
torch.manual_seed(args.explainer_run)
torch.cuda.manual_seed(args.explainer_run)
np.random.seed(args.explainer_run)
random.seed(args.explainer_run)

if args.dataset in ['Graph-SST2']:
    args.lr = args.lr * 0.05  # smaller lr for large dataset
    args.beta_ = args.beta_ * 10  # to make the explanations more sparse

adj, feat, label, num_nodes, node_embs_pads = get_rce_format(dataset, node_embeddings)
explainer = ExplainModule(
    num_nodes=adj.shape[1],
    emb_dims=model.dim * 2,  # gnn_model.num_layers * 2,
    device=device,
    args=args
)

if args.dataset in ['Mutagenicity']:
    args.beta_ = args.beta_ * 30

if args.dataset in ['NCI1']:
    args.beta_ = args.beta_ * 300

if args.robustness == 'na':
    explainer, last_epoch = train_explainer(explainer, model, rule_dict, adj, feat, label, preds, num_nodes, graph_embeddings, node_embs_pads, args, train_indices, val_indices, device)
    all_loss, all_explanations = evaluator_explainer(explainer, model, rule_dict, adj, feat, label, preds, num_nodes, graph_embeddings, node_embs_pads, range(len(dataset)), device)
    explanation_graphs = []
    entered = 0
    if (args.lambda_ != 0.0):
        counterfactual_graphs = []
    for i, graph in enumerate(dataset):
        entered += 1
        explanation = all_explanations[i]
        explanation_adj = torch.from_numpy(explanation[:graph.num_nodes][:, :graph.num_nodes])
        edge_index = graph.edge_index
        edge_weight = explanation_adj[[index[0] for index in graph.edge_index.T], [index[1] for index in graph.edge_index.T]]

        if (args.lambda_ != 0.0):
            d = Data(edge_index=edge_index.clone(), edge_weight=edge_weight.clone(), x=graph.x.clone(), y=graph.y.clone())
            c = Data(edge_index=edge_index.clone(), edge_weight=(1 - edge_weight).clone(), x=graph.x.clone(), y=graph.y.clone())
            explanation_graphs.append(d)
            counterfactual_graphs.append(c)
        else:
            # print('A')
            # added edge attributes to graphs, in order to take a forward pass with edge attributes and check if label changes for finding counterfactual explanation.
            d = Data(edge_index=edge_index.clone(), edge_weight=edge_weight.clone(), x=graph.x.clone(), y=graph.y.clone(), edge_attr=graph.edge_attr.clone() if graph.edge_attr is not None else None)
            c = Data(edge_index=edge_index.clone(), edge_weight=(1 - edge_weight).clone(), x=graph.x.clone(), y=graph.y.clone(), edge_attr=graph.edge_attr.clone() if graph.edge_attr is not None else None)
            label = int(graph.y)
            pred = model(d.to(args.device), d.edge_weight.to(args.device))[-1][0]
            pred_cf = model(c.to(args.device), c.edge_weight.to(args.device))[-1][0]
            explanation_graphs.append({
                "graph": d.cpu(), "graph_cf": c.cpu(),
                "label": label, "pred": pred.cpu(), "pred_cf": pred_cf.cpu()
            })

    print("check: ", entered, len(dataset))
    torch.save(explanation_graphs, explanations_path)
    if (args.lambda_ != 0.0):
        torch.save(counterfactual_graphs, counterfactuals_path)
elif args.robustness == 'topology_random':
    explainer.load_state_dict(torch.load(best_explainer_model_path, map_location=device))
    for noise in [1, 2, 3, 4, 5]:
        explanations_path = os.path.join(result_folder, f'explanations_{args.gnn_type}_run_{args.explainer_run}_noise_{noise}.pt')
        if (args.lambda_ != 0.0):
            counterfactuals_path = os.path.join(result_folder, f'counterfactuals_{args.gnn_type}_run_{args.explainer_run}_noise_{noise}.pt')
        explanation_graphs = []
        noisy_dataset = data_utils.load_dataset(data_utils.get_noisy_dataset_name(dataset_name=args.dataset, noise=noise))
        with torch.no_grad():
            node_embeddings = []
            graph_embeddings = []
            outs = []
            for graph in noisy_dataset:
                node_embedding, graph_embedding, out = model(graph.to(device))
                node_embeddings.append(node_embedding)
                graph_embeddings.append(graph_embedding)
                outs.append(out)
            graph_embeddings = torch.cat(graph_embeddings)
            outs = torch.cat(outs)
            preds = torch.argmax(outs, dim=-1)
        adj, feat, label, num_nodes, node_embs_pads = get_rce_format(noisy_dataset, node_embeddings)
        all_loss, all_explanations = evaluator_explainer(explainer, model, rule_dict, adj, feat, label, preds, num_nodes, graph_embeddings, node_embs_pads, range(len(noisy_dataset)), device)
        explanation_graphs = []
        if (args.lambda_ != 0.0):
            counterfactual_graphs = []
        for i, graph in enumerate(noisy_dataset):
            explanation = all_explanations[i]
            explanation_adj = torch.from_numpy(explanation[:graph.num_nodes][:, :graph.num_nodes])
            edge_index = graph.edge_index
            edge_weight = explanation_adj[[index[0] for index in graph.edge_index.T], [index[1] for index in graph.edge_index.T]]
            if (args.lambda_ != 0.0):
                d = Data(edge_index=edge_index.clone(), edge_weight=edge_weight.clone(), x=graph.x.clone(), y=graph.y.clone())
                c = Data(edge_index=edge_index.clone(), edge_weight=(1 - edge_weight).clone(), x=graph.x.clone(), y=graph.y.clone())
                explanation_graphs.append(d)
                counterfactual_graphs.append(c)
            else:
                # added edge attributes to graphs, in order to take a forward pass with edge attributes and check if label changes for finding counterfactual explanation.
                d = Data(edge_index=edge_index.clone(), edge_weight=edge_weight.clone(), x=graph.x.clone(), y=graph.y.clone(), edge_attr=graph.edge_attr.clone() if graph.edge_attr is not None else None)
                c = Data(edge_index=edge_index.clone(), edge_weight=(1 - edge_weight).clone(), x=graph.x.clone(), y=graph.y.clone(), edge_attr=graph.edge_attr.clone() if graph.edge_attr is not None else None)
                label = int(graph.y)
                pred = model(d.to(args.device), d.edge_weight.to(args.device))[-1][0]
                pred_cf = model(c.to(args.device), c.edge_weight.to(args.device))[-1][0]
                explanation_graphs.append({
                    "graph": d.cpu(), "graph_cf": c.cpu(),
                    "label": label, "pred": pred.cpu(), "pred_cf": pred_cf.cpu()
                })

        torch.save(explanation_graphs, explanations_path)
        if (args.lambda_ != 0.0):
            torch.save(counterfactual_graphs, counterfactuals_path)
elif args.robustness == 'feature':
    explainer.load_state_dict(torch.load(best_explainer_model_path, map_location=device))
    for noise in [10, 20, 30, 40, 50]:
        explanations_path = os.path.join(result_folder, f'explanations_{args.gnn_type}_run_{args.explainer_run}_feature_noise_{noise}.pt')
        if (args.lambda_ != 0.0):
            counterfactuals_path = os.path.join(result_folder, f'counterfactuals_{args.gnn_type}_run_{args.explainer_run}_feature_noise_{noise}.pt')
        explanation_graphs = []
        noisy_dataset = data_utils.load_dataset(data_utils.get_noisy_feature_dataset_name(dataset_name=args.dataset, noise=noise))
        with torch.no_grad():
            node_embeddings = []
            graph_embeddings = []
            outs = []
            for graph in noisy_dataset:
                node_embedding, graph_embedding, out = model(graph.to(device))
                node_embeddings.append(node_embedding)
                graph_embeddings.append(graph_embedding)
                outs.append(out)
            graph_embeddings = torch.cat(graph_embeddings)
            outs = torch.cat(outs)
            preds = torch.argmax(outs, dim=-1)
        adj, feat, label, num_nodes, node_embs_pads = get_rce_format(noisy_dataset, node_embeddings)
        all_loss, all_explanations = evaluator_explainer(explainer, model, rule_dict, adj, feat, label, preds, num_nodes, graph_embeddings, node_embs_pads, range(len(noisy_dataset)), device)
        explanation_graphs = []
        if (args.lambda_ != 0.0):
            counterfactual_graphs = []
        for i, graph in enumerate(noisy_dataset):
            explanation = all_explanations[i]
            explanation_adj = torch.from_numpy(explanation[:graph.num_nodes][:, :graph.num_nodes])
            edge_index = graph.edge_index
            edge_weight = explanation_adj[[index[0] for index in graph.edge_index.T], [index[1] for index in graph.edge_index.T]]
            if (args.lambda_ != 0.0):
                d = Data(edge_index=edge_index.clone(), edge_weight=edge_weight.clone(), x=graph.x.clone(), y=graph.y.clone())
                c = Data(edge_index=edge_index.clone(), edge_weight=(1 - edge_weight).clone(), x=graph.x.clone(), y=graph.y.clone())
                explanation_graphs.append(d)
                counterfactual_graphs.append(c)
            else:
                # added edge attributes to graphs, in order to take a forward pass with edge attributes and check if label changes for finding counterfactual explanation.
                d = Data(edge_index=edge_index.clone(), edge_weight=edge_weight.clone(), x=graph.x.clone(), y=graph.y.clone(), edge_attr=graph.edge_attr.clone() if graph.edge_attr is not None else None)
                c = Data(edge_index=edge_index.clone(), edge_weight=(1 - edge_weight).clone(), x=graph.x.clone(), y=graph.y.clone(), edge_attr=graph.edge_attr.clone() if graph.edge_attr is not None else None)
                label = int(graph.y)
                pred = model(d.to(args.device), d.edge_weight.to(args.device))[-1][0]
                pred_cf = model(c.to(args.device), c.edge_weight.to(args.device))[-1][0]
                explanation_graphs.append({
                    "graph": d.cpu(), "graph_cf": c.cpu(),
                    "label": label, "pred": pred.cpu(), "pred_cf": pred_cf.cpu()
                })

        torch.save(explanation_graphs, explanations_path)
        if (args.lambda_ != 0.0):
            torch.save(counterfactual_graphs, counterfactuals_path)
elif args.robustness == 'topology_adversarial':
    explainer.load_state_dict(torch.load(best_explainer_model_path, map_location=device))
    for flip_count in [1, 2, 3, 4, 5]:
        explanations_path = os.path.join(result_folder, f'explanations_{args.gnn_type}_run_{args.explainer_run}_topology_adversarial_{flip_count}.pt')
        if (args.lambda_ != 0.0):
            counterfactuals_path = os.path.join(result_folder, f'counterfactuals_{args.gnn_type}_run_{args.explainer_run}_topology_adversarial_{flip_count}.pt')
        explanation_graphs = []
        noisy_dataset = data_utils.load_dataset(data_utils.get_topology_adversarial_attack_dataset_name(dataset_name=args.dataset, flip_count=flip_count))
        with torch.no_grad():
            node_embeddings = []
            graph_embeddings = []
            outs = []
            for graph in noisy_dataset:
                node_embedding, graph_embedding, out = model(graph.to(device))
                node_embeddings.append(node_embedding)
                graph_embeddings.append(graph_embedding)
                outs.append(out)
            graph_embeddings = torch.cat(graph_embeddings)
            outs = torch.cat(outs)
            preds = torch.argmax(outs, dim=-1)
        adj, feat, label, num_nodes, node_embs_pads = get_rce_format(noisy_dataset, node_embeddings)
        all_loss, all_explanations = evaluator_explainer(explainer, model, rule_dict, adj, feat, label, preds, num_nodes, graph_embeddings, node_embs_pads, range(len(noisy_dataset)), device)
        explanation_graphs = []
        if (args.lambda_ != 0.0):
            counterfactual_graphs = []
        for i, graph in enumerate(noisy_dataset):
            explanation = all_explanations[i]
            explanation_adj = torch.from_numpy(explanation[:graph.num_nodes][:, :graph.num_nodes])
            edge_index = graph.edge_index
            edge_weight = explanation_adj[[index[0] for index in graph.edge_index.T], [index[1] for index in graph.edge_index.T]]
            if (args.lambda_ != 0.0):
                d = Data(edge_index=edge_index.clone(), edge_weight=edge_weight.clone(), x=graph.x.clone(), y=graph.y.clone())
                c = Data(edge_index=edge_index.clone(), edge_weight=(1 - edge_weight).clone(), x=graph.x.clone(), y=graph.y.clone())
                explanation_graphs.append(d)
                counterfactual_graphs.append(c)
            else:
                # added edge attributes to graphs, in order to take a forward pass with edge attributes and check if label changes for finding counterfactual explanation.
                d = Data(edge_index=edge_index.clone(), edge_weight=edge_weight.clone(), x=graph.x.clone(), y=graph.y.clone(), edge_attr=graph.edge_attr.clone() if graph.edge_attr is not None else None)
                c = Data(edge_index=edge_index.clone(), edge_weight=(1 - edge_weight).clone(), x=graph.x.clone(), y=graph.y.clone(), edge_attr=graph.edge_attr.clone() if graph.edge_attr is not None else None)
                label = int(graph.y)
                pred = model(d.to(args.device), d.edge_weight.to(args.device))[-1][0]
                pred_cf = model(c.to(args.device), c.edge_weight.to(args.device))[-1][0]
                explanation_graphs.append({
                    "graph": d.cpu(), "graph_cf": c.cpu(),
                    "label": label, "pred": pred.cpu(), "pred_cf": pred_cf.cpu()
                })

        torch.save(explanation_graphs, explanations_path)
        if (args.lambda_ != 0.0):
            torch.save(counterfactual_graphs, counterfactuals_path)
else:
    raise NotImplementedError
