#reference: https://github.com/GraphFramEx/graphframex/blob/main/code/utils/plot_utils.py
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import torch
import argparse
from gnn_trainer import GNNTrainer
import data_utils
import cf_metrics as metrics
from torch_geometric.utils import to_networkx
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Mutag', choices=['Mutagenicity', 'Mutag'], help="Dataset name")
    parser.add_argument('--gnn_type', type=str, default='gcn', choices=['gcn', 'gat', 'gin', 'sage'], help='GNN layer type to use.')
    parser.add_argument('--device', type=int, default=0, help='Index of cuda device to use. Default is 0.')
    parser.add_argument('--gnn_run', type=int, default=1, help='random seed for gnn run')
    parser.add_argument('--explainer_run', type=int, default=1, help='random seed for explainer run')
    parser.add_argument('--explainer_name', type=str, default='cff_0.0', choices=['cff_0.0','rcexplainer_0.0', 'clear', 'pgexplainer', 'tagexplainer_1', 'tagexplainer_2', 'cff_1.0',
                                                               'rcexplainer_1.0', 'gnnexplainer', 'gem', 'subgraphx'], help='Name of explainer to use.')
    parser.add_argument('--verbose', type=int, default=1, help='Default: 1 (print computed metric), else 0')
    return parser.parse_args()

# code for counterfactual explanation visualization
def find_cf_edges(data_dict, explainer_name):
    relu = torch.nn.ReLU()
    orig_adj, cf_adj = metrics.get_adj_mat(data_dict, explainer_name)
    
    perbs = (orig_adj - cf_adj)
    perbs_del = relu(torch.from_numpy(perbs)).numpy()
    perbs_add = relu(torch.from_numpy(-perbs)).numpy()

    l1 = np.transpose(np.nonzero(perbs_del)).tolist()
    perbs_list = []
    perb_type = []
    for l in l1:
        if((l[1], l[0]) not in perbs_list):
            perbs_list.append((l[0], l[1]))
            perb_type.append('del')
    
    l1 = np.transpose(np.where(perbs_add == 1)).tolist()
    for l in l1:
        if((l[1], l[0]) not in perbs_list):
            perbs_list.append((l[0], l[1]))
            perb_type.append('add')
    
    # print(perbs_list)
    # print(perb_type)
    return perbs_list, perb_type 
    
    
def plot_cfexplanation(data_dict, result_folder, explainer_name, gid):
    MUTAG_NODE_LABELS = {0: "C", 1: "N", 2: "O", 3: "F", 4: "I", 5: "Cl", 6: "Br"}
    data, label = data_dict['graph'], data_dict['label']

    atoms = np.argmax(data.x.cpu().detach().numpy(), axis=1)
    nmb_nodes = data.num_nodes
    max_label = np.max(atoms) + 1
    
    colors = [
        "orange",
        "red",
        "lime",
        "green",
        "blue",
        "orchid",
        "darksalmon",
        "darkslategray",
        "gold",
        "bisque",
        "tan",
        "lightseagreen",
        "indigo",
        "navy",
    ]
    
    #plot graph
    g = to_networkx(data, to_undirected=True, remove_self_loops=True)
    # print(g.nodes)
    pos = nx.kamada_kawai_layout(g)
    
    node_labels = {i: MUTAG_NODE_LABELS[node_feat] for i, node_feat in enumerate(atoms)}

    label2nodes = []
    for i in range(max_label):
        label2nodes.append([])
    for i in range(nmb_nodes):
        if i in g.nodes():
            label2nodes[atoms[i]].append(i)
    for i in range(max_label):
        node_filter = []
        for j in range(len(label2nodes[i])):
            node_filter.append(label2nodes[i][j])

        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=node_filter,
            node_color=colors[i],
            node_size=300,
            label=MUTAG_NODE_LABELS[i],
        )
    #find cf edges
    perbs_list, perb_type = find_cf_edges(data_dict, explainer_name)
    # add perb edges to graph
    for i, cf in enumerate(perbs_list):
        if perb_type[i] == 'add':
            g.add_edge(cf[0], cf[1])
   
    #assign colors
    edge_color_map = list()
    for edge in g.edges:
        if edge not in perbs_list :
            edge_color_map.append("black")
            continue
        index = perbs_list.index(edge)
        action = perb_type[index]
        if action == 'del':
            edge_color_map.append("red")
        else:
            edge_color_map.append("green")  
            
    #plot cf edges  
    nx.draw_networkx_edges(
        g,
        pos,
        edge_color=edge_color_map, 
        width=2,
    )
    
    nx.draw_networkx_labels(g, pos, node_labels, font_size=15, font_color="black")

    plt.title("Graph: " + str(gid) + " - label: " + str(label))
    plt.axis("off")
    plt.savefig(result_folder+f'/exp_{gid}.png')
    plt.clf()
    plt.close()

#get top-k edges in factual explanation
def get_topk_edges(graph, k=5):
    directed_edge_weight = graph.edge_weight[graph.edge_index[0] <= graph.edge_index[1]]
    directed_edge_index = graph.edge_index[:, graph.edge_index[0] <= graph.edge_index[1]]

    threshold = directed_edge_weight.topk(min(k, directed_edge_weight.shape[0]))[0][-1]
    #pick top-k edges
    idx = directed_edge_weight >= threshold
    directed_edge_index = directed_edge_index[:, idx].numpy()
    
    perbs_list = []
    for i in range(directed_edge_index.shape[1]):
        perbs_list.append((directed_edge_index[:,i][0], directed_edge_index[:,i][1]))
        
    return perbs_list
# code for factual explanation visualization
def plot_fac_explanation(graph, result_folder, gid):
    MUTAG_NODE_LABELS = {0: "C", 1: "N", 2: "O", 3: "F", 4: "I", 5: "Cl", 6: "Br"}
    data, label = graph, graph.y.item()
    
    plt.figure()

    atoms = np.argmax(data.x.cpu().detach().numpy(), axis=1)
    nmb_nodes = data.num_nodes
    max_label = np.max(atoms) + 1
    
    colors = [
        "orange",
        "red",
        "lime",
        "green",
        "blue",
        "orchid",
        "darksalmon",
        "darkslategray",
        "gold",
        "bisque",
        "tan",
        "lightseagreen",
        "indigo",
        "navy",
    ]
    
    #plot graph
    g = to_networkx(data, to_undirected=True, remove_self_loops=True)
    pos = nx.kamada_kawai_layout(g)
    
    node_labels = {i: MUTAG_NODE_LABELS[node_feat] for i, node_feat in enumerate(atoms)}

    label2nodes = []
    for i in range(max_label):
        label2nodes.append([])
    for i in range(nmb_nodes):
        if i in g.nodes():
            label2nodes[atoms[i]].append(i)
    for i in range(max_label):
        node_filter = []
        for j in range(len(label2nodes[i])):
            node_filter.append(label2nodes[i][j])
            
        # print(colors[i], node_filter, MUTAG_NODE_LABELS[i])
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=node_filter,
            node_color=colors[i],
            node_size=300,
            label=MUTAG_NODE_LABELS[i],
        )
    #find factual edges - top-k
    perbs_list = get_topk_edges(graph)
    
    #assign colors
    edge_color_map = list()
    for edge in g.edges:
        if edge not in perbs_list :
            edge_color_map.append("black")
            continue
        edge_color_map.append("red")
        
    #plot edges
    nx.draw_networkx_edges(
    g,
    pos,
    edge_color=edge_color_map, 
    width=2,
    )
    
    nx.draw_networkx_labels(g, pos, node_labels, font_size=15, font_color="black")

    plt.title("Graph: " + str(gid) + " - label: " + str(label))
    plt.axis("off")
    plt.savefig(result_folder+f'/exp_{gid}.png')
    plt.clf()
    plt.close()

if __name__ == '__main__':
    args = parse_args()
    cfac_baselines = ['cff_0.0','rcexplainer_0.0', 'clear']
    fac_baselines = ['pgexplainer', 'tagexplainer_1', 'tagexplainer_2', 'cff_1.0','rcexplainer_1.0', 'gnnexplainer', 'gem', 'subgraphx']
    
    trainer = GNNTrainer(dataset_name=args.dataset, gnn_type=args.gnn_type, task='basegnn', device=args.device)
    model = trainer.load(args.gnn_run)
    model.eval()

    result_folder = f'data/{args.dataset}/{args.explainer_name}/plot/'
    isExist = os.path.exists(result_folder)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(result_folder)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    dataset = data_utils.load_dataset(args.dataset)
    splits, indices = data_utils.split_data(dataset)
    test_indices = indices[2]
    
    explanations = data_utils.load_explanations(args.dataset, args.explainer_name, args.gnn_type, torch.device('cpu'), args.explainer_run)
    
    if(args.explainer_name in cfac_baselines):
        #generate score
        if(args.explainer_name == 'rcexplainer_0.0'):
            explanations_updated = metrics.remove_top_k_incremental(explanations, model, device)
            explanations = explanations_updated
        
        if(args.explainer_name == 'clear'):
            test_indices = np.arange(len(explanations))
            
        for i, idx in enumerate(test_indices):
            plot_cfexplanation(explanations[idx], result_folder, args.explainer_name, i) 
    elif(args.explainer_name in fac_baselines):
        if(args.explainer_name == 'subgraphx'):
            test_indices = np.arange(len(explanations))
            
        for i, idx in enumerate(test_indices):
            plot_fac_explanation(explanations[idx], result_folder, i) 
    else:
        raise NotImplementedError
