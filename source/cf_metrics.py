#env: rc_fac
import torch
from torch_geometric.utils import to_dense_adj, is_undirected
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import RemoveIsolatedNodes, ToUndirected
from torch_geometric.utils import to_networkx
import networkx as nx
import math

def mean_stdev(stdev_list, num_samples=5, include_nan=False):
    total_sum = []
    non_nan = 0
    for s in stdev_list:
        if(math.isnan(s) == False):
            total_sum.append(s*s)
            non_nan += 1
        else:
            total_sum.append(0)
    if(include_nan):
        np.round(math.sqrt(np.sum(total_sum)/num_samples),3)
    else:
        return np.round(math.sqrt(np.sum(total_sum)/non_nan),3) if (non_nan!=0) else float('NaN')

def mean_samples(sample_list, include_nan=False):
    if(include_nan):
        mean = np.sum([val for val in sample_list if not np.isnan(val)])/len(sample_list)
        return np.round(mean, 3) if len(sample_list) > 0 else float('NaN')
    else:
        return np.round(np.nanmean(sample_list),3)

def stdev_samples(sample_list, include_nan=False):
    if(include_nan):
        mean = mean_samples(sample_list)
        std = np.sqrt(np.sum([abs(val-mean)**2 for val in sample_list if not np.isnan(val)])/len(sample_list))
        return np.round(std, 3) if len(sample_list) > 0 else float('NaN')
    else:
        return np.round(np.nanstd(sample_list),3)

def remove_top_k(model, explanation, k=1):
    directed_edge_weight = explanation.edge_weight[explanation.edge_index[0] <= explanation.edge_index[1]]
    directed_edge_index = explanation.edge_index[:, explanation.edge_index[0] <= explanation.edge_index[1]]
    directed_edge_attr = explanation.edge_attr[explanation.edge_index[0] <= explanation.edge_index[1],:] if explanation.edge_attr is not None else None#shape: num_edges * attr_dim
    #remove top-k edges
    idx = torch.ones(directed_edge_weight.numel(), dtype=torch.bool)
    del_idx = directed_edge_weight.topk(min(k, directed_edge_weight.shape[0]), largest=False).indices 
    edge_mask = torch.ones(directed_edge_weight.shape[0])
    edge_mask[del_idx] = 0
    directed_edge_weight[del_idx] = 0
    
    new_data = Data(
        edge_index=directed_edge_index.clone(),
        x=explanation.x.clone(),
        edge_weight=directed_edge_weight.clone(),
        edge_mask = edge_mask,
        edge_attr=directed_edge_attr.clone() if directed_edge_attr is not None else None    
    )
    
    # make graph undirected
    new_data = ToUndirected()(new_data)    
    # not removing isolated nodes to prevent change in graph max nodes
    # new_data = RemoveIsolatedNodes()(new_data)

    #return edge_index after removing top-k edges
    return new_data
    
def remove_top_k_incremental(explanations, model, device):
    for i, exp in enumerate(explanations):
        pred_orig =  torch.argmax(model(exp['graph'].to(device), exp['graph'].edge_weight.to(device))[-1][0])
        pred_cf_orig = torch.argmax(model(exp['graph_cf'].to(device), exp['graph_cf'].edge_weight.to(device))[-1][0])
        pred_cf = pred_orig
        k = 1
        final_cf = exp['graph_cf']
        final_pred_cf = exp['pred_cf']
  
        if(pred_cf_orig.item() != pred_cf.item()):    
            # print(f'---------------------- node: {i} --------------------')
            while((pred_cf == pred_orig) and k<=exp['graph'].edge_index.shape[1]):
                cf_data = remove_top_k(model, exp['graph_cf'], k)
                pred_cf_logits = model(cf_data.to(device), cf_data.edge_weight.to(device))[-1][0]
                pred_cf = torch.argmax(pred_cf_logits)
                if(pred_cf != pred_orig):
                    final_cf = cf_data
                    final_pred_cf = pred_cf_logits
                    break
                k+=1   
                  
        exp['graph_cf_up'] = final_cf
        exp['pred_cf_up'] = final_pred_cf
        
    return explanations
    
def get_adj_mat(data_dict, explainer, device=0):
    max_num_nodes = max(data_dict['graph'].num_nodes, data_dict['graph_cf'].num_nodes)
    orig_adj = to_dense_adj(data_dict['graph'].edge_index, max_num_nodes=max_num_nodes)[0].cpu().detach().numpy()
    if(explainer == 'rcexplainer_0.0'):
        try:
            edge_attr = data_dict['graph_cf_up'].edge_mask
        except:
            edge_attr = torch.ones(data_dict['graph_cf_up'].edge_weight.shape[0]).to(f'cuda:{device}') #hard coded
        #second attribute prevents removing isolated nodes from adj hence change in shape of matrix
        cf_adj = to_dense_adj(
        data_dict['graph_cf_up'].edge_index,
        edge_attr = edge_attr, 
        max_num_nodes=max_num_nodes
        )[0].cpu().detach().numpy()
    else:
        if(explainer == 'clear'):
            edge_attr_val  = None
        else:
            edge_attr_val = data_dict['graph_cf'].edge_weight
        cf_adj = to_dense_adj(
            data_dict['graph_cf'].edge_index,
            edge_attr = edge_attr_val, 
            max_num_nodes=max_num_nodes
        )[0].cpu().detach().numpy()
    return orig_adj, cf_adj


# code for counterfactual explanation visualization
def find_cf_edges(data_dict, explainer_name):
    relu = torch.nn.ReLU()
    orig_adj, cf_adj = get_adj_mat(data_dict, explainer_name)
    
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

    return perbs_list, perb_type 

def sufficiency(explainer, explanations, indices):
    same = 0
    
    for i in indices:
        data_dict = explanations[i] 
        pred_orig = torch.argmax(data_dict['pred']).item()
        pred_cf = torch.argmax(data_dict['pred_cf']).item()

        #sufficiency = number of instances whose label remains the same, or for whom we are unable to find a counterfactual
        if (pred_orig == pred_cf):
            same += 1
    
    return np.round((same/len(indices)), 4)

def size(explainer, explanations, indices):
    #check if graph is undirected, if true divide size by 2
    undirected = is_undirected(explanations[0]['graph'].edge_index)
    size = []
                
    for i in indices:
        data_dict = explanations[i] 
        pred_orig = torch.argmax(data_dict['pred']).item()
        pred_cf = torch.argmax(data_dict['pred_cf']).item()
        
        if(pred_orig != pred_cf):
            #initialize inside the if condition, since graph_cf_up would not exist if cf does not exist
            orig_adj, cf_adj = get_adj_mat(data_dict, explainer)
            if(sum(sum(abs(orig_adj - cf_adj))) == 0):
                continue
            
            if(undirected):
                size.append(sum(sum(abs(orig_adj - cf_adj)))/2)
            else:
                size.append(sum(sum(abs(orig_adj - cf_adj))))    
       
    return np.round(np.mean(size), 4), np.round(np.std(size), 4)

def sparsity(explainer, explanations, indices):
    relu = torch.nn.ReLU()
    sparsity = []
    for i in indices:
        data_dict = explanations[i] 
        pred_orig = torch.argmax(data_dict['pred']).item()
        pred_cf = torch.argmax(data_dict['pred_cf']).item()
        
        if(pred_orig != pred_cf):
            orig_adj, cf_adj = get_adj_mat(data_dict, explainer)
            if(sum(sum(abs(orig_adj - cf_adj))) == 0):
                continue
            #for sparsity
            perbs = (orig_adj - cf_adj)
            perbs_del = relu(torch.from_numpy(perbs)).numpy()
            # perbs_add = relu(-perbs)
            # sparsity = num deletions/ num possible deletions
            sparsity.append(1 - (sum(sum(perbs_del)) / data_dict['graph'].edge_index.shape[1]))
    
    return np.round(np.mean(sparsity),4)

#computes jaccard similarity between generated counterfactual graphs
def jaccard_cf_graph(explainer, data_dict1, data_dict2):

    _, cf_adj1 = get_adj_mat(data_dict1, explainer)
    _, cf_adj2 = get_adj_mat(data_dict2, explainer)
    
    #take out non-zero (r,c) tuples cf1 list
    l1 = np.transpose(np.nonzero(cf_adj1)).tolist()
    l1 = set([(l[0], l[1]) for l in l1])
    
    #take out non-zero (r,c) tuples cf2 list
    l2 = np.transpose(np.nonzero(cf_adj2)).tolist()
    l2 = set([(l[0], l[1]) for l in l2])
       
    #return intersection/union
    return float(len(l1.intersection(l2))) / len(l2.union(l1))

def robustness(explainer, explanation_1, explanation_2, indices):
    jaccard = []
    for i in indices:
        data_dict1 = explanation_1[i]
        data_dict2 = explanation_2[i]
        pred_orig1 = torch.argmax(data_dict1['pred']).item()
        pred_cf1 = torch.argmax(data_dict1['pred_cf']).item()
        
        pred_orig2 = torch.argmax(data_dict2['pred']).item()
        pred_cf2 = torch.argmax(data_dict2['pred_cf']).item()
        
        #if both are cf graphs
        if((pred_orig1 != pred_cf1) and (pred_orig2 != pred_cf2)):
            #for each graph compute jaccard of cf edge_index
            jaccard.append(jaccard_cf_graph(explainer, data_dict1, data_dict2))
        
    #return avg jaccard, std jaccard
    return np.mean(jaccard), np.std(jaccard)

#compute num connected components
def is_connected_check(exp, data_dict, explainer, graph_type='orig'):
    #convert to networkx
    if(graph_type=='orig'):
        nx_graph = to_networkx(exp, node_attrs=["x"], edge_attrs=["edge_weight"], to_undirected=True)
        return nx.is_connected(nx_graph)
    else:
        #nx.from_numpy_matrix
        if(graph_type == 'cf'):
            _, cf_adj = get_adj_mat(data_dict, explainer)
            nx_graph = nx.from_numpy_matrix(cf_adj)
            return nx.is_connected(nx_graph)

#check connectivity
def feasibility(explainer, explanations, indices):
    #for test set
    connected_orig_total = 0
    connected_cf = 0
    flipped = 0
    for i in indices:
        data_dict = explanations[i]
        pred_orig = torch.argmax(data_dict['pred']).item()
        pred_cf = torch.argmax(data_dict['pred_cf']).item()

        #for counterfactual graph check feasibility
        connected_orig_total += is_connected_check(data_dict['graph'], data_dict,  explainer)
        if(pred_orig != pred_cf):
            # connected_orig += is_connected_check(data_dict['graph'], data_dict, explainer)
            if(explainer == 'rcexplainer_0.0'):
                connected_cf += is_connected_check(data_dict['graph_cf_up'], data_dict, explainer, 'cf')
            else:
                connected_cf += is_connected_check(data_dict['graph_cf'], data_dict, explainer, 'cf')
            flipped+=1


    N = flipped #C + C'
    M = len(indices) #O+O'
    O = connected_orig_total
    O_prime = M - O
    C = connected_cf
    C_prime = N - C
    C_E = (O/M)*N
    C_E_prime = (O_prime/M)*N
    if(C_E!=0 and C_E_prime!=0): #C_E might be 0
        chi_squared =  ((C_E - C)**2)/C_E + ((C_E_prime - C_prime)**2)/C_E_prime
    elif(C_E_prime!=0):
        #print(f"C_E:{C_E} is 0")
        chi_squared =  ((C_E_prime - C_prime)**2)/C_E_prime
    elif(C_E != 0):
        #print(f"C_E_prime:{C_E_prime} is 0")
        chi_squared =  ((C_E - C)**2)/C_E
    else:
        chi_squared = 0
    return C_E, C, chi_squared
