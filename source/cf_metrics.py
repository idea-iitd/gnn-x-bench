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
    # print(explanation)
    directed_edge_weight = explanation.edge_weight[explanation.edge_index[0] <= explanation.edge_index[1]]
    directed_edge_index = explanation.edge_index[:, explanation.edge_index[0] <= explanation.edge_index[1]]
    directed_edge_attr = explanation.edge_attr[explanation.edge_index[0] <= explanation.edge_index[1],:] if explanation.edge_attr is not None else None#shape: num_edges * attr_dim
    #remove top-k edges
    idx = torch.ones(directed_edge_weight.numel(), dtype=torch.bool)
    del_idx = directed_edge_weight.topk(min(k, directed_edge_weight.shape[0]), largest=False).indices #[0][-1]
    # idx[del_idx] = False
    # directed_edge_index = directed_edge_index[:, idx]
    edge_mask = torch.ones(directed_edge_weight.shape[0])
    edge_mask[del_idx] = 0
    directed_edge_weight[del_idx] = 0
    # directed_edge_attr = directed_edge_attr[idx, :] if directed_edge_attr is not None else None
    new_data = Data(
        edge_index=directed_edge_index.clone(),
        x=explanation.x.clone(),
        edge_weight=directed_edge_weight.clone(),
        edge_mask = edge_mask,
        edge_attr=directed_edge_attr.clone() if directed_edge_attr is not None else None,
        # target_node = explanation.target_node     
    )
    
    # make graph undirected
    new_data = ToUndirected()(new_data)
    # print(directed_edge_weight, new_data.edge_weight)
    
    # not removing isolated nodes to prevent change in graph max nodes
    # new_data = RemoveIsolatedNodes()(new_data)

    #return edge_index after removing top-k edges
    return new_data
    
def remove_top_k_incremental(explanations, model, device):
    # device = torch.device(f'cuda:{device}' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
    for i, exp in enumerate(explanations):
        pred_orig =  torch.argmax(model(exp['graph'].to(device), exp['graph'].edge_weight.to(device))[-1][0])
        pred_cf_orig = torch.argmax(model(exp['graph_cf'].to(device), exp['graph_cf'].edge_weight.to(device))[-1][0])
        pred_cf = pred_orig
        k = 1
        final_cf = exp['graph_cf']
        final_pred_cf = exp['pred_cf']
    
        # print('pred labels:', pred_cf_orig.item() , pred_cf.item())
        if(pred_cf_orig.item() != pred_cf.item()):    
            # print(f'---------------------- node: {i} --------------------')
            while((pred_cf == pred_orig) and k<=exp['graph'].edge_index.shape[1]):
                # print(f"k: {k}")
                cf_data = remove_top_k(model, exp['graph_cf'], k)
                pred_cf_logits = model(cf_data.to(device), cf_data.edge_weight.to(device))[-1][0]
                pred_cf = torch.argmax(pred_cf_logits)
                # print(k, pred_cf, pred_orig, pred_cf_logits, cf_data.edge_weight)
                if(pred_cf != pred_orig):
                    final_cf = cf_data
                    final_pred_cf = pred_cf_logits
                    break
                k+=1   
                  
        exp['graph_cf_up'] = final_cf
        exp['pred_cf_up'] = final_pred_cf
        
    return explanations
    
def get_adj_mat(data_dict, explainer):
    max_num_nodes = max(data_dict['graph'].num_nodes, data_dict['graph_cf'].num_nodes)
    orig_adj = to_dense_adj(data_dict['graph'].edge_index, max_num_nodes=max_num_nodes)[0].cpu().detach().numpy()
    if(explainer == 'rcexplainer_0.0'):
        try:
            edge_attr = data_dict['graph_cf_up'].edge_mask
        except:
            edge_attr = torch.ones(data_dict['graph_cf_up'].edge_weight.shape[0]).to('cuda:0') #hard coded
        #second attribute prevents rmeoving isolated nodes from adj hence change in shape of matrix
        # in case there is only one edge in graph, then removing is throws error. Bwlow condition handles this
        # if(data_dict['graph_cf_up'].edge_index.shape[1] == 0):
        cf_adj = to_dense_adj(
        data_dict['graph_cf_up'].edge_index,
        edge_attr = edge_attr, #data_dict['graph_cf_up'].edge_mask,
        max_num_nodes=max_num_nodes
        )[0].cpu().detach().numpy()
        # else:
        #     cf_adj = to_dense_adj(data_dict['graph_cf_up'].edge_index, 
        #                         max_num_nodes=max_num_nodes)[0].cpu().detach().numpy() 
    else:
        if(explainer == 'clear'):
            edge_attr_val  = None
        else:
            edge_attr_val = data_dict['graph_cf'].edge_weight
        cf_adj = to_dense_adj(
            data_dict['graph_cf'].edge_index,
            edge_attr = edge_attr_val, #data_dict['graph_cf'].edge_weight,
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
    
    # print(perbs_list)
    # print(perb_type)
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
        # print(data_dict)
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
        
            # print(f"exp-{i}: ", find_cf_edges(data_dict, explainer), data_dict)
       
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

#computes jaccard similarity between generated counterfactual explanations (removed/added edges)
def jaccard_cf():
    pass

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

# #sum incident edges
# def sum_incident_edges(data_dict, explainer):
#     data = data_dict['graph']
#     bonds = [bond + 1 for bond in np.argmax(data.edge_attr.cpu().detach().numpy(), axis=1).tolist()]
#     sum_orig = np.sum(bonds)
#     sum_cf = 0
    
#     if(explainer == 'rcexplainer_0.0'):
#         try:
#             for i in range(data_dict['graph_cf_up'].edge_mask.shape[0]):
#                 if(data_dict['graph_cf_up'].edge_mask[i].item() == 1):
#                     sum_cf += bonds[i]
#         except:
#             sum_cf = 0 #label not flipped so mask does not exist
#     elif(explainer == 'cff_0.0'):
#         for i in range(data_dict['graph_cf'].edge_weight.shape[0]):
#                 if(data_dict['graph_cf'].edge_weight[i]):
#                     sum_cf += bonds[i]
#     else:
#         sum_cf = 0 #will implement for clear 
    
#     return sum_orig/2, sum_cf/2    

# #compute valency feasibility
# def feasibility_valence(explainer, explanations, indices):          
#     #for test set
#     sum_valency_orig_total = 0
#     sum_valency_cf_total = 0
#     flipped = 0
#     for i in indices:
#         data_dict = explanations[i] 
#         pred_orig = torch.argmax(data_dict['pred']).item()
#         pred_cf = torch.argmax(data_dict['pred_cf']).item()
        
#         #for counterfactual graph check feasibility valency
#         sum_valency_orig, sum_valency_cf = sum_incident_edges(data_dict, explainer)
#         sum_valency_orig_total += sum_valency_orig
        
#         if(pred_orig != pred_cf):
#             sum_valency_cf_total += sum_valency_cf
#             print(f"Total valency: {sum_valency_orig}; CF valency : {sum_valency_cf}")
#             flipped+=1
        
    
#     r = sum_valency_orig_total/len(indices)
#     expected_count = r*flipped
#     observed_count = sum_valency_cf_total
#     chi_squared = ((expected_count- observed_count)**2)/expected_count
#     # exit(0)
#     return expected_count, observed_count, chi_squared



#compute num connected components
def is_connected_check(exp, data_dict, explainer, graph_type='orig'):
    #convert to networkx
    if(explainer != 'rcexplainer_0.0' or graph_type=='orig'):
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
    connected_orig = 0
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
            connected_orig += is_connected_check(data_dict['graph'], data_dict, explainer)
            if(explainer == 'rcexplainer_0.0'):
                connected_cf += is_connected_check(data_dict['graph_cf_up'], data_dict, explainer, 'cf')
            else:
                connected_cf += is_connected_check(data_dict['graph_cf'], data_dict, explainer, 'cf')
            flipped+=1
        
    
    r = connected_orig_total/len(indices)
    expected_count = r*flipped
    observed_count = connected_cf
    chi_squared = ((expected_count- observed_count)**2)/expected_count
    
    return expected_count, observed_count, chi_squared

def get_label(target_id, node_id, dataset):
    subgraph_labels = torch.load(f"/home/graphAttack/gnn-x-bench/{dataset}_subgraph_labels.pt")
    # print(target_id, len(subgraph_labels[target_id]))
    return subgraph_labels[target_id][node_id]

#only valid for datasets with ground truth
def accuracy(explainer, dataset, explanations, indices):
    undirected = is_undirected(explanations[0]['graph'].edge_index)
    relu = torch.nn.ReLU()
    accuracy = []
    #iterate over each index
    for i in range(len(indices)):
        data_dict = explanations[i]
        node_labels = data_dict['graph'].node_labels
        pred_orig = torch.argmax(data_dict['pred']).item()
        pred_cf = torch.argmax(data_dict['pred_cf']).item()      
        correct = 0   
        #check for instances method find counterfactuals
        if(pred_orig != pred_cf):
            #adj graph orig, adj graph_cf
            orig_adj, cf_adj = get_adj_mat(data_dict, explainer)
            if(sum(sum(abs(orig_adj - cf_adj))) == 0):
                continue
            #compute cf size
            if(undirected):
                cf_size = (sum(sum(abs(orig_adj - cf_adj)))/2)
            else:
                cf_size = (sum(sum(abs(orig_adj - cf_adj))))   

            #get perturbations
            perbs = (orig_adj - cf_adj)
            #get dels
            perbs_del = relu(torch.from_numpy(perbs)).numpy()
            #get adds 
            perbs_add = relu(torch.from_numpy(-1*perbs)).numpy() 
            # all adds are correct
            correct += sum(sum(perbs_add))
            # get all dels (u,v)s 
            perbs_del_uv = np.transpose(np.nonzero(perbs_del)).tolist()
            perbs_del_uv = set([(l[0], l[1]) for l in perbs_del_uv])   
            #for each deletion if label(u) != 0 and label(v)!=0, correct+=1 
            for pdel in perbs_del_uv:
                u = pdel[0]
                v = pdel[1]
                # print(node_labels[u].item(), node_labels[v].item())
                if node_labels[u] != 0 and node_labels[v] != 0:
                    correct += 1
            #avg accuracy = correct/size(cf)
            accuracy.append(correct / (2 * cf_size)) #double counting of edges in adds and dels

    return np.mean(accuracy)
    
    
#  mutagenicity = {0:	'C',
# 	1:	O
# 	2:	Cl
# 	3:	H
# 	4:	N
# 	5:	F
# 	6:	Br
# 	7:	S
# 	8:	P
# 	9:	I
# 	10:	Na
# 	11:	K
# 	12:	Li
# 	13:	Ca}   