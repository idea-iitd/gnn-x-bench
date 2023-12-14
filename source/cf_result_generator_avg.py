# This file generates explanation quality based on metrics.

import torch
import argparse
from gnn_trainer import GNNTrainer
import data_utils
import cf_metrics as metrics
import numpy as np
import os
import pickle
import copy
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Mutagenicity',
                        choices=['Mutagenicity', 'Proteins', 'Mutag', 'IMDB-B', 'AIDS'],
                        help="Dataset name")
    parser.add_argument('--gnn_type', type=str, default='gcn', choices=['gcn', 'gat', 'gin', 'sage'], help='GNN layer type to use.')
    parser.add_argument('--device', type=int, default=0, help='Index of cuda device to use. Default is 0.')
    parser.add_argument('--gnn_run', type=int, default=1, help='random seed for gnn run')
    parser.add_argument('--explainer_run', type=int, default=1, help='random seed for explainer run')
    parser.add_argument('--explainer_name', type=str, choices=['cff_0.0','rcexplainer_0.0', 'clear'], help='Name of explainer to use.')
    parser.add_argument('--explanation_metric','--list', nargs='+', help='<Required> Set flag', required=True, type=str, choices=['sufficiency', 'size', 'sparsity', 'stability_noise', 'stability_feature_noise', 'stability_adversarial_noise', 'stability_seed', 'stability_base', 'feasibility'])
    parser.add_argument('--verbose', type=int, default=1, help='Default: 1 (print computed metric), else 0')
    return parser.parse_args()

def load_exps_res_dict(args, metric, variants, model, device):
    exps_res_dict = {}
    for var in variants:
        if(metric == 'stability_feature_noise'):
            res_feat_noise = data_utils.load_explanations_feature_noisy(args.dataset, args.explainer_name, args.gnn_type, device, args.explainer_run, k=var)
            if(args.explainer_name == 'rcexplainer_0.0'):
                exps_res_dict[var] = metrics.remove_top_k_incremental(res_feat_noise, model, device)
            else:
                exps_res_dict[var] = res_feat_noise
        elif(metric == 'stability_noise'):
            res_noise = data_utils.load_explanations_noisy(args.dataset, args.explainer_name, args.gnn_type, device, args.explainer_run, k=var)
            if(args.explainer_name == 'rcexplainer_0.0'):
                exps_res_dict[var] = metrics.remove_top_k_incremental(res_noise, model, device)
            else:
                exps_res_dict[var] = res_noise
        elif(metric == 'stability_adversarial_noise'):
            res_adv_noise = data_utils.load_explanations_adversarial_noisy(args.dataset, args.explainer_name, args.gnn_type, device, args.explainer_run, k=var)
            if(args.explainer_name == 'rcexplainer_0.0'):
                exps_res_dict[var] = metrics.remove_top_k_incremental(res_adv_noise, model, device)
            else:
                exps_res_dict[var] = res_adv_noise
        elif(metric == 'stability_base'):
            res_base = data_utils.load_explanations(args.dataset, args.explainer_name, var, torch.device('cpu'), run=1)
            if(args.explainer_name == 'rcexplainer_0.0'):
                exps_res_dict[var] = metrics.remove_top_k_incremental(res_base, model, device)
            else:
                exps_res_dict[var] = res_base
        elif(metric == 'stability_seed'):  
            res_seed = data_utils.load_explanations(args.dataset, args.explainer_name, args.gnn_type, torch.device('cpu'), run=var)  
            if(args.explainer_name == 'rcexplainer_0.0'):
                exps_res_dict[var] = metrics.remove_top_k_incremental(res_seed, model, device)
            else:
                exps_res_dict[var] = res_seed
    return exps_res_dict

if __name__ == '__main__':
    args = parse_args()
    
    trainer = GNNTrainer(dataset_name=args.dataset, gnn_type=args.gnn_type, task='basegnn', device=args.device)
    model = trainer.load(args.gnn_run)
    model.eval()

    result_folder = f'data/{args.dataset}/{args.explainer_name}/result/'
    isExist = os.path.exists(result_folder)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(result_folder)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    dataset = data_utils.load_dataset(args.dataset)
    splits, indices = data_utils.split_data(dataset)
    
    if(os.path.isfile(f'data/{args.dataset}/test_subsets.pkl')):
        f = open(f'data/{args.dataset}/test_subsets.pkl', 'rb')
        test_samples = pickle.load(f)
        f.close()
    else:
        test_samples = data_utils.sample_subsets(indices[2], args.dataset, num_samples=5)
    
    test_indices = indices[2]
    #cff and clear have only test_idx. mapping test ids to integer range. ensured order is maintained
    if(args.explainer_name == 'cff_0.0' or args.explainer_name == 'clear'):
        test_indices_map = {}
        for i in range(len(indices[2])):
            test_indices_map[indices[2][i]] = i
        
        test_samples_mapped = []
        for sample in test_samples:
            test_samples_mapped.append([test_indices_map[val] for val in sample])
            
        test_samples = test_samples_mapped
        test_indices = np.arange(len(indices[2]))
        
    explanations = data_utils.load_explanations(args.dataset, args.explainer_name, args.gnn_type, torch.device('cpu'), args.explainer_run)
   
    #generate score
    if(args.explainer_name == 'rcexplainer_0.0'):
        explanations_updated = metrics.remove_top_k_incremental(explanations, model, device)
        explanations = explanations_updated
        
    print(f'Started: {args.dataset}, {args.explainer_name}, {args.gnn_type}, {args.explanation_metric}')

    #flag to track non-implemeted metric
    flag = False
    # Generate explanation quality based on metrics
    if 'sufficiency' in args.explanation_metric:
        flag = True
        sufficiency_scores_dict = {'sufficiency':[]}
        for sample in test_samples: 
            sufficiency = metrics.sufficiency(args.explainer_name, explanations, sample)
            sufficiency_scores_dict['sufficiency'].append(sufficiency)
        
        sufficiency_scores_dict['mean'] = metrics.mean_samples(sufficiency_scores_dict['sufficiency'])
        sufficiency_scores_dict['std'] = metrics.stdev_samples(sufficiency_scores_dict['sufficiency'])
        if(args.verbose):
            print('Sufficiency: ', sufficiency_scores_dict['mean'], '+-', sufficiency_scores_dict['std'])
            
        torch.save(sufficiency_scores_dict, result_folder + f'cf_sufficiency_{args.gnn_type}_run_{args.explainer_run}.pt')
        
    if 'size' in args.explanation_metric:
        flag = True
        size_scores_dict = {'size': []}
        for sample in test_samples: 
            avg_size, stdev_size = metrics.size(args.explainer_name, explanations, sample)
            size_scores_dict['size'].append([avg_size, stdev_size])
        
        size_scores_dict['mean'] =  metrics.mean_samples([size[0] for size in size_scores_dict['size']])
        size_scores_dict['std'] = metrics.mean_stdev([size[1] for size in size_scores_dict['size']])
        if(args.verbose):
            print('Size: ', size_scores_dict['mean'] , '+-', size_scores_dict['std'])
            
        torch.save(size_scores_dict, result_folder + f'cf_size_{args.gnn_type}_run_{args.explainer_run}.pt')
        
    
    if 'sparsity' in args.explanation_metric:
        flag = True
        sparsity_scores_dict = {'sparsity':[]}
        for sample in test_samples: 
            sparsity = metrics.sparsity(args.explainer_name, explanations, sample)
            sparsity_scores_dict['sparsity'].append(sparsity)
        
        sparsity_scores_dict['mean'] = metrics.mean_samples(sparsity_scores_dict['sparsity'])
        sparsity_scores_dict['std'] = metrics.stdev_samples(sparsity_scores_dict['sparsity'])
        
        if(args.verbose):
            print('Sparsity: ', sparsity_scores_dict['mean'], '+-', sparsity_scores_dict['std'])
        
        torch.save(sparsity_scores_dict, result_folder + f'cf_sparsity_{args.gnn_type}_run_{args.explainer_run}.pt')
        
    if 'stability_noise' in args.explanation_metric:
        flag = True
        ks = [1, 2, 3, 4, 5]
        metric_names = ['jaccard', 'size', 'sufficiency']
        robustness_scores_dict = {k:{metric: [] for metric in metric_names} for k in ks}
        explanations_noise_dict = load_exps_res_dict(args, 'stability_noise', ks, model, device)
        
        for k in ks:
            explanations_noise = explanations_noise_dict[k]
            
            for sample in test_samples:  
                #generate score
                avg_jaccard, stdev_jaccard = metrics.robustness(args.explainer_name, explanations, explanations_noise, sample)
                sufficiency = metrics.sufficiency(args.explainer_name, explanations_noise, sample)
                avg_size, stdev_size = metrics.size(args.explainer_name, explanations_noise, sample)
            
                robustness_scores_dict[k]['jaccard'].append([avg_jaccard, stdev_jaccard])
                robustness_scores_dict[k]['size'].append([avg_size, stdev_size])
                robustness_scores_dict[k]['sufficiency'].append(sufficiency)
                
            for metric in metric_names:
                if(metric == 'sufficiency' or metric == 'sparsity'):
                    robustness_scores_dict[k][f'mean_{metric}'] = metrics.mean_samples(robustness_scores_dict[k][metric])
                    robustness_scores_dict[k][f'std_{metric}'] = metrics.stdev_samples(robustness_scores_dict[k][metric])
                else:
                    robustness_scores_dict[k][f'mean_{metric}'] = metrics.mean_samples([val[0] for val in robustness_scores_dict[k][metric]])
                    robustness_scores_dict[k][f'std_{metric}'] = metrics.mean_stdev([val[1] for val in robustness_scores_dict[k][metric]])
            
            if(args.verbose):
                print(f'----------- noise budget:{k} ---------------')
                for metric in metric_names:
                    print(f'{metric}: ', robustness_scores_dict[k][f'mean_{metric}'], '+-', robustness_scores_dict[k][f'std_{metric}'])
            
        torch.save(robustness_scores_dict, result_folder + f'cf_stability_noise_{args.gnn_type}_run_{args.explainer_run}.pt')
        
    if 'stability_adversarial_noise' in args.explanation_metric:
        flag = True
        ks = [1, 2, 3, 4, 5]
        metric_names = ['jaccard', 'size', 'sufficiency']
        robustness_adv_scores_dict = {k:{metric: [] for metric in metric_names} for k in ks}
        assert(args.dataset in ['Proteins'] and args.explainer_name in ['cff_0.0', 'rcexplainer_0.0']) #clear gave OOM error
             
        explanations_adv_noise_dict = load_exps_res_dict(args, 'stability_adversarial_noise', ks, model, device)
        
        for k in ks:
            explanations_adv_noise = explanations_adv_noise_dict[k]
            
            for sample in test_samples:  
                #generate score
                avg_jaccard, stdev_jaccard = metrics.robustness(args.explainer_name, explanations, explanations_adv_noise, sample)
                sufficiency = metrics.sufficiency(args.explainer_name, explanations_adv_noise, sample)
                avg_size, stdev_size = metrics.size(args.explainer_name, explanations_adv_noise, sample)
            
                robustness_adv_scores_dict[k]['jaccard'].append([avg_jaccard, stdev_jaccard])
                robustness_adv_scores_dict[k]['size'].append([avg_size, stdev_size])
                robustness_adv_scores_dict[k]['sufficiency'].append(sufficiency)
                
            for metric in metric_names:
                if(metric == 'sufficiency' or metric == 'sparsity'):
                    robustness_adv_scores_dict[k][f'mean_{metric}'] = metrics.mean_samples(robustness_adv_scores_dict[k][metric])
                    robustness_adv_scores_dict[k][f'std_{metric}'] = metrics.stdev_samples(robustness_adv_scores_dict[k][metric])
                else:
                    robustness_adv_scores_dict[k][f'mean_{metric}'] = metrics.mean_samples([val[0] for val in robustness_adv_scores_dict[k][metric]])
                    robustness_adv_scores_dict[k][f'std_{metric}'] = metrics.mean_stdev([val[1] for val in robustness_adv_scores_dict[k][metric]])
            
            if(args.verbose):
                print(f'----------- noise budget:{k} ---------------')
                for metric in metric_names:
                    print(f'{metric}: ', robustness_adv_scores_dict[k][f'mean_{metric}'], '+-', robustness_adv_scores_dict[k][f'std_{metric}'])
            
        torch.save(robustness_adv_scores_dict, result_folder + f'cf_stability_adversarial_noise_{args.gnn_type}_run_{args.explainer_run}.pt')
            
    if 'stability_feature_noise' in args.explanation_metric:
        flag = True
        ks = [10, 20, 30, 40, 50]
        metric_names = ['jaccard', 'size', 'sufficiency']
        robustness_feat_scores_dict = {k:{metric: [] for metric in metric_names} for k in ks}
        
        assert(args.dataset in ['Proteins', 'Mutag', 'Mutagenicity']) 
        if(args.dataset in ['Proteins', 'Mutagenicity']):
             assert(args.explainer_name in ['cff_0.0', 'rcexplainer_0.0'])#clear gave OOM error
        explanations_feat_noise_dict = load_exps_res_dict(args, 'stability_feature_noise', ks, model, device)
        
        for k in ks:
            explanations_feat_noise = explanations_feat_noise_dict[k]
            
            for sample in test_samples:  
                #generate score
                avg_jaccard, stdev_jaccard = metrics.robustness(args.explainer_name, explanations, explanations_feat_noise, sample)
                sufficiency = metrics.sufficiency(args.explainer_name, explanations_feat_noise, sample)
                avg_size, stdev_size = metrics.size(args.explainer_name, explanations_feat_noise, sample)
            
                robustness_feat_scores_dict[k]['jaccard'].append([avg_jaccard, stdev_jaccard])
                robustness_feat_scores_dict[k]['size'].append([avg_size, stdev_size])
                robustness_feat_scores_dict[k]['sufficiency'].append(sufficiency)
                
            for metric in metric_names:
                if(metric == 'sufficiency' or metric == 'sparsity'):
                    robustness_feat_scores_dict[k][f'mean_{metric}'] = metrics.mean_samples(robustness_feat_scores_dict[k][metric])
                    robustness_feat_scores_dict[k][f'std_{metric}'] = metrics.stdev_samples(robustness_feat_scores_dict[k][metric])
                else:
                    robustness_feat_scores_dict[k][f'mean_{metric}'] = metrics.mean_samples([val[0] for val in robustness_feat_scores_dict[k][metric]])
                    robustness_feat_scores_dict[k][f'std_{metric}'] = metrics.mean_stdev([val[1] for val in robustness_feat_scores_dict[k][metric]])
            
            if(args.verbose):
                print(f'----------- noise budget:{k} ---------------')
                for metric in metric_names:
                    print(f'{metric}: ', robustness_feat_scores_dict[k][f'mean_{metric}'], '+-', robustness_feat_scores_dict[k][f'std_{metric}'])
            
        torch.save(robustness_feat_scores_dict, result_folder + f'cf_stability_feature_noise_{args.gnn_type}_run_{args.explainer_run}.pt')
            
    if 'stability_seed' in args.explanation_metric:
        flag = True
        seeds = [1, 2, 3]
        metric_names = ['jaccard', 'size', 'sufficiency']
        stability_seed_scores_dict = {i: {} for i in range(len(seeds))}
        
        for i in range(len(seeds)):
            for metric in metric_names:
                if(metric != 'jaccard'):
                    stability_seed_scores_dict[i][metric] = []                   
                else:
                    stability_seed_scores_dict[i][metric] = {}
                    for j in range(i + 1, len(seeds)):
                         stability_seed_scores_dict[i][metric][f'{i}_{j}'] = [] 
        
        explanations_seed_dict = load_exps_res_dict(args, 'stability_seed', seeds, model, device)
                    
        for i in range(len(seeds)):
            explanations_seed_i = explanations_seed_dict[seeds[i]]

            for k, sample in enumerate(test_samples): 
                sufficiency = metrics.sufficiency(args.explainer_name, explanations_seed_i, sample)
                avg_size, stdev_size = metrics.size(args.explainer_name, explanations_seed_i, sample)
                
                stability_seed_scores_dict[i]['size'].append([avg_size, stdev_size])
                stability_seed_scores_dict[i]['sufficiency'].append(sufficiency)

                for j in range(i + 1, len(seeds)):
                    explanations_seed_j = explanations_seed_dict[seeds[j]]
                    avg_jaccard, stdev_jaccard = metrics.robustness(args.explainer_name, explanations_seed_i, explanations_seed_j, sample)
                    stability_seed_scores_dict[i]['jaccard'][f'{i}_{j}'].append([avg_jaccard, stdev_jaccard])
            
            #compute avg metric for seed[i]
            for metric in metric_names:
                if(metric == 'sufficiency' or metric == 'sparsity'):
                    stability_seed_scores_dict[i][f'mean_{metric}'] = metrics.mean_samples(stability_seed_scores_dict[i][metric])
                    stability_seed_scores_dict[i][f'std_{metric}'] = metrics.stdev_samples(stability_seed_scores_dict[i][metric])
                elif(metric == 'size'):
                    stability_seed_scores_dict[i][f'mean_{metric}'] = metrics.mean_samples([val[0] for val in stability_seed_scores_dict[i][metric]])
                    stability_seed_scores_dict[i][f'std_{metric}'] = metrics.mean_stdev([val[1] for val in stability_seed_scores_dict[i][metric]])
                elif(metric == 'jaccard'):
                    dict_cpy = copy.deepcopy(stability_seed_scores_dict[i][metric])
                    for key in dict_cpy.keys():
                        stability_seed_scores_dict[i][metric][f'mean_{metric}_{key}'] = metrics.mean_samples([val[0] for val in dict_cpy[key]])
                        stability_seed_scores_dict[i][metric][f'std_{metric}_{key}'] = metrics.mean_stdev([val[1] for val in dict_cpy[key]])
            
            if(args.verbose):
                print(f'################ seed {i} ###############')
                for metric in metric_names:
                    if(metric != 'jaccard'):
                        print(f'{metric}: ', stability_seed_scores_dict[i][f'mean_{metric}'], '+-', stability_seed_scores_dict[i][f'std_{metric}'])
                    else:
                        for key in dict_cpy.keys():
                            print(f'{metric}-{key}: ', stability_seed_scores_dict[i][metric][f'mean_{metric}_{key}'], '+-', stability_seed_scores_dict[i][metric][f'std_{metric}_{key}'])
                            
        torch.save(stability_seed_scores_dict, result_folder + f'cf_stability_seed_{args.gnn_type}_run_{args.explainer_run}.pt')
        
    if 'stability_base' in args.explanation_metric:
        flag = True
        bases = ['gcn', 'gat', 'gin', 'sage']
        metric_names = ['jaccard', 'size', 'sufficiency']
        stability_base_scores_dict = {base: {} for base in bases}
        for i in range(len(bases)):
            for metric in metric_names:
                if(metric != 'jaccard'):
                    stability_base_scores_dict[bases[i]][metric] = []                   
                else:
                    stability_base_scores_dict[bases[i]][metric] = {}
                    for j in range(i + 1, len(bases)):
                         stability_base_scores_dict[bases[i]][metric][bases[i]+'_'+ bases[j]] = [] 
        
        explanations_base_dict = load_exps_res_dict(args, 'stability_base', bases, model, device)
                         
        for i in range(len(bases)):
            explanations_base1 = explanations_base_dict[bases[i]]
            for k, sample in enumerate(test_samples): 
                sufficiency = metrics.sufficiency(args.explainer_name, explanations_base1, sample)
                avg_size, stdev_size = metrics.size(args.explainer_name, explanations_base1, sample)
                
                stability_base_scores_dict[bases[i]]['size'].append([avg_size, stdev_size])
                stability_base_scores_dict[bases[i]]['sufficiency'].append(sufficiency)
             
                for j in range(i + 1, len(bases)):
                    explanations_base2 = explanations_base_dict[bases[j]]
                    avg_jaccard, stdev_jaccard = metrics.robustness(args.explainer_name, explanations_base1, explanations_base2, sample)
                    stability_base_scores_dict[bases[i]]['jaccard'][bases[i]+'_'+ bases[j]].append([avg_jaccard, stdev_jaccard])

            #compute avg metric for seed[i]
            for metric in metric_names:
                if(metric == 'sufficiency' or metric == 'sparsity'):
                    stability_base_scores_dict[bases[i]][f'mean_{metric}'] = metrics.mean_samples(stability_base_scores_dict[bases[i]][metric])
                    stability_base_scores_dict[bases[i]][f'std_{metric}'] = metrics.stdev_samples(stability_base_scores_dict[bases[i]][metric])
                elif(metric == 'size'):
                    stability_base_scores_dict[bases[i]][f'mean_{metric}'] = metrics.mean_samples([val[0] for val in stability_base_scores_dict[bases[i]][metric]])
                    stability_base_scores_dict[bases[i]][f'std_{metric}'] = metrics.mean_stdev([val[1] for val in stability_base_scores_dict[bases[i]][metric]])
                elif(metric == 'jaccard'):
                    dict_cpy = copy.deepcopy(stability_base_scores_dict[bases[i]][metric])
                    for key in dict_cpy.keys():
                        stability_base_scores_dict[bases[i]][metric][f'mean_{metric}_{key}'] = metrics.mean_samples([val[0] for val in dict_cpy[key]])
                        stability_base_scores_dict[bases[i]][metric][f'std_{metric}_{key}'] = metrics.mean_stdev([val[1] for val in dict_cpy[key]])
            
            if(args.verbose):
                print(f'################ base: {bases[i]} ###############')
                for metric in metric_names:
                    if(metric != 'jaccard'):
                        print(f'{metric}: ', stability_base_scores_dict[bases[i]][f'mean_{metric}'], '+-', stability_base_scores_dict[bases[i]][f'std_{metric}'])
                    else:
                        for key in dict_cpy.keys():
                            print(f'{metric}-{key}: ', stability_base_scores_dict[bases[i]][metric][f'mean_{metric}_{key}'], '+-', stability_base_scores_dict[bases[i]][metric][f'std_{metric}_{key}'])
                            
                
        torch.save(stability_base_scores_dict, result_folder + f'cf_stability_base_run_{args.explainer_run}.pt')

    if 'feasibility' in args.explanation_metric:
        flag = True
        assert(args.dataset in ['Mutagenicity', 'Proteins', 'Mutag','AIDS'])
        feasibility_scores_dict = {}
        e_c, o_c, chi_sq = metrics.feasibility(args.explainer_name, explanations, test_indices)
        feasibility_scores_dict['feasibility'] = {'expected_count':e_c, 'observed_count':o_c, 'chi_squared':chi_sq}
        torch.save(feasibility_scores_dict, result_folder + f'cf_feasibility_{args.gnn_type}_run_{args.explainer_run}.pt')

        if(args.verbose):
            print(f'----------- Feasibility ---------------')    
            print(f'Expected_count: {e_c}')
            print(f'Observed_count: {o_c}')
            print(f'Chi_squared: {chi_sq}')

    if(flag == False):
        raise NotImplementedError

    print(f'Finished: {args.dataset}, {args.explainer_name}, {args.gnn_type}, {args.explanation_metric}')
