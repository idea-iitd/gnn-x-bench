# This file generates explanation quality based on metrics.

import torch
import argparse
from gnn_trainer import GNNTrainer
import data_utils
import metrics
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Mutagenicity',
                        choices=['Mutagenicity', 'Proteins', 'Mutag', 'IMDB-B', 'AIDS', 'NCI1', 'Tree-of-Life', 'Graph-SST2', 'DD', 'REDDIT-B'],
                        help="Dataset name")
    parser.add_argument('--gnn_type', type=str, default='gcn', choices=['gcn', 'gat', 'gin', 'sage'], help='GNN layer type to use.')
    parser.add_argument('--device', type=str, default="0", help='Index of cuda device to use. Default is 0.')
    parser.add_argument('--gnn_run', type=int, default=1, help='random seed for gnn run')
    parser.add_argument('--explainer_run', type=int, default=1, help='random seed for explainer run')
    parser.add_argument('--explainer_name', type=str, choices=['pgexplainer', 'tagexplainer', 'tagexplainer_1', 'tagexplainer_2',
                                                               'cff_1.0', 'rcexplainer_1.0', 'gnnexplainer', 'gem', 'subgraphx'],
                        help='Name of explainer to use.')
    parser.add_argument('--explanation_metric', type=str, choices=['faithfulness', 'faithfulness_with_removal', 'faithfulness_on_test', 'stability_noise', 'stability_seed',
                                                                   'stability_base', 'stability_noise_feature', 'stability_topology_adversarial'],
                        help='Explanation metric to use.')
    parser.add_argument('--folded', action='store_true', help='Whether to use folded results.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    trainer = GNNTrainer(dataset_name=args.dataset, gnn_type=args.gnn_type, task='basegnn', device=args.device)
    model = trainer.load(args.gnn_run)
    model.eval()

    result_folder = f'data/{args.dataset}/{args.explainer_name}_fold/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    dataset = data_utils.load_dataset(args.dataset)
    splits, indices = data_utils.split_data(dataset)
    test_indices = indices[2]

    if args.explainer_name == 'subgraphx':
        # we only have explanations for test set
        explanations = data_utils.load_explanations_test(args.dataset, args.explainer_name, args.gnn_type, torch.device('cpu'), args.explainer_run)
        dataset = dataset[test_indices]  # test only
    else:
        explanations = data_utils.load_explanations(args.dataset, args.explainer_name, args.gnn_type, torch.device('cpu'), args.explainer_run)

    if args.explanation_metric == 'faithfulness_on_test':  # for inductive methods
        dataset = dataset[test_indices]
        explanations = [explanations[idx] for idx in test_indices]

    print(f'Started: {args.dataset}, {args.explainer_name}, {args.gnn_type}, {args.explanation_metric}')

    # split dataset and explanations into 5
    dataset_splits, dataset_splits_indices = data_utils.split_data_equally(dataset, num_splits=5)

    if args.folded:
        fold_number = 5
    else:
        fold_number = 1

    for fold in range(fold_number):
        if args.folded:
            print('Fold', fold)
            indices = dataset_splits_indices[fold]
            explanations_fold = [explanations[idx] for idx in indices]
            dataset_fold = [dataset[idx] for idx in indices]
        else:
            explanations_fold = explanations
            dataset_fold = dataset

        # Generate explanation quality based on metrics
        if args.explanation_metric == 'faithfulness':
            ks = [5, 10, 15, 20, 25]
            metric_names = ['sufficiency']
            faithfulness_scores_dict = {metric: [] for metric in metric_names}
            for k in ks:
                faithfulness_scores = metrics.faithfulness(model, dataset_fold, explanations_fold, k, metric_names, device=device)
                for i in range(len(metric_names)):
                    faithfulness_scores_dict[metric_names[i]].append(faithfulness_scores[i])
            if args.folded:
                torch.save(faithfulness_scores_dict, result_folder + f'faithfulness_{args.gnn_type}_run_{args.explainer_run}_fold_{fold}.pt')
            else:
                torch.save(faithfulness_scores_dict, result_folder + f'faithfulness_{args.gnn_type}_run_{args.explainer_run}.pt')
        elif args.explanation_metric == 'faithfulness_on_test':  # only for inductive methods
            ks = [5, 10, 15, 20, 25]
            metric_names = ['sufficiency']
            faithfulness_scores_dict = {metric: [] for metric in metric_names}
            for k in ks:
                faithfulness_scores = metrics.faithfulness(model, dataset_fold, explanations_fold, k, metric_names, device=device)
                for i in range(len(metric_names)):
                    faithfulness_scores_dict[metric_names[i]].append(faithfulness_scores[i])
            if args.folded:
                torch.save(faithfulness_scores_dict, result_folder + f'faithfulness_{args.gnn_type}_run_{args.explainer_run}_test_only_fold_{fold}.pt')
            else:
                torch.save(faithfulness_scores_dict, result_folder + f'faithfulness_{args.gnn_type}_run_{args.explainer_run}_test_only.pt')
        elif args.explanation_metric == 'faithfulness_with_removal':
            ks = [2, 4, 6, 8, 10]
            metric_names = ['necessity']
            faithfulness_scores_dict = {metric: [] for metric in metric_names}
            for k in ks:
                faithfulness_scores = metrics.faithfulness_with_removal(model, dataset_fold, explanations_fold, k, metric_names, device=device)
                for i in range(len(metric_names)):
                    faithfulness_scores_dict[metric_names[i]].append(faithfulness_scores[i])
            if args.folded:
                torch.save(faithfulness_scores_dict, result_folder + f'faithfulness_with_removal_{args.gnn_type}_run_{args.explainer_run}_fold_{fold}.pt')
            else:
                torch.save(faithfulness_scores_dict, result_folder + f'faithfulness_with_removal_{args.gnn_type}_run_{args.explainer_run}.pt')
        elif args.explanation_metric == 'stability_noise':
            ks = [1, 2, 3, 4, 5]
            metric_names = ['jaccard']
            robustness_scores_dict = {metric: [] for metric in metric_names}
            top_k = 10
            for k in ks:
                explanations_noise = data_utils.load_explanations_noisy(args.dataset, args.explainer_name, args.gnn_type, device, args.explainer_run, k=k)
                explanations_noise_fold = [explanations_noise[idx] for idx in indices]
                robustness_scores = metrics.robustness(explanations_fold, explanations_noise_fold, top_k, metric_names)
                for i in range(len(metric_names)):
                    robustness_scores_dict[metric_names[i]].append(robustness_scores[i])
            if args.folded:
                torch.save(robustness_scores_dict, result_folder + f'stability_noise_{args.gnn_type}_run_{args.explainer_run}_fold_{fold}.pt')
            else:
                torch.save(robustness_scores_dict, result_folder + f'stability_noise_{args.gnn_type}_run_{args.explainer_run}.pt')
        elif args.explanation_metric == 'stability_noise_feature':
            ks = [10, 20, 30, 40, 50]
            metric_names = ['jaccard']
            robustness_scores_dict = {metric: [] for metric in metric_names}
            top_k = 10
            for k in ks:
                explanations_noise = data_utils.load_explanations_noisy_feature(args.dataset, args.explainer_name, args.gnn_type, device, args.explainer_run, k=k)
                explanations_noise_fold = [explanations_noise[idx] for idx in indices]
                robustness_scores = metrics.robustness(explanations_fold, explanations_noise_fold, top_k, metric_names)
                for i in range(len(metric_names)):
                    robustness_scores_dict[metric_names[i]].append(robustness_scores[i])
            if args.folded:
                torch.save(robustness_scores_dict, result_folder + f'stability_noise_feature_{args.gnn_type}_run_{args.explainer_run}_fold_{fold}.pt')
            else:
                torch.save(robustness_scores_dict, result_folder + f'stability_noise_feature_{args.gnn_type}_run_{args.explainer_run}.pt')
        elif args.explanation_metric == 'stability_topology_adversarial':
            ks = [1, 2, 3, 4, 5]
            metric_names = ['jaccard']
            robustness_scores_dict = {metric: [] for metric in metric_names}
            top_k = 10
            for k in ks:
                explanations_noise = data_utils.load_explanations_topology_adversarial(args.dataset, args.explainer_name, args.gnn_type, device, args.explainer_run, k=k)
                explanations_noise_fold = [explanations_noise[idx] for idx in indices]
                robustness_scores = metrics.robustness(explanations_fold, explanations_noise_fold, top_k, metric_names)
                for i in range(len(metric_names)):
                    robustness_scores_dict[metric_names[i]].append(robustness_scores[i])
            if args.folded:
                torch.save(robustness_scores_dict, result_folder + f'stability_topology_adversarial_{args.gnn_type}_run_{args.explainer_run}_fold_{fold}.pt')
            else:
                torch.save(robustness_scores_dict, result_folder + f'stability_topology_adversarial_{args.gnn_type}_run_{args.explainer_run}.pt')
        elif args.explanation_metric == 'stability_seed':
            seeds = [1, 2, 3]
            metric_names = ['jaccard']
            stability_seed_scores_dict = {metric: [] for metric in metric_names}
            top_k = 10
            for i in range(len(seeds)):
                for j in range(i + 1, len(seeds)):
                    explanations_seed_i = data_utils.load_explanations(args.dataset, args.explainer_name, args.gnn_type, torch.device('cpu'), run=seeds[i])
                    explanations_seed_i_fold = [explanations_seed_i[idx] for idx in indices]
                    explanations_seed_j = data_utils.load_explanations(args.dataset, args.explainer_name, args.gnn_type, torch.device('cpu'), run=seeds[j])
                    explanations_seed_j_fold = [explanations_seed_j[idx] for idx in indices]
                    stability_seed_scores = metrics.robustness(explanations_seed_i_fold, explanations_seed_j_fold, top_k, metric_names)
                    for l in range(len(metric_names)):
                        stability_seed_scores_dict[metric_names[l]].append((seeds[i], seeds[j], stability_seed_scores[l]))
            if args.folded:
                torch.save(stability_seed_scores_dict, result_folder + f'stability_seed_{args.gnn_type}_run_{args.explainer_run}_fold_{fold}.pt')
            else:
                torch.save(stability_seed_scores_dict, result_folder + f'stability_seed_{args.gnn_type}_run_{args.explainer_run}.pt')
        elif args.explanation_metric == 'stability_base':
            bases = ['gcn', 'gat', 'gin', 'sage']
            metric_names = ['jaccard']
            stability_base_scores_dict = {metric: [] for metric in metric_names}
            top_k = 10
            for i in range(len(bases)):
                for j in range(i + 1, len(bases)):
                    explanations_base1 = data_utils.load_explanations(args.dataset, args.explainer_name, bases[i], torch.device('cpu'), run=1)
                    explanations_base1_fold = [explanations_base1[idx] for idx in indices]
                    explanations_base2 = data_utils.load_explanations(args.dataset, args.explainer_name, bases[j], torch.device('cpu'), run=1)
                    explanations_base2_fold = [explanations_base2[idx] for idx in indices]
                    stability_base_scores = metrics.robustness(explanations_base1_fold, explanations_base2_fold, top_k, metric_names)
                    for l in range(len(metric_names)):
                        stability_base_scores_dict[metric_names[l]].append((bases[i], bases[j], stability_base_scores[l]))
            if args.folded:
                torch.save(stability_base_scores_dict, result_folder + f'stability_base_run_{args.explainer_run}_{fold}.pt')
            else:
                torch.save(stability_base_scores_dict, result_folder + f'stability_base_run_{args.explainer_run}.pt')
        else:
            raise NotImplementedError

        print(f'Finished: {args.dataset}, {args.explainer_name}, {args.gnn_type}, {args.explanation_metric}')
