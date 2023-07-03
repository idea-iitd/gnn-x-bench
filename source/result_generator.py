# This file generates explanation quality based on metrics.

import torch
import argparse
from gnn_trainer import GNNTrainer
import data_utils
import metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Mutagenicity',
                        choices=['Mutagenicity', 'Proteins', 'Mutag', 'IMDB-B', 'AIDS', 'NCI1', 'Graph-SST2', 'DD', 'REDDIT-B'],
                        help="Dataset name")
    parser.add_argument('--gnn_type', type=str, default='gcn', choices=['gcn', 'gat', 'gin', 'sage'], help='GNN layer type to use.')
    parser.add_argument('--device', type=str, default="0", help='Index of cuda device to use. Default is 0.')
    parser.add_argument('--gnn_run', type=int, default=1, help='random seed for gnn run')
    parser.add_argument('--explainer_run', type=int, default=1, help='random seed for explainer run')
    parser.add_argument('--explainer_name', type=str, choices=['pgexplainer', 'tagexplainer_1', 'tagexplainer_2', 'cff_1.0',
                                                               'rcexplainer_1.0', 'gnnexplainer', 'gem', 'subgraphx'],
                        help='Name of explainer to use.')
    parser.add_argument('--explanation_metric', type=str, choices=['faithfulness', 'faithfulness_with_removal', 'faithfulness_on_test', 'stability_noise', 'stability_seed', 'stability_base'],
                        help='Explanation metric to use.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    trainer = GNNTrainer(dataset_name=args.dataset, gnn_type=args.gnn_type, task='basegnn', device=args.device)
    model = trainer.load(args.gnn_run)
    model.eval()

    result_folder = f'data/{args.dataset}/{args.explainer_name}/'

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    dataset = data_utils.load_dataset(args.dataset)
    splits, indices = data_utils.split_data(dataset)
    if args.explainer_name == 'subgraphx':
        # we only have explanations for test set
        explanations = data_utils.load_explanations_test(args.dataset, args.explainer_name, args.gnn_type, torch.device('cpu'), args.explainer_run)
        dataset = dataset[indices[2]]  # test only
    else:
        explanations = data_utils.load_explanations(args.dataset, args.explainer_name, args.gnn_type, torch.device('cpu'), args.explainer_run)

    print(f'Started: {args.dataset}, {args.explainer_name}, {args.gnn_type}, {args.explanation_metric}')

    # Generate explanation quality based on metrics
    if args.explanation_metric == 'faithfulness':
        ks = [5, 10, 15, 20, 25]
        metric_names = ['sufficiency']
        faithfulness_scores_dict = {metric: [] for metric in metric_names}
        for k in ks:
            faithfulness_scores = metrics.faithfulness(model, dataset, explanations, k, metric_names, device=device)
            for i in range(len(metric_names)):
                faithfulness_scores_dict[metric_names[i]].append(faithfulness_scores[i])
        torch.save(faithfulness_scores_dict, result_folder + f'faithfulness_{args.gnn_type}_run_{args.explainer_run}.pt')
    elif args.explanation_metric == 'faithfulness_on_test':  # only for inductive methods
        ks = [5, 10, 15, 20, 25]
        metric_names = ['sufficiency']
        faithfulness_scores_dict = {metric: [] for metric in metric_names}
        test_indices = indices[2]
        test_dataset = dataset[test_indices]
        test_explanations = [explanations[idx] for idx in test_indices]
        for k in ks:
            faithfulness_scores = metrics.faithfulness(model, test_dataset, test_explanations, k, metric_names, device=device)
            for i in range(len(metric_names)):
                faithfulness_scores_dict[metric_names[i]].append(faithfulness_scores[i])
        torch.save(faithfulness_scores_dict, result_folder + f'faithfulness_{args.gnn_type}_run_{args.explainer_run}_test_only.pt')
    elif args.explanation_metric == 'faithfulness_with_removal':
        ks = [2, 4, 6, 8, 10]
        metric_names = ['necessity']
        faithfulness_scores_dict = {metric: [] for metric in metric_names}
        for k in ks:
            faithfulness_scores = metrics.faithfulness_with_removal(model, dataset, explanations, k, metric_names, device=device)
            for i in range(len(metric_names)):
                faithfulness_scores_dict[metric_names[i]].append(faithfulness_scores[i])
        torch.save(faithfulness_scores_dict, result_folder + f'faithfulness_with_removal_{args.gnn_type}_run_{args.explainer_run}.pt')
    elif args.explanation_metric == 'stability_noise':
        ks = [1, 2, 3, 4, 5]
        metric_names = ['jaccard']
        robustness_scores_dict = {metric: [] for metric in metric_names}
        top_k = 10
        for k in ks:
            explanations_noise = data_utils.load_explanations_noisy(args.dataset, args.explainer_name, args.gnn_type, device, args.explainer_run, k=k)
            robustness_scores = metrics.robustness(explanations, explanations_noise, top_k, metric_names)
            for i in range(len(metric_names)):
                robustness_scores_dict[metric_names[i]].append(robustness_scores[i])
        torch.save(robustness_scores_dict, result_folder + f'stability_noise_{args.gnn_type}_run_{args.explainer_run}.pt')
    elif args.explanation_metric == 'stability_seed':
        seeds = [1, 2, 3]
        metric_names = ['jaccard']
        stability_seed_scores_dict = {metric: [] for metric in metric_names}
        top_k = 10
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                explanations_seed_i = data_utils.load_explanations(args.dataset, args.explainer_name, args.gnn_type, torch.device('cpu'), run=seeds[i])
                explanations_seed_j = data_utils.load_explanations(args.dataset, args.explainer_name, args.gnn_type, torch.device('cpu'), run=seeds[j])
                stability_seed_scores = metrics.robustness(explanations_seed_i, explanations_seed_j, top_k, metric_names)
                for l in range(len(metric_names)):
                    stability_seed_scores_dict[metric_names[l]].append((seeds[i], seeds[j], stability_seed_scores[l]))
        torch.save(stability_seed_scores_dict, result_folder + f'stability_seed_{args.gnn_type}_run_{args.explainer_run}.pt')
    elif args.explanation_metric == 'stability_base':
        bases = ['gcn', 'gat', 'gin', 'sage']
        metric_names = ['jaccard']
        stability_base_scores_dict = {metric: [] for metric in metric_names}
        top_k = 10
        for i in range(len(bases)):
            for j in range(i + 1, len(bases)):
                explanations_base1 = data_utils.load_explanations(args.dataset, args.explainer_name, bases[i], torch.device('cpu'), run=1)
                explanations_base2 = data_utils.load_explanations(args.dataset, args.explainer_name, bases[j], torch.device('cpu'), run=1)
                stability_base_scores = metrics.robustness(explanations_base1, explanations_base2, top_k, metric_names)
                for l in range(len(metric_names)):
                    stability_base_scores_dict[metric_names[l]].append((bases[i], bases[j], stability_base_scores[l]))
        torch.save(stability_base_scores_dict, result_folder + f'stability_base_run_{args.explainer_run}.pt')
    else:
        raise NotImplementedError

    print(f'Finished: {args.dataset}, {args.explainer_name}, {args.gnn_type}, {args.explanation_metric}')
