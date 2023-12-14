from gnn_trainer import GNNTrainer
import argparse
import data_utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Mutagenicity',
                        choices=['Mutagenicity', 'Proteins', 'Mutag', 'IMDB-B', 'AIDS', 'NCI1'],
                        help="Dataset name")
    parser.add_argument('--gnn_type', type=str, default='gcn', choices=['gcn', 'gat', 'gin', 'sage'], help='GNN layer type to use.')
    parser.add_argument('--device', type=int, default=0, help='Index of cuda device to use. Default is 0.')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--start_run', type=int, default=1)
    parser.add_argument('--explainer_name', type=str, choices=['pgexplainer', 'tagexplainer_1', 'tagexplainer_2', 'cff_1.0',
                                                               'rcexplainer_1.0', 'gnnexplainer', 'gem', 'subgraphx'])
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print(f'Started: {args.dataset}, {args.gnn_type}, {args.explainer_name}')

    for top_k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        trainer = GNNTrainer(dataset_name=args.dataset, gnn_type=args.gnn_type, task='reproducibility', device=args.device, explainer_name=args.explainer_name, top_k=top_k)
        runs = range(args.start_run, args.start_run + args.runs)
        trainer.run(runs=runs)

    print(f'Ended: {args.dataset}, {args.gnn_type}, {args.explainer_name}')

