from gnn_trainer import GNNTrainer
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Mutagenicity',
                        choices=['Mutagenicity', 'Proteins', 'Mutag', 'IMDB-B', 'AIDS', 'NCI1', 'Tree-of-Life', 'Graph-SST2', 'DD', 'REDDIT-B'],
                        help="Dataset name")
    parser.add_argument('--gnn_type', type=str, default='gcn', choices=['gcn', 'gat', 'gin', 'sage'], help='GNN layer type to use.')
    parser.add_argument('--device', type=int, default=0, help='Index of cuda device to use. Default is 0.')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--start_run', type=int, default=1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    trainer = GNNTrainer(dataset_name=args.dataset, gnn_type=args.gnn_type, task='basegnn', device=args.device)

    runs = range(args.start_run, args.start_run + args.runs)
    trainer.run(runs=runs)
