import torch

datasets = ['Mutagenicity', 'Proteins', 'Mutag', 'IMDB-B', 'AIDS', 'NCI1']
explainer_names = ['pgexplainer', 'tagexplainer_1', 'rcexplainer_1.0', 'gnnexplainer', 'cff_1.0']
explainer_name_map = {
    'pgexplainer': 'PGExplainer',
    'tagexplainer_1': 'TAGExplainer',
    'rcexplainer_1.0': 'RCExplainer',
    'gnnexplainer': 'GNNExplainer',
    'cff_1.0': r'CF$^2$'
}

gnn_model = 'gcn'
for dataset in datasets:
    for explainer in explainer_names:
        print(f"{dataset} {explainer}")
        path = f'data/{dataset}/{explainer}/stability_seed_{gnn_model}_run_1.pt'
        jaccard_scores = torch.load(path, map_location=torch.device('cpu'))
        for from_, to_, score_ in jaccard_scores['jaccard']:
            print(from_, to_, round(score_, 2))
