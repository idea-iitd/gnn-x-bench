import os
import torch
import numpy as np
import copy

from matplotlib import rcParams
import matplotlib.pyplot as plt

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

methods = ["pgexplainer", "tagexplainer_1", 'cff_1.0', 'rcexplainer_1.0', 'gnnexplainer', 'gem']
method_name_map = {
    "pgexplainer": "PGExplainer",
    "tagexplainer_1": "TAGExplainer",
    'cff_1.0': r'CF$^2$',
    'rcexplainer_1.0': 'RCExplainer',
    'gnnexplainer': 'GNNExplainer',
    'gem': 'GEM',
    'subgraphx': 'SubgraphX'
}
datasets = ["Mutagenicity", "Proteins", "Mutag", "IMDB-B", "AIDS", "NCI1"]

markers = {
    "pgexplainer": "v",
    "tagexplainer_1": "<",
    "cff_1.0": "s",
    "rcexplainer_1.0": "P",
    "gnnexplainer": "X",
    "gem": "d",
    "subgraphx": "h"
}

colors = {
    "pgexplainer": "r",
    "tagexplainer_1": "g",
    "cff_1.0": "m",
    "rcexplainer_1.0": "y",
    "gnnexplainer": "k",
    "gem": "orange",
    "subgraphx": "brown"
}

folded = True

# read results
gnn_type = 'gcn'
dataset_results = {}
explainers_results = {}
for dataset in datasets:
    path = f'data/{dataset}/basegnn/{gnn_type}-max/all_scores_1_10.pt'
    scores = torch.load(path)
    test_scores = scores['test_scores']
    auc_test = np.mean(test_scores["auc_or_r2"])
    for explainer in methods:
        mean_res = []
        std_res = []
        for k in range(1, 11, 1):
            path = f'data/{dataset}/reproducibility_{k}/{explainer}/{gnn_type}-max/all_scores_1_10.pt'
            if os.path.exists(path):
                scores_k = torch.load(path)
                test_scores_k = scores_k['test_scores']
                mean_res.append(np.mean(test_scores_k["auc_or_r2"]) / auc_test)
                std_res.append(np.std(test_scores_k["auc_or_r2"]) / auc_test)
            else:
                mean_res.append(None)
                std_res.append(None)
        explainers_results[explainer] = {'mean': mean_res, 'std': std_res}
    dataset_results[dataset] = copy.deepcopy(explainers_results)

nrows = 2
ncols = 3
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 4), sharex=True)

labelsize = 14
ticksize = 12
markersize = 6
linewidth = 1.5

xticks = range(1, 11, 1)

count = 0
ls = [None] * len(methods)
for row_i in range(nrows):
    for col_i in range(ncols):
        ax = axes[row_i][col_i]
        dataset_name = datasets[count]

        if not folded:
            for i, method_key in enumerate(methods):
                if method_key in dataset_results[dataset_name]:
                    ls_results, = ax.plot(xticks, dataset_results[dataset_name][method_key]['mean'], label=method_key, marker=markers[method_key], color=colors[method_key])
                    if count == 0:
                        ls[i] = ls_results
        else:
            for i, method_key in enumerate(methods):
                if method_key in dataset_results[dataset_name]:
                    if 'mean' in dataset_results[dataset_name][method_key] and None not in dataset_results[dataset_name][method_key]['mean']:
                        if None in dataset_results[dataset_name][method_key]['std']:
                            ax.plot(xticks, dataset_results[dataset_name][method_key]['mean'], label=method_key, marker=markers[method_key], color=colors[method_key])
                        else:
                            ax.errorbar(x=xticks, y=dataset_results[dataset_name][method_key]['mean'], yerr=dataset_results[dataset_name][method_key]['std'],
                                        label=method_key, marker=markers[method_key], color=colors[method_key], capsize=5)
            if count == 0:
                handles, labels = ax.get_legend_handles_labels()
                ls = [h[0] for h in handles]

        ax.minorticks_off()
        ax.set_xticks(xticks)
        ax.set_title(dataset_name, fontsize=labelsize)
        ax.set_xticklabels(xticks)
        if col_i == 0:
            ax.set_ylabel(r'Reproducibility$^+$', fontsize=labelsize)
        ax.tick_params(axis='x', labelsize=ticksize)
        ax.tick_params(axis='y', labelsize=ticksize)
        ax.set_xlim(0.5, 10.5)
        ax.grid(True)

        count += 1

# Add common x axis.
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel('size (#edges in the explanation)', fontsize=labelsize)

fig.tight_layout()
fig.subplots_adjust(left=0.035, bottom=0.16, right=0.99, wspace=0.22)

method_names = [method_name_map[method] for method in methods]
axes[1][1].legend(handles=ls, labels=method_names,
                  loc='upper center', bbox_to_anchor=(0.4, -0.2), fancybox=False, shadow=False, ncol=len(methods), fontsize=labelsize)

if not folded:
    fig.savefig(f'plots/reproducibility_{gnn_type}.pdf', bbox_inches='tight')
else:
    fig.savefig(f'plots/reproducibility_{gnn_type}_fold.pdf', bbox_inches='tight')
plt.show(tight_layout=True)
