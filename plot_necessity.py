import os.path

import matplotlib.pyplot as plt
import torch
from matplotlib import rcParams

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

methods = ["pgexplainer", "tagexplainer_1", 'cff_1.0', 'rcexplainer_1.0', 'gnnexplainer', 'gem', 'subgraphx']
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

# read results
gnn_type = 'gcn'
dataset_results = {dataset: {} for dataset in datasets}
for dataset in datasets:
    for method in methods:
        path = f"data/{dataset}/{method}/faithfulness_with_removal_{gnn_type}_run_1.pt"
        if os.path.exists(path):
            faithfulness_results = torch.load(path)
            dataset_results[dataset][method] = faithfulness_results['necessity']
        else:
            dataset_results[dataset][method] = [None] * 5

nrows = 2
ncols = 3
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 4), sharex=True)

labelsize = 14
ticksize = 12
markersize = 6
linewidth = 1.5

xticks = [2, 4, 6, 8, 10]

count = 0
ls = [None] * len(methods)
for row_i in range(nrows):
    for col_i in range(ncols):
        ax = axes[row_i][col_i]
        dataset_name = datasets[count]

        for i, method_key in enumerate(methods):
            if method_key in dataset_results[dataset_name]:
                ls_results, = ax.plot(xticks, dataset_results[dataset_name][method_key], label=method_key, marker=markers[method_key], color=colors[method_key])
                if count == 0:
                    ls[i] = ls_results

        ax.minorticks_off()
        ax.set_xticks(xticks)
        ax.set_title(dataset_name, fontsize=labelsize)
        ax.set_xticklabels(xticks)
        if col_i == 0:
            ax.set_ylabel('Necessity', fontsize=labelsize)
        ax.tick_params(axis='x', labelsize=ticksize)
        ax.tick_params(axis='y', labelsize=ticksize)
        ax.set_xlim(1, 11)
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

fig.savefig(f'plots/necessity_{gnn_type}.pdf', bbox_inches='tight')
plt.show(tight_layout=True)