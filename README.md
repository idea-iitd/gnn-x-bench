# gnn-x-bench
Benchmarking GNN explainers.

## Requirements

The easiest way to install the dependencies is via [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). Once you have conda installed, run this command:

```setup
conda env create -f env.yml
```

If you want to install dependencies manually, we tested our code in Python 3.9.7 using the following main dependencies:

- [PyTorch](https://pytorch.org/get-started/locally/) v1.11.0
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) v2.1.0
- [NetworkX](https://networkx.org/documentation/networkx-2.5/install.html) v2.7.1
- [NumPY](https://numpy.org/install/) v1.22.3

Experiments were conducted using the Ubuntu 18.04 operating system on an NVIDIA DGX Station equipped with four V100 GPU cards, each having 128GB of GPU memory. 
The system also included 256GB of RAM and a 20-core Intel Xeon E5-2698 v4 2.2 GHz CPU.

## Usage

### Data installation

Every datasets except for Graph-SST2 is ready to install from PyTorch Geometric libraries. For Graph-SST2, you can download the dataset from 
[here](https://drive.google.com/file/d/1-PiLsjepzT8AboGMYLdVHmmXPpgR8eK1/view?usp=sharing) and put it in `data/` directory.

Then, you can run the following command to preprocess the data. This will preprocess every dataset including generating noisy variants of four datasets.

```setup
python source/data_utils.py
```

### Training Base GNNs

We provide the pretrained models for every dataset and gnn architectures. However, if you want to train the models from scratch, you can run the following command:

```setup
python source/basegnn.py --dataset <dataset_name> --gnn_type <gnn_type> --runs 1
```

We modified GAT, GIN, SAGE implementation of PyTorch Geometric to support our training pipeline. You can find the modified version of the code in `source/wrappers/` directory.

### Training Explainers

We provide the pretrained models for every dataset and inductive explainers. However, you may want to train the explainers from scratch.

Each explainer has their own code and training pipeline. You can find the code for each explainer in `source/` directory.

We also provide shell scripts to run the explainers. You can find the scripts in main directory. For example, to run the GNNExplainer, you can run the following command:

```setup
./gnnexplainer.sh
```

Please check the script files to see which command should be run to receive which results.

### Explanations and Evaluation

We provide the explanations and evaluation scores for every dataset and explainer. You can find the explanations in `data/<dataset_name>/<explainer_name>` directory.

If you want to reproduce the evaluations, you can run the following command:

```setup
./generate_results.sh
```

### Reproducibility Experiments

Reproducibility experiments needs the explanations from the explainers. It trains from-scratch GNNs using the explanations and evaluate them. We use top-1 to top-10 from explanations and 
re-train them.

We already provide our run performance but, you can run the following command to run reproducibility experiments:

```setup
./reproducibility.sh
```

### Visualization

We use the following files to plot and print the results:

- [plot_sufficiency.py](plot_sufficiency.py): Plots the sufficiency results.
- [plot_sufficiency_inductive.py](plot_sufficiency_inductive.py): Plots the sufficiency results for inductive explainers.
- [plot_necessity.py](plot_necessity.py): Plots the necessity results.
- [plot_stability.py](plot_stability.py): Plots the stability results for noise addition.
- [plot_reproducibility.py](plot_reproducibility.py): Plots the results for the reproducibility experiments.
- [stability_seed_scores.py](stability_seed_scores.py): Prints the results for the stability based on Explainer seed experiments.
- [stability_base_scores.py](stability_base_scores.py): Prints the results for the stability based on GNN base experiments.

Links: \
[GNNX-BENCH: Unravelling the Utility of Perturbation-based GNN Explainers through In-depth Benchmarking](https://arxiv.org/abs/2310.01794) \
[Slides](https://github.com/idea-iitd/gnn-x-bench/blob/main/ICLR_GNNXBENCH.pdf)

