"""Convert a PyG dataset to CLEAR's dataset format"""
import pickle
from argparse import ArgumentParser

import numpy as np
import torch
import torch_geometric as pyg
from sklearn.model_selection import train_test_split

from data_sampler import GraphData

SEED = 7
np.random.seed(SEED)
torch.manual_seed(SEED)

parser = ArgumentParser()
parser.add_argument("--class", "-c", dest="class_", required=True, type=str,
                    help="The dataset class e.g., TUDataset for Mutag.")
parser.add_argument("--name", "-n", required=True,
                    type=str, help="Dataset's name.")
parser.add_argument("--root", "-r", required=True, type=str,
                    help="The argument to be supplied for root while loading the dataset.")
parser.add_argument("--dest", "-d", type=str, default="./", help="Output's path.")
args = parser.parse_args()
print(args)

# * Load the pyg dataset.
dataset = eval(f"pyg.datasets.{args.class_}(name='{args.name}', root='{args.root}')")

# * Create a GraphData object.
# max_num_nodes
max_num_nodes = max((graph.num_nodes for graph in dataset))

# adj_all
adj_all = []
for graph in dataset:
    adj = pyg.utils.to_dense_adj(graph.edge_index).squeeze(0).numpy()
    adj = adj + np.eye(adj.shape[0])
    adj_all.append(adj)

# feature_all
feature_all = [graph.x.numpy() for graph in dataset]

# padded
PADDED = True

# labels_all
labels_all = [np.array(label.item(), dtype=int) for label in dataset.y]

# u_all
U_NUM = 10
u_all = [np.random.choice(U_NUM, size=1).astype(float) for __ in range(len(adj_all))]

# GraphData
dataset_clear = GraphData(
    adj_all=adj_all,
    features_all=feature_all,
    u_all=u_all,
    labels_all=labels_all,
    max_num_nodes=max_num_nodes,
    padded=PADDED,
    index=None,
)


# * Create train_val_test splits.
# Create ten splits as numpy arrays & save each of them in a list.
indices = np.arange(len(dataset_clear))
splits = {"idx_train_list": [], "idx_val_list": [], "idx_test_list":[]}
for __ in range(10):
    # train:val:test = 60:20:20
    train, test, __, __ = train_test_split(
        indices,
        dataset.y[indices],
        stratify=dataset.y[indices],
        test_size=0.2,
        random_state=SEED
    )
    train, val, __, __ = train_test_split(
        train,
        dataset.y[train],
        stratify=dataset.y[train],
        test_size=0.25,
        random_state=SEED
    )
    splits["idx_train_list"].append(train)
    splits["idx_val_list"].append(val)
    splits["idx_test_list"].append(test)


# * Write to disk.
with open(f"{args.dest}/{args.name}.pickle", "wb") as file_0,\
     open(f"{args.dest}/{args.name}_datasplit.pickle", "wb") as file_1:
    pickle.dump(obj=dataset_clear, file=file_0, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(obj=splits, file=file_1, protocol=pickle.HIGHEST_PROTOCOL)
