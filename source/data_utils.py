from torch_geometric.data import Dataset, InMemoryDataset, Data
import os
import glob
import numpy as np
import json
import pickle

from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree, dense_to_sparse
from torch_geometric.transforms import RemoveIsolatedNodes, ToUndirected

from torch.utils.data import random_split
import torch
import torch.nn.functional as F

from torch_geometric.utils import negative_sampling, sort_edge_index, to_dense_adj
import random


class MutagenicityNoisy(Dataset):
    def __init__(self, root, noise, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.root = root
        self.noise = noise
        self.name = f'MutagenicityNoisy{noise}'
        self.cleaned = False
        self.max_graph_size = float('inf')

        self.original_graphs = TUDataset(root=root, name='Mutagenicity', use_node_attr=True)
        self.graph_count = len(self.original_graphs)

        super(MutagenicityNoisy, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return []

    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped"""
        return [f'data_{i}.pt' for i in range(self.graph_count)]

    def download(self):
        pass

    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def num_classes(self) -> int:
        return 2

    @staticmethod
    def sample_negative_edges(graph, num_samples):
        random.seed(0)
        new_edges = negative_sampling(graph.edge_index, num_neg_samples=num_samples * 2, num_nodes=graph.num_nodes, force_undirected=True)
        return new_edges

    def noise_graph(self, graph, num_samples):
        new_edges = self.sample_negative_edges(graph, num_samples)
        new_edge_index = torch.hstack([graph.edge_index, new_edges])
        new_edge_index = sort_edge_index(new_edge_index)
        data = Data(edge_index=new_edge_index.clone(),
                    x=graph.x.clone(),
                    y=graph.y.clone())
        return data

    def process(self):
        for i, graph in enumerate(self.original_graphs):
            data = self.noise_graph(graph, self.noise)
            torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))

    def len(self):
        return self.graph_count

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        data = torch.load(os.path.join(self.processed_dir,
                                       f'data_{idx}.pt'))
        return data


class ProteinsNoisy(Dataset):
    def __init__(self, root, noise, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.root = root
        self.noise = noise
        self.name = f'ProteinsNoisy{noise}'
        self.cleaned = False
        self.max_graph_size = float('inf')

        self.original_graphs = TUDataset(root=root, name='PROTEINS_full', use_node_attr=True)
        self.graph_count = len(self.original_graphs)

        super(ProteinsNoisy, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return []

    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped"""
        return [f'data_{i}.pt' for i in range(self.graph_count)]

    def download(self):
        pass

    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def num_classes(self) -> int:
        return 2

    @staticmethod
    def sample_negative_edges(graph, num_samples):
        random.seed(0)
        new_edges = negative_sampling(graph.edge_index, num_neg_samples=num_samples * 2, num_nodes=graph.num_nodes,
                                      force_undirected=True)
        return new_edges

    def noise_graph(self, graph, num_samples):
        new_edges = self.sample_negative_edges(graph, num_samples)
        new_edge_index = torch.hstack([graph.edge_index, new_edges])
        new_edge_index = sort_edge_index(new_edge_index)
        data = Data(edge_index=new_edge_index.clone(),
                    x=graph.x.clone(),
                    y=graph.y.clone())
        return data

    def process(self):
        for i, graph in enumerate(self.original_graphs):
            data = self.noise_graph(graph, self.noise)
            torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))

    def len(self):
        return self.graph_count

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        data = torch.load(os.path.join(self.processed_dir,
                                       f'data_{idx}.pt'))
        return data


class IMDBNoisy(Dataset):
    def __init__(self, root, noise, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.root = root
        self.noise = noise
        self.name = f'IMDBNoisy{noise}'
        self.cleaned = False
        self.max_graph_size = float('inf')

        self.original_graphs = TUDataset(root=root, name='IMDB-BINARY', pre_transform=IMDBPreTransform())
        self.graph_count = len(self.original_graphs)

        super(IMDBNoisy, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return []

    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped"""
        return [f'data_{i}.pt' for i in range(self.graph_count)]

    def download(self):
        pass

    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def num_classes(self) -> int:
        return 2

    @staticmethod
    def sample_negative_edges(graph, num_samples):
        random.seed(0)
        new_edges = negative_sampling(graph.edge_index, num_neg_samples=num_samples * 2, num_nodes=graph.num_nodes,
                                      force_undirected=True)
        return new_edges

    def noise_graph(self, graph, num_samples):
        new_edges = self.sample_negative_edges(graph, num_samples)
        new_edge_index = torch.hstack([graph.edge_index, new_edges])
        new_edge_index = sort_edge_index(new_edge_index)
        data = Data(edge_index=new_edge_index.clone(),
                    x=graph.x.clone(),
                    y=graph.y.clone())
        return data

    def process(self):
        for i, graph in enumerate(self.original_graphs):
            data = self.noise_graph(graph, self.noise)
            torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))

    def len(self):
        return self.graph_count

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        data = torch.load(os.path.join(self.processed_dir,
                                       f'data_{idx}.pt'))
        return data


class AIDSNoisy(Dataset):
    def __init__(self, root, noise, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.root = root
        self.noise = noise
        self.name = f'AIDSNoisy{noise}'
        self.cleaned = False
        self.max_graph_size = float('inf')

        self.original_graphs = TUDataset(root=root, name='AIDS', use_node_attr=True)
        self.graph_count = len(self.original_graphs)

        super(AIDSNoisy, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return []

    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped"""
        return [f'data_{i}.pt' for i in range(self.graph_count)]

    def download(self):
        pass

    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def num_classes(self) -> int:
        return 2

    @staticmethod
    def sample_negative_edges(graph, num_samples):
        random.seed(0)
        new_edges = negative_sampling(graph.edge_index, num_neg_samples=num_samples * 2, num_nodes=graph.num_nodes,
                                      force_undirected=True)
        return new_edges

    def noise_graph(self, graph, num_samples):
        new_edges = self.sample_negative_edges(graph, num_samples)
        new_edge_index = torch.hstack([graph.edge_index, new_edges])
        new_edge_index = sort_edge_index(new_edge_index)
        data = Data(edge_index=new_edge_index.clone(),
                    x=graph.x.clone(),
                    y=graph.y.clone())
        return data

    def process(self):
        for i, graph in enumerate(self.original_graphs):
            data = self.noise_graph(graph, self.noise)
            torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))

    def len(self):
        return self.graph_count

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        data = torch.load(os.path.join(self.processed_dir,
                                       f'data_{idx}.pt'))
        return data


def undirected_graph(data):
    data.edge_index = torch.cat([torch.stack([data.edge_index[1], data.edge_index[0]], dim=0),
                                 data.edge_index], dim=1)
    return data


def split(data, batch):
    # i-th contains elements from slice[i] to slice[i+1]
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])
    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)
    data.__num_nodes__ = np.bincount(batch).tolist()

    slices = dict()
    slices['x'] = node_slice
    slices['edge_index'] = edge_slice
    slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    return data, slices


class SentiGraphDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=undirected_graph):
        self.name = name
        super(SentiGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices, self.supplement = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['node_features', 'node_indicator', 'sentence_tokens', 'edge_index',
                'graph_labels', 'split_indices']

    @property
    def processed_file_names(self):
        return ['data.pt']

    @staticmethod
    def read_file(folder, prefix, name):
        file_path = os.path.join(folder, prefix + f'_{name}.txt')
        return np.genfromtxt(file_path, dtype=np.int64)

    @staticmethod
    def read_sentigraph_data(folder: str, prefix: str):
        txt_files = glob.glob(os.path.join(folder, "{}_*.txt".format(prefix)))
        json_files = glob.glob(os.path.join(folder, "{}_*.json".format(prefix)))
        txt_names = [f.split(os.sep)[-1][len(prefix) + 1:-4] for f in txt_files]
        json_names = [f.split(os.sep)[-1][len(prefix) + 1:-5] for f in json_files]
        names = txt_names + json_names

        with open(os.path.join(folder, prefix + "_node_features.pkl"), 'rb') as f:
            x: np.array = pickle.load(f)
        x: torch.FloatTensor = torch.from_numpy(x)
        edge_index: np.array = SentiGraphDataset.read_file(folder, prefix, 'edge_index')
        edge_index: torch.tensor = torch.tensor(edge_index, dtype=torch.long).T
        batch: np.array = SentiGraphDataset.read_file(folder, prefix, 'node_indicator') - 1  # from zero
        y: np.array = SentiGraphDataset.read_file(folder, prefix, 'graph_labels')
        y: torch.tensor = torch.tensor(y, dtype=torch.long)

        supplement = dict()
        if 'split_indices' in names:
            split_indices: np.array = SentiGraphDataset.read_file(folder, prefix, 'split_indices')
            split_indices = torch.tensor(split_indices, dtype=torch.long)
            supplement['split_indices'] = split_indices
        if 'sentence_tokens' in names:
            with open(os.path.join(folder, prefix + '_sentence_tokens.json')) as f:
                sentence_tokens: dict = json.load(f)
            supplement['sentence_tokens'] = sentence_tokens

        data = Data(x=x, edge_index=edge_index, y=y)
        data, slices = split(data, batch)

        return data, slices, supplement

    def process(self):
        # Read data into huge `Data` list.
        self.data, self.slices, self.supplement = SentiGraphDataset.read_sentigraph_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices, self.supplement), self.processed_paths[0])


def split_data(data, train_ratio=0.8, val_ratio=0.1):
    gen = torch.Generator().manual_seed(0)
    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)
    test_size = len(data) - train_size - val_size
    splits = random_split(data, lengths=[train_size, val_size, test_size], generator=gen)
    return splits, [split.indices for split in splits]


def sample_negative_edges(graph, num_samples):
    random.seed(0)
    new_edges = negative_sampling(graph.edge_index, num_neg_samples=num_samples * 2, num_nodes=graph.num_nodes, force_undirected=True)
    return new_edges


def noise_graph(graph, num_samples):
    new_edges = sample_negative_edges(graph, num_samples)
    new_edge_index = torch.hstack([graph.edge_index, new_edges])
    new_edge_index = sort_edge_index(new_edge_index)
    data = Data(edge_index=new_edge_index.clone(),
                x=graph.x.clone(),
                y=graph.y.clone())
    return data


def adj_from_edge_index(graph):
    if graph.edge_index.shape[1] == 0:
        adj = torch.zeros(graph.num_nodes, graph.num_nodes)
    else:
        adj = to_dense_adj(graph.edge_index, edge_attr=graph.edge_weight, max_num_nodes=graph.num_nodes)[0]
    return adj


def get_noisy_dataset_name(dataset_name, noise):
    if dataset_name == 'Mutagenicity':
        return f'MutagenicityNoisy{noise}'
    elif dataset_name == 'Proteins':
        return f'ProteinsNoisy{noise}'
    elif dataset_name == 'IMDB-B':
        return f'IMDBNoisy{noise}'
    elif dataset_name == 'AIDS':
        return f'AIDSNoisy{noise}'
    else:
        raise NotImplementedError


class IMDBPreTransform(object):
    def __call__(self, data):
        data.x = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
        data.x = F.one_hot(data.x, num_classes=136).to(torch.float)
        return data


class REDDITPreTransform(object):
    def __call__(self, data):
        data.x = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
        data.x = F.one_hot(data.x, num_classes=3063).to(torch.float)
        return data


def load_dataset(dataset_name, root='data/'):
    if dataset_name == 'Mutagenicity':
        data = TUDataset(root=root, name='Mutagenicity', use_node_attr=True)
    elif "MutagenicityNoisy" in dataset_name:
        noise = int(dataset_name[17:])
        data = MutagenicityNoisy(root=root, noise=noise)
    elif dataset_name == 'Mutag':
        data = TUDataset(root=root, name='MUTAG', use_node_attr=True)
    elif dataset_name == 'Proteins':
        data = TUDataset(root=root, name='PROTEINS_full', use_node_attr=True)
    elif "ProteinsNoisy" in dataset_name:
        noise = int(dataset_name[13:])
        data = ProteinsNoisy(root=root, noise=noise)
    elif dataset_name == 'IMDB-B':
        data = TUDataset(root=root, name='IMDB-BINARY', pre_transform=IMDBPreTransform())
    elif 'IMDBNoisy' in dataset_name:
        noise = int(dataset_name[9:])
        data = IMDBNoisy(root=root, noise=noise)
    elif dataset_name == 'AIDS':
        data = TUDataset(root=root, name='AIDS', use_node_attr=True)
    elif 'AIDSNoisy' in dataset_name:
        noise = int(dataset_name[9:])
        data = AIDSNoisy(root=root, noise=noise)
    elif dataset_name == 'NCI1':
        data = TUDataset(root=root, name='NCI1', use_node_attr=True)
    elif dataset_name == 'Graph-SST2':
        data = SentiGraphDataset(root=root, name='Graph-SST2')
    elif dataset_name == 'DD':
        data = TUDataset(root=root, name='DD', use_node_attr=True)
    elif dataset_name == 'REDDIT-B':
        data = TUDataset(root=root, name='REDDIT-BINARY', pre_transform=REDDITPreTransform())
    else:
        raise NotImplementedError(f'Dataset: {dataset_name} is not implemented!')

    return data


def load_explanations(dataset_name, explainer_name, gnn_type, device, run):
    path = f'data/{dataset_name}/{explainer_name}/explanations_{gnn_type}_run_{run}.pt'
    return torch.load(path, map_location=device)


def load_explanations_test(dataset_name, explainer_name, gnn_type, device, run):
    path = f'data/{dataset_name}/{explainer_name}/explanations_{gnn_type}_run_{run}_test.pt'
    return torch.load(path, map_location=device)


def load_explanations_noisy(dataset_name, explainer_name, gnn_type, device, run, k):
    path = f'data/{dataset_name}/{explainer_name}/explanations_{gnn_type}_run_{run}_noise_{k}.pt'
    return torch.load(path, map_location=device)


def select_top_k_explanations(dataset, top_k):
    top_k_dataset = []
    for graph in dataset:
        if graph.edge_index.shape[1] > 0 and not graph.edge_weight.sum().isnan().item():
            directed_edge_weight = graph.edge_weight[graph.edge_index[0] <= graph.edge_index[1]]
            directed_edge_index = graph.edge_index[:, graph.edge_index[0] <= graph.edge_index[1]]
            idx = directed_edge_weight >= directed_edge_weight.topk(min(top_k, directed_edge_weight.shape[0]))[0][-1]
            directed_edge_index = directed_edge_index[:, idx]
            new_data = Data(
                edge_index=directed_edge_index.clone(),
                x=graph.x.clone()
            )
            new_data = ToUndirected()(new_data)
            new_data = RemoveIsolatedNodes()(new_data)
            new_data.y = graph.y.clone()
            top_k_dataset.append(new_data)
        else:
            top_k_dataset.append(graph)
    return top_k_dataset


if __name__ == '__main__':
    # generate datasets
    for dataset_name in ['Mutagenicity', 'Proteins', 'IMDB-B', 'Mutag', 'AIDS', 'NCI1', 'Graph-SST2', 'DD', 'REDDIT-B']:
        dataset = load_dataset(dataset_name)

    # generate noisy datasets
    for dataset_name in ['Mutagenicity', 'Proteins', 'IMDB', 'AIDS']:
        for noise in [1, 2, 3, 4, 5]:
            load_dataset(f'{dataset_name}Noisy{noise}')
