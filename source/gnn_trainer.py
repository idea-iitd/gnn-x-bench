import torch
import numpy as np
import random
import os
import time

import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool
from torch_geometric.loader import DataLoader
from wrappers.gin import GINConv
from wrappers.gat import GATConvModified
from wrappers.sage import SAGEConvModified
from tqdm import tqdm

import data_utils
import metrics


class GNN(torch.nn.Module):

    def __init__(self, num_features, num_classes=2, num_layers=3, dim=20, dropout=0.0, layer='gcn', pool='max'):
        super(GNN, self).__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dim = dim
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.pool = pool
        self.layer = None
        if layer == 'gcn':
            self.layer = GCNConv
        elif layer == 'gat':
            self.layer = GATConvModified
        elif layer == 'gin':
            self.layer = GINConv
        elif layer == 'sage':
            self.layer = SAGEConvModified
        else:
            raise NotImplementedError(f'Layer: {layer} is not implemented!')

        # First GCN layer.
        self.convs.append(self.layer(num_features, dim))
        self.bns.append(torch.nn.BatchNorm1d(dim))

        # Follow-up GCN layers.
        for i in range(self.num_layers - 1):
            self.convs.append(self.layer(dim, dim))
            self.bns.append(torch.nn.BatchNorm1d(dim))

        # Fully connected layer.
        self.fc = torch.nn.Linear(dim, num_classes)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, self.layer):
                m.reset_parameters()
            elif isinstance(m, torch.nn.BatchNorm1d):
                m.reset_parameters()
            elif isinstance(m, torch.nn.Linear):
                m.reset_parameters()

    def forward(self, data, edge_weight=None):

        x = data.x.float()
        edge_index = data.edge_index
        batch = data.batch

        # GCNs.
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight=edge_weight)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout after every layer.

        # Pooling and FCs.
        node_embeddings = x
        if self.pool == 'max':
            graph_embedding = global_max_pool(node_embeddings, batch)
        else:
            raise NotImplementedError(f'Pooling: {self.pool} is not implemented!')
        out = self.fc(graph_embedding)

        return node_embeddings, graph_embedding, out


class GnnSynthetic(torch.nn.Module):
    """3-layer GCN used in GNN Explainer synthetic tasks"""
    def __init__(self, nfeat, nhid, nout, nclass, dropout):
        super(GnnSynthetic, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nhid)
        self.gc3 = GCNConv(nhid, nout)
        self.fc = torch.nn.Linear(nhid + nhid + nout, nclass)
        self.dropout = dropout
        self.dim = 20 # for RCExplainer

    def forward(self, data, edge_weight=None):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        target_node = data.target_node

        # convert edge index to dense_adj
        x1 = F.relu(self.gc1(x, edge_index, edge_weight))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x1, edge_index, edge_weight))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x3 = self.gc3(x2, edge_index, edge_weight)

        node_embeddings = torch.cat((x1, x2, x3), dim=1)
        graph_embedding = global_max_pool(node_embeddings, batch)

        x = self.fc(node_embeddings)
        # out = F.log_softmax(x, dim=1)
        out = x[target_node: target_node+1, :] # x[[target_node]] has different behaviours when target_node is an int and when it is a tensor.

        return node_embeddings, graph_embedding, out

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


class GNNTrainer:
    def __init__(self, dataset_name, gnn_type, task, device, explainer_name=None, top_k=10):

        self.dataset_name = dataset_name
        self.gnn_type = gnn_type
        self.task = task
        self.device_name = f'cuda:{device}' if torch.cuda.is_available() and device != 'cpu' else 'cpu'
        self.device = torch.device(self.device_name)
        self.explainer_name = explainer_name
        self.top_k = top_k

        self.num_layers = 3
        self.dim = 20
        self.dropout = 0.0
        self.pool = 'max'

        self.epochs = 1000
        self.batch_size = 128
        self.lr = 0.001

        self.model = None
        self.optimizer = None

        # Load and split the datasets based on the task
        if self.task == 'basegnn':
            self.dataset = data_utils.load_dataset(self.dataset_name)
        elif self.task == 'reproducibility':
            assert self.explainer_name is not None
            if self.explainer_name == 'subgraphx':  # only for subgraphx because of time constraints
                self.dataset = data_utils.load_explanations_test(self.dataset_name, self.explainer_name, self.gnn_type, device='cpu', run=1)
            else:
                self.dataset = data_utils.load_explanations(self.dataset_name, self.explainer_name, self.gnn_type, device='cpu', run=1)
            self.dataset = data_utils.select_top_k_explanations(self.dataset, self.top_k)

        splits, indices = data_utils.split_data(self.dataset)
        self.train_set, self.valid_set, self.test_set = splits
        self.train_loader, self.valid_loader, self.test_loader = None, None, None

        # Logging.
        if self.explainer_name is None:
            self.gnn_folder = f'data/{self.dataset_name}/{self.task}/{self.gnn_type}-{self.pool}/'
        else:
            self.gnn_folder = f'data/{self.dataset_name}/{self.task}_{self.top_k}/{self.explainer_name}/{self.gnn_type}-{self.pool}/'
        if not os.path.exists(self.gnn_folder):
            os.makedirs(self.gnn_folder)
        self.log_file = os.path.join(self.gnn_folder, f'log.txt')
        with open(self.log_file, 'w') as _:
            pass

        self.method = 'classification'

    def load(self, run):
        if self.dataset_name in ['syn1', 'syn4', 'syn5']:
            self.model = GnnSynthetic(
                nfeat=self.dataset.num_features,
                nhid=20,
                nout=20,
                nclass=self.dataset.num_classes,
                dropout=0.0
            ).to(self.device)
            self.model.load_state_dict(torch.load(os.path.join(self.gnn_folder, f'gcn_3layer_{self.dataset_name}.pt'), map_location=self.device))
        else:
            self.model = GNN(
                num_features=self.dataset.num_features,
                num_classes=self.dataset.num_classes,
                num_layers=self.num_layers,
                dim=self.dim,
                dropout=self.dropout,
                layer=self.gnn_type,
                pool=self.pool,
            ).to(self.device)
            self.model.load_state_dict(torch.load(os.path.join(self.gnn_folder, f'best_model_run_{run}.pt'), map_location=self.device))
        return self.model

    @torch.no_grad()
    def load_gnn_outputs(self, run):
        node_embeddings_path = os.path.join(self.gnn_folder, f'node_embeddings_run_{run}.pt')
        graph_embeddings_path = os.path.join(self.gnn_folder, f'graph_embeddings_run_{run}.pt')
        outs_path = os.path.join(self.gnn_folder, f'outs_run_{run}.pt')

        if os.path.exists(node_embeddings_path) and os.path.exists(graph_embeddings_path) and os.path.exists(outs_path):
            node_embeddings = torch.load(node_embeddings_path, map_location=self.device)
            graph_embeddings = torch.load(graph_embeddings_path, map_location=self.device)
            outs = torch.load(outs_path, map_location=self.device)
            return node_embeddings, graph_embeddings, outs
        else:
            self.model = self.load(run)
            self.model.eval()
            loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
            graph_embeddings, node_embeddings, outs = [], [], []
            for batch in tqdm(loader):
                node_emb, graph_emb, out = self.model(batch.to(self.device))
                max_batch_number = max(batch.batch)
                for i in range(max_batch_number + 1):
                    idx = torch.where(batch.batch == i)[0]
                    node_embeddings.append(node_emb[idx])
                graph_embeddings.append(graph_emb)
                outs.append(out)
            graph_embeddings = torch.cat(graph_embeddings)
            outs = torch.cat(outs)
            torch.save([node_embedding for node_embedding in node_embeddings], node_embeddings_path)
            torch.save(graph_embeddings, graph_embeddings_path)
            torch.save(outs, outs_path)
            return node_embeddings, graph_embeddings, outs

    def run(self, runs):
        train_scores = {'accuracy_or_mae': [], 'auc_or_r2': [], 'ap_or_mse': []}
        valid_scores = {'accuracy_or_mae': [], 'auc_or_r2': [], 'ap_or_mse': []}
        test_scores = {'accuracy_or_mae': [], 'auc_or_r2': [], 'ap_or_mse': []}
        eval_times = []
        for run in tqdm(runs, desc='Run'):
            self.one_run(run)

            # evaluation
            start_eval = time.time()
            train_loss, train_auc_or_r2, train_ap_or_mse, train_acc_or_mae, train_preds, train_grounds = self.eval(self.train_loader)
            valid_loss, valid_auc_or_r2, valid_ap_or_mse, valid_acc_or_mae, valid_preds, valid_grounds = self.eval(self.valid_loader)
            test_loss, test_auc_or_r2, test_ap_or_mse, test_acc_or_mae, test_preds, test_grounds = self.eval(self.test_loader)
            train_scores['auc_or_r2'].append(train_auc_or_r2)
            train_scores['ap_or_mse'].append(train_ap_or_mse)
            train_scores['accuracy_or_mae'].append(train_acc_or_mae)
            valid_scores['auc_or_r2'].append(valid_auc_or_r2)
            valid_scores['ap_or_mse'].append(valid_ap_or_mse)
            valid_scores['accuracy_or_mae'].append(valid_acc_or_mae)
            test_scores['auc_or_r2'].append(test_auc_or_r2)
            test_scores['ap_or_mse'].append(test_ap_or_mse)
            test_scores['accuracy_or_mae'].append(test_acc_or_mae)

            torch.save((train_preds, train_grounds), os.path.join(self.gnn_folder, f'train_predictions_run_{run}.pt'))
            torch.save((valid_preds, valid_grounds), os.path.join(self.gnn_folder, f'valid_predictions_run_{run}.pt'))
            torch.save((test_preds, test_grounds), os.path.join(self.gnn_folder, f'test_predictions_run_{run}.pt'))
            end_eval = time.time()
            eval_times.append(end_eval - start_eval)

        self.log(train_scores, valid_scores, test_scores, eval_times, runs)

    def one_run(self, run):
        random.seed(run)
        torch.manual_seed(run)
        torch.cuda.manual_seed(run)
        np.random.seed(run)

        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.valid_loader = DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True, num_workers=0)

        # Initialize the model.
        num_features = self.dataset[0].x.shape[1]
        num_classes = len(torch.unique(torch.tensor([self.dataset[i].y for i in range(len(self.dataset))])))

        self.model = GNN(
            num_features=num_features,
            num_classes=num_classes,
            num_layers=self.num_layers,
            dim=self.dim,
            dropout=self.dropout,
            layer=self.gnn_type,
            pool=self.pool,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_valid = float('inf')
        patience = int(self.epochs / 5)
        cur_patience = 0
        for _ in range(self.epochs):
            self.train()
            valid_loss = self.eval(self.valid_loader)[0]
            if valid_loss < best_valid:
                cur_patience = 0
                best_valid = valid_loss
                torch.save(self.model.state_dict(), os.path.join(self.gnn_folder, f'best_model_run_{run}.pt'))
            else:
                cur_patience += 1
                if cur_patience >= patience:
                    break

        self.model.load_state_dict(torch.load(os.path.join(self.gnn_folder, f'best_model_run_{run}.pt'), map_location=self.device))

    def iteration(self, batch):
        out = self.model(batch.to(self.device))[-1]

        if self.method == 'classification':
            loss = F.nll_loss(F.log_softmax(out, dim=-1), batch.y.flatten().long())
        else:
            loss = F.mse_loss(out.flatten(), batch.y.flatten())
        return loss, out

    def train(self):
        self.model.train()
        total_loss = 0

        for train_batch in self.train_loader:
            self.optimizer.zero_grad()
            loss, out = self.iteration(train_batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * out.shape[0]

        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def eval(self, eval_loader):
        self.model.eval()
        total_loss = 0

        preds = []
        grounds = []
        for eval_batch in eval_loader:
            loss, out = self.iteration(eval_batch)
            preds.append(out)
            grounds.append(eval_batch.y.flatten())
            total_loss += loss.item() * out.shape[0]  # eval_batch.num_graphs

        if self.method == 'classification':
            preds = torch.softmax(torch.cat(preds, dim=0), dim=1)
            grounds = torch.cat(grounds, dim=0)
            return total_loss, metrics.auc(grounds, preds), metrics.ap(grounds, preds), metrics.accuracy(grounds, preds), preds, grounds
        else:
            preds = torch.cat(preds, dim=0).flatten()
            grounds = torch.cat(grounds, dim=0).flatten()
            return total_loss, metrics.r_squared(grounds, preds), metrics.mse(grounds, preds), metrics.mae(grounds, preds), preds, grounds

    def log(self, train_scores, valid_scores, test_scores, eval_times, runs):
        all_scores = {'train': train_scores, 'valid': valid_scores, 'test_scores': test_scores}
        torch.save(all_scores, self.gnn_folder + f'all_scores_{runs[0]}_{runs[-1]}.pt')
        torch.save(eval_times, self.gnn_folder + f'eval_times_{runs[0]}_{runs[-1]}.pt')
        with open(self.log_file, 'a') as f:
            print(file=f)
            print(f"Train Scores = {train_scores}", file=f)
            print(f"Valid Scores = {valid_scores}", file=f)
            print(f"Test Scores = {test_scores}", file=f)

            print(f"Train AUC or R2 = {np.mean(train_scores['auc_or_r2'])} +- {np.std(train_scores['auc_or_r2'])}", file=f)
            print(f"Valid AUC or R2 = {np.mean(valid_scores['auc_or_r2'])} +- {np.std(valid_scores['auc_or_r2'])}", file=f)
            print(f"Test AUC or R2 = {np.round(np.mean(test_scores['auc_or_r2']), 4)} +- {np.round(np.std(test_scores['auc_or_r2']), 4)}", file=f)

            print(f"Train AP or MSE = {np.mean(train_scores['ap_or_mse'])} +- {np.std(train_scores['ap_or_mse'])}", file=f)
            print(f"Valid AP or MSE = {np.mean(valid_scores['ap_or_mse'])} +- {np.std(valid_scores['ap_or_mse'])}", file=f)
            print(f"Test AP or MSE = {np.round(np.mean(test_scores['ap_or_mse']), 4)} +- {np.round(np.std(test_scores['ap_or_mse']), 4)}", file=f)

            print(f"Train Accuracy or MAE = {np.mean(train_scores['accuracy_or_mae'])} +- {np.std(train_scores['accuracy_or_mae'])}", file=f)
            print(f"Valid Accuracy or MAE = {np.mean(valid_scores['accuracy_or_mae'])} +- {np.std(valid_scores['accuracy_or_mae'])}", file=f)
            print(f"Test Accuracy or MAE = {np.round(np.mean(test_scores['accuracy_or_mae']), 4)} +- {np.round(np.std(test_scores['accuracy_or_mae']), 4)}", file=f)

            print(file=f)
            print(f'Eval takes: {np.mean(eval_times)}s +- {np.std(eval_times)}', file=f)
