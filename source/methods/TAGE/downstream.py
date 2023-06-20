import torch
import torch.nn.functional as F
from torch import optim, nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

"""
This is an adaptation of the original TAGE code from: https://github.com/divelab/DIG/tree/main/dig/xgraph/TAGE/downstream.py
"""


class MLP(torch.nn.Module):

    def __init__(self, num_layer, emb_dim, hidden_dim, out_dim=1):
        super(MLP, self).__init__()
        self.num_layer = num_layer
        self.layers = nn.ModuleList()
        if num_layer > 1:
            self.layers.append(nn.Linear(emb_dim, hidden_dim))
            for n in range(num_layer - 1):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, out_dim))
        else:
            self.layers.append(nn.Linear(emb_dim, out_dim))

    def forward(self, emb):
        out = self.layers[0](emb)
        for layer in self.layers[1:]:
            out = layer(F.relu(out))
        return out


def train_MLP(embed_model, emd_dim, device, loader, val_loader, save_to=None):
    """
    Train MLP model on top of embedding model for downstream task.
    :param embed_model: base GNN embedding model
    :param device: PyTorch device
    :param loader: Training data loader
    :param val_loader: Validation data loader
    :param save_to: model save path
    :return: trained MLP model
    """
    mlp_model = MLP(2, emd_dim, emd_dim)
    embed_model = embed_model.to(device)
    mlp_model = mlp_model.to(device)

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    lr = 0.001
    weight_decay = 0
    epochs = 100

    optimizer = optim.Adam(mlp_model.parameters(), lr=lr, weight_decay=weight_decay)
    best_roc = 0
    for _ in tqdm(range(epochs)):
        embed_model.eval()
        mlp_model.train()
        for step, batch in enumerate(loader):
            _, embeds, _ = embed_model(batch.to(device))
            embeds = embeds.detach()
            pred = mlp_model(embeds)
            y = batch.y.view(pred.shape).to(torch.float64)
            loss_mat = criterion(pred.double(), y)
            optimizer.zero_grad()
            loss = torch.mean(loss_mat)
            loss.backward()
            optimizer.step()

        mlp_model.eval()
        y_true = []
        y_scores = []

        for step, batch in enumerate(val_loader):
            with torch.no_grad():
                _, embeds, _ = embed_model(batch.to(device))
                embeds = embeds.detach()
                pred = mlp_model(embeds)

            y_true.append(batch.y.view(pred.shape))
            y_scores.append(pred)

        y_true = torch.cat(y_true, dim=0).cpu().numpy()
        y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

        roc_score = roc_auc_score(y_true.flatten(), y_scores.flatten())
        if roc_score > best_roc and save_to:
            best_roc = roc_score
            torch.save(mlp_model.state_dict(), save_to)

    mlp_model.load_state_dict(torch.load(save_to))
    return mlp_model
