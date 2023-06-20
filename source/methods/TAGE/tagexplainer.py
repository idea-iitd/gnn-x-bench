from typing import Optional
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam
from torch_geometric.data import Data

"""
This is an adaptation of TAGE code from: https://github.com/divelab/DIG/tree/main/dig/xgraph/TAGE/tagexplainer.py
"""

def NCE_loss(zs):
    """
    :param zs: graph embeddings for each view
    :return:
    """
    tau = 0.5
    norm = True
    return NT_Xent(zs[0], zs[1], tau, norm)


def NT_Xent(z1, z2, tau=0.5, norm=True):
    """
    Normalized temperature-scaled cross entropy loss.
    :param z1: embedding of graph1
    :param z2: embedding of graph2
    :param tau:
    :param norm:
    :return:
    """
    batch_size, _ = z1.size()
    sim_matrix = torch.einsum('ik,jk->ij', z1, z2)

    if norm:
        z1_abs = z1.norm(dim=1)
        z2_abs = z2.norm(dim=1)
        sim_matrix = sim_matrix / torch.einsum('i,j->ij', z1_abs, z2_abs)

    sim_matrix = torch.exp(sim_matrix / tau)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()
    return loss


def JSE_loss(zs):
    """
    Jensen-Shannon divergence between two views.
    :param zs: List of tensors of shape [n_views, batch_size, z_dim].
    :return:
    """
    jse = JSE_global_global
    if len(zs) == 2:
        return jse(zs[0], zs[1])
    else:
        raise NotImplementedError('JSE loss for more than 2 views is not implemented yet.')


def JSE_global_global(z1, z2):
    """
    :param z1: embedding of graph1
    :param z2: embedding of graph2
    :return:
    """
    device = z1.device
    num_graphs = z1.shape[0]

    pos_mask = torch.zeros((num_graphs, num_graphs)).to(device)
    neg_mask = torch.ones((num_graphs, num_graphs)).to(device)
    for graphidx in range(num_graphs):
        pos_mask[graphidx][graphidx] = 1.
        neg_mask[graphidx][graphidx] = 0.

    d_prime = torch.matmul(z1, z2.t())

    E_pos = get_expectation(d_prime * pos_mask, positive=True).sum()
    E_pos = E_pos / num_graphs
    E_neg = get_expectation(d_prime * neg_mask, positive=False).sum()
    E_neg = E_neg / (num_graphs * (num_graphs - 1))
    return E_neg - E_pos


def get_expectation(masked_d_prime, positive=True):
    """
    :param masked_d_prime: Tensor of shape [n_graphs, n_graphs] for global_global, tensor of shape [n_nodes, n_graphs] for local_global.
    :param positive: True if the d_prime is masked for positive pairs, False for negative pairs.
    :return:
    """
    log_2 = np.log(2.)
    if positive:
        score = log_2 - F.softplus(-masked_d_prime)
    else:
        score = F.softplus(-masked_d_prime) + masked_d_prime - log_2
    return score


class Explainer(nn.Module):
    """
    The parametric explainer takes node embeddings and condition vector as inputs, and predicts edge importance scores. Constructed as a 2-layer MLP.
    Args:
        embed_dim: Dimension of node embeddings.
        graph_level: Whether to explain a graph-level prediction task or node-level prediction task.
        hidden_dim: Hidden dimension of the MLP in the explainer.
    """

    def __init__(self, embed_dim: int, graph_level: bool, hidden_dim: int = 600):
        super(Explainer, self).__init__()

        self.embed_dims = embed_dim * (2 if graph_level else 3)
        self.cond_dims = embed_dim

        self.emb_linear1 = nn.Sequential(nn.Linear(self.embed_dims, hidden_dim), nn.ReLU())
        self.emb_linear2 = nn.Linear(hidden_dim, 1)

        self.cond_proj = nn.Sequential(nn.Linear(self.cond_dims, self.embed_dims), nn.ReLU())

    def forward(self, embed, cond):
        cond = self.cond_proj(cond)
        out = embed * cond
        out = self.emb_linear1(out)
        out = self.emb_linear2(out)
        return out


class MLPExplainer(torch.nn.Module):
    """
    MLP explainer based on gradient of output w.r.t. input embedding.
    Args:
        mlp_model: The downstream model to be explained.
        device: Torch CUDA device.
    """

    def __init__(self, mlp_model, device):
        super(MLPExplainer, self).__init__()
        self.model = mlp_model.to(device)
        self.device = device

    def forward(self, embeds, mode='explain'):
        """
        :param embeds: graph embeddings
        :param mode:
        :return: probs or grads
        """
        embeds = embeds.detach().to(self.device)
        self.model.eval()
        if mode == 'explain':
            return self.get_grads(embeds)
        elif mode == 'pred':
            return self.get_probs(embeds)
        else:
            raise NotImplementedError

    def get_probs(self, embeds):
        logits = self.model(embeds)
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits)
            probs = torch.cat([1 - probs, probs], 1)
        else:
            probs = F.softmax(logits, dim=-1)
        return probs

    def get_grads(self, embeds):
        optimizer = torch.optim.SGD([embeds.requires_grad_()], lr=0.01)
        optimizer.zero_grad()
        logits = self.model(embeds)
        max_logits, _ = logits.max(dim=-1)
        max_logits.sum().backward()
        grads = embeds.grad
        grads = grads / torch.abs(grads).mean()
        return F.relu(grads)


class TAGExplainer(nn.Module):
    """
    The TAGExplainer that performs 2-stage explanations. Includes training and inference.
    Args:
        model: GNN model to be explained.
        embed_dim: Dimension of node embeddings.
        device: Torch CUDA device.
        coff_size, coff_ent: Hyper-parameters for mask regularization.
        grad_scale: The scale parameter for generating random condition vectors.
        loss_type: Type of the contrastive loss.
    """

    def __init__(self, model, embed_dim: int, device, coff_size: float = 0.01, coff_ent: float = 5e-4, grad_scale: float = 0.25,
                 loss_type='NCE', t0: float = 5.0, t1: float = 1.0, num_hops: Optional[int] = None):

        super(TAGExplainer, self).__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.model = model.to(device)
        self.explainer = Explainer(embed_dim, graph_level=True).to(device)

        # objective parameters for PGExplainer
        self.grad_scale = grad_scale
        self.coff_size = coff_size
        self.coff_ent = coff_ent
        self.t0 = t0
        self.t1 = t1
        self.loss_type = loss_type

    def __loss__(self, embed: Tensor, pruned_embed: Tensor, condition: Tensor, edge_mask: Tensor, **kwargs):
        """
        Calculate contrastive loss and size loss
        :param embed: graph embeddings
        :param pruned_embed: new graph embeddings
        :param condition: condition vector
        :param edge_mask: edge weights
        :param kwargs:
        :return:
        """
        if self.loss_type == 'NCE':
            contrast_loss = NCE_loss([condition * embed, condition * pruned_embed])
        elif self.loss_type == 'JSE':
            contrast_loss = JSE_loss([condition * embed, condition * pruned_embed])
        else:
            raise NotImplementedError(f'Loss type {self.loss_type} not implemented.')

        size_loss = self.coff_size * torch.mean(edge_mask)
        edge_mask = edge_mask * 0.99 + 0.005  # avoid nan
        mask_ent = - edge_mask * torch.log(edge_mask) - (1 - edge_mask) * torch.log(1 - edge_mask)
        mask_ent = self.coff_ent * torch.mean(mask_ent)

        loss = contrast_loss + size_loss + mask_ent
        return loss

    def __rand_cond__(self, n_sample, max_val=None):
        lap = torch.distributions.laplace.Laplace(loc=0, scale=self.grad_scale)
        cond = F.relu(lap.sample([n_sample, self.embed_dim])).to(self.device)
        if max_val is not None:
            cond = torch.clip(cond, max=max_val)
        return cond

    def concrete_sample(self, log_alpha: Tensor, beta: float = 1.0, training: bool = True):
        """ Sample from the instantiation of concrete distribution when training """
        if training:
            random_noise = torch.rand(log_alpha.shape)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (random_noise.to(log_alpha.device) + log_alpha) / beta
            gate_inputs = gate_inputs.sigmoid()
        else:
            gate_inputs = log_alpha.sigmoid()  # to make sure the values are between 0 and 1

        return gate_inputs

    def explain(self, data: Data, embed: Tensor, condition: Tensor = None, tmp: float = 1.0, training: bool = False):
        """ Explain the GNN behavior for graph with explanation network
        :param data: Pytorch Geometric Data object (Batch)
        :param embed: Node embedding matrix with shape :obj:`[num_nodes, dim_embedding]`
        :param condition: pre-computed condition vector with MLP
        :param tmp: The temperature parameter fed to the sample procedure
        :param training: Whether in training procedure or not
        :return:
        """
        node_size = embed.shape[0]
        col, row = data.edge_index
        f1 = embed[col]
        f2 = embed[row]
        f12self = torch.cat([f1, f2], dim=-1)

        # using the node embedding to calculate the edge weight
        condition = self.__rand_cond__(1) if condition is None else condition
        h = self.explainer(f12self.to(self.device), condition.to(self.device))

        mask_val = h.reshape(-1)
        values = self.concrete_sample(mask_val, beta=tmp, training=training)
        mask_sparse = torch.sparse_coo_tensor(data.edge_index, values, (node_size, node_size))
        mask_sigmoid = mask_sparse.to_dense()

        # set the symmetric edge weights
        sym_mask = (mask_sigmoid + mask_sigmoid.transpose(0, 1)) / 2
        edge_mask = sym_mask[col, row]

        # the model prediction with edge mask
        node_embed, graph_embed, out = self.model(data, edge_weight=edge_mask)

        return node_embed, graph_embed, edge_mask

    def train_explainer_graph(self, loader, lr=0.001, epochs=10):
        """ training the explanation network by gradient descent(GD) using Adam optimizer """
        optimizer = Adam(self.explainer.parameters(), lr=lr)
        for epoch in range(epochs):
            tmp = float(self.t0 * np.power(self.t1 / self.t0, epoch / epochs))
            self.model.eval()
            self.explainer.train()
            pbar = tqdm(loader)
            for data in pbar:
                optimizer.zero_grad()
                data = data.to(self.device)
                node_embed, graph_embed, out = self.model(data)
                cond = self.__rand_cond__(1)
                pruned_node_embed, pruned_graph_embed, mask = self.explain(data, embed=node_embed, condition=cond, tmp=tmp, training=True)
                loss = self.__loss__(graph_embed, pruned_graph_embed, cond, mask)
                loss.backward()
                optimizer.step()

    def __edge_mask_to_node__(self, data, edge_mask, top_k):
        threshold = float(edge_mask.reshape(-1).sort(descending=True).values[min(top_k, edge_mask.shape[0] - 1)])
        hard_mask = (edge_mask > threshold).cpu()
        edge_idx_list = torch.where(hard_mask == 1)[0]

        selected_nodes = []
        edge_index = data.edge_index.cpu().numpy()
        for edge_idx in edge_idx_list:
            selected_nodes += [edge_index[0][edge_idx], edge_index[1][edge_idx]]
        selected_nodes = list(set(selected_nodes))
        maskout_nodes = [node for node in range(data.x.shape[0]) if node not in selected_nodes]

        node_mask = torch.zeros(data.num_nodes).type(torch.float32).to(self.device)
        node_mask[maskout_nodes] = 1.0
        return node_mask

    def forward(self, data: Data, mlp_explainer: nn.Module):
        """ explain the GNN behavior for graph.
        Args:
            data: Pytorch Geometric Data object (Batch)
            mlp_explainer: The explanation network for downstream task
        """
        self.model.eval()
        mlp_explainer = mlp_explainer.to(self.device).eval()
        data = data.to(self.device)
        node_embed, graph_embed, out = self.model(data)
        grads = mlp_explainer(graph_embed, mode='explain')
        _, _, explanation = self.explain(data, embed=node_embed, condition=grads, tmp=1.0, training=False)
        return explanation
