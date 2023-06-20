import os.path

import torch
import torch_geometric as ptgeom
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from methods.PGExplainer.explainers.BaseExplainer import BaseExplainer
from methods.PGExplainer.utils.graph import index_edge
import torch.nn.functional as F
import time

"""
This is an adaptation of PGExplainer code from: https://github.com/LarsHoldijk/RE-ParameterizedExplainerForGraphNeuralNetworks/blob/main/ExplanationEvaluation/explainers/PGExplainer.py
"""


class PGExplainer(BaseExplainer):
    """
    A class encaptulating the PGExplainer (https://arxiv.org/abs/2011.04573).
    
    :param model_to_explain: graph classification model who's predictions we wish to explain.
    :param graphs: the collections of edge_indices representing the graphs.
    :param features: the collcection of features for each node in the graphs.
    :param task: str "node" or "graph".
    :param epochs: amount of epochs to train our explainer.
    :param lr: learning rate used in the training of the explainer.
    :param temp: the temperture parameters dictacting how we sample our random graphs.
    :param reg_coefs: reguaization coefficients used in the loss. The first item in the tuple restricts the size of the explainations, the second rescticts the entropy matrix mask.
    :params sample_bias: the bias we add when sampling random graphs.
    
    :function _create_explainer_input: utility;
    :function _sample_graph: utility; sample an explanatory subgraph.
    :function _loss: calculate the loss of the explainer during training.
    :function train: train the explainer
    :function explain: search for the subgraph which contributes most to the clasification decision of the model-to-be-explained.
    """

    def __init__(self, model_to_explain, graphs, embeds, task, lr=0.003, temp=(5.0, 2.0), reg_coefs=(0.05, 1.0), sample_bias=0, device='cpu', save_folder=None, args=None):
        super().__init__(model_to_explain, graphs, embeds, task)

        self.epochs = args.epochs
        self.lr = lr
        self.temp = temp
        self.reg_coefs = reg_coefs
        self.sample_bias = sample_bias
        self.device = device

        if self.type == "graph":
            self.expl_embedding = self.model_to_explain.dim * 2
        else:
            self.expl_embedding = self.model_to_explain.dim * 3

        self.save_folder = save_folder
        self.args = args

    def _create_explainer_input(self, pair, embeds):
        """
        Given the embedding of the sample by the model that we wish to explain, this method construct the input to the mlp explainer model.
        Depending on if the task is to explain a graph or a sample, this is done by either concatenating two or three embeddings.
        :param pair: edge pair
        :param embeds: embedding of all nodes in the graph
        :return: concatenated embedding
        """
        rows = pair[0]
        cols = pair[1]
        row_embeds = embeds[rows]
        col_embeds = embeds[cols]
        input_expl = torch.cat([row_embeds, col_embeds], 1)
        return input_expl

    def _sample_graph(self, sampling_weights, temperature=1.0, bias=0.0, training=True):
        """
        Implementation of the re-parameterization trick to obtain a sample graph while maintaining the possibility to backprop.
        :param sampling_weights: Weights provided by the mlp
        :param temperature: annealing temperature to make the procedure more deterministic
        :param bias: Bias on the weights to make sampling less deterministic
        :param training: If set to false, the sampling will be entirely deterministic
        :return: sample graph
        """
        if training:
            bias = bias + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(sampling_weights.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = (gate_inputs.to(sampling_weights.device) + sampling_weights) / temperature
            graph = torch.sigmoid(gate_inputs)
        else:
            graph = torch.sigmoid(sampling_weights)
        return graph

    def _loss(self, masked_outs, original_outs, mask, reg_coefs):
        """
        Returns the loss score based on the given mask.
        :param masked_outs: Prediction based on the current explanation
        :param original_outs: Prediction based on the original graph
        :param mask: Current explanation
        :param reg_coefs: regularization coefficients
        :return: loss
        """
        size_reg = reg_coefs[0]
        entropy_reg = reg_coefs[1]

        original_labels = torch.argmax(original_outs).unsqueeze(0)

        # Regularization losses
        if size_reg > 0:
            size_loss = torch.sum(mask) * size_reg
        else:
            size_loss = 0
        if entropy_reg > 0:
            EPS = 1e-15
            mask_ent_reg = -mask * torch.log(mask + EPS) - (1 - mask) * torch.log(1 - mask + EPS)
            mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)
        else:
            mask_ent_loss = 0

        # Explanation loss
        if self.args.method == 'classification':
            cce_loss = torch.nn.functional.cross_entropy(masked_outs, original_labels)
            total_loss = cce_loss + size_loss + mask_ent_loss
        else:
            mse_loss = F.mse_loss(masked_outs.flatten(), original_labels.flatten())
            total_loss = mse_loss + size_loss + mask_ent_loss

        return total_loss

    def prepare(self, train_indices=None, val_indices=None, start_training=True):
        """
        Before we can use the explainer we first need to train it. This is done here.
        :param train_indices: Indices over which we wish to train.
        :param val_indices: Indices over which we wish to validate.
        :param start_training: If set to false, the explainer will not be trained.
        """
        # Creation of the explainer_model is done here to make sure that the seed is set
        self.explainer_model = nn.Sequential(
            nn.Linear(self.expl_embedding, self.args.hidden_units),
            nn.ReLU(),
            nn.Linear(self.args.hidden_units, 1),
        ).to(self.device)

        if train_indices is None:
            train_indices = range(0, self.graphs.size(0))
            val_indices = range(0, self.graphs.size(0))

        if start_training:
            self.train(train_indices=train_indices, val_indices=val_indices)

    def train(self, train_indices=None, val_indices=None):
        """
        Main method to train the model
        :param train_indices: Indices over which we wish to train.
        :param val_indices: Indices over which we wish to validate.
        :return:
        """
        # Create optimizer and temperature schedule
        optimizer = Adam(self.explainer_model.parameters(), lr=self.lr)
        temp_schedule = lambda e: self.temp[0] * ((self.temp[1] / self.temp[0]) ** (e / self.epochs))
        best_model_path = self.args.best_explainer_model_path

        # Start training loop
        best_val_loss = float('inf')
        cur_patience = 0
        patience = int(self.epochs / 5)
        for e in tqdm(range(self.epochs)):
            t = temp_schedule(e)

            # train
            self.explainer_model.train()
            train_loader = DataLoader(train_indices, batch_size=self.args.batch_size)

            for batch_indices in train_loader:
                optimizer.zero_grad()
                loss = torch.FloatTensor([0]).detach().to(self.device)

                for n in batch_indices:
                    n = int(n)
                    graph = self.graphs[n].detach().to(self.device)
                    embeds = self.embeds[n].detach().to(self.device)

                    # Sample possible explanation
                    input_expl = self._create_explainer_input(graph.edge_index, embeds).unsqueeze(0)
                    sampling_weights = self.explainer_model(input_expl)
                    mask = self._sample_graph(sampling_weights, t, bias=self.sample_bias).squeeze()

                    with torch.no_grad():
                        _, _, masked_outs = self.model_to_explain(graph, edge_weight=mask)
                        _, _, original_outs = self.model_to_explain(graph)

                    id_loss = self._loss(masked_outs, original_outs, mask, self.reg_coefs)
                    loss += id_loss

                loss.backward()
                optimizer.step()

            with torch.no_grad():
                self.explainer_model.eval()
                val_loss = 0
                for n in val_indices:
                    graph = self.graphs[n].clone().detach()
                    explanation = self.explain(n)

                    _, _, masked_outs = self.model_to_explain(graph.to(self.device), edge_weight=explanation.to(self.device))
                    _, _, original_outs = self.model_to_explain(graph.to(self.device))

                    id_loss = self._loss(masked_outs, original_outs, explanation, self.reg_coefs)
                    val_loss += id_loss

                if val_loss < best_val_loss:
                    cur_patience = 0
                    best_val_loss = val_loss
                    torch.save(self.explainer_model.state_dict(), best_model_path)
                else:
                    cur_patience += 1
                    if cur_patience >= patience:
                        break

        self.explainer_model.load_state_dict(torch.load(best_model_path, map_location=self.device))

    @torch.no_grad()
    def explain(self, index):
        """
        Given the index of a node/graph this method returns its explanation. This only gives sensible results if the prepare method has already been called.
        :param index: index of the node/graph that we wish to explain
        :return: explanation graph and edge weights
        """
        self.explainer_model.eval()

        index = int(index)
        graph = self.graphs[index].clone().detach()
        embeds = self.embeds[index].clone().detach()

        # Use explainer mlp to get an explanation
        input_expl = self._create_explainer_input(graph.edge_index, embeds).unsqueeze(dim=0)
        sampling_weights = self.explainer_model(input_expl)
        mask = self._sample_graph(sampling_weights, training=False).squeeze()
        return mask

    @torch.no_grad()
    def explain_graph(self, graph):
        """
        Given graph, this method returns its explanation. This only gives sensible results if the prepare method has already been called.
        :param graph: graph that we wish to explain
        :return: explanation graph and edge weights
        """
        self.explainer_model.eval()
        self.model_to_explain.eval()

        graph = graph.clone().detach()
        embeds, _, _ = self.model_to_explain(graph)
        embeds = embeds.clone().detach()

        # Use explainer mlp to get an explanation
        input_expl = self._create_explainer_input(graph.edge_index, embeds).unsqueeze(dim=0)
        sampling_weights = self.explainer_model(input_expl)
        mask = self._sample_graph(sampling_weights, training=False).squeeze()
        return mask
