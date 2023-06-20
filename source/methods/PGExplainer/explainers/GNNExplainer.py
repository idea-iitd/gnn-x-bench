from math import sqrt

import torch
from torch.optim import Adam

from methods.PGExplainer.explainers.BaseExplainer import BaseExplainer
from methods.PGExplainer.utils.graph import index_edge

"""
This is an adaptation of GNNExplainer code from: https://github.com/LarsHoldijk/RE-ParameterizedExplainerForGraphNeuralNetworks/blob/main/ExplanationEvaluation/explainers/GNNExplainer.py
"""


class GNNExplainer(BaseExplainer):
    """
    A class encaptulating the GNNexplainer (https://arxiv.org/abs/1903.03894).
    
    :param model_to_explain: graph classification model who's predictions we wish to explain.
    :param graphs: the collections of edge_indices representing the graphs
    :param task: str "node" or "graph"
    :param epochs: amount of epochs to train our explainer
    :param lr: learning rate used in the training of the explainer
    :param reg_coefs: reguaization coefficients used in the loss. The first item in the tuple restricts the size of the explainations, the second rescticts the entropy matrix mask.
    
    :function __set_masks__: utility; sets learnable mask on the graphs.
    :function __clear_masks__: utility; rmoves the learnable mask.
    :function _loss: calculates the loss of the explainer
    :function explain: trains the explainer to return the subgraph which explains the classification of the model-to-be-explained.
    """

    def __init__(self, model_to_explain, graphs, task, device, epochs=30, lr=0.003, reg_coefs=(0.05, 1.0)):
        super().__init__(model_to_explain, graphs, None, task)
        self.epochs = epochs
        self.lr = lr
        self.reg_coefs = reg_coefs
        self.device = device

    def _loss(self, masked_pred, original_pred, edge_mask, reg_coefs):
        """
        Returns the loss score based on the given mask.
        :param masked_pred: Prediction based on the current explanation
        :param original_pred: Predicion based on the original graph
        :param edge_mask: Current explanaiton
        :param reg_coefs: regularization coefficients
        :return: loss
        """
        size_reg = reg_coefs[0]
        entropy_reg = reg_coefs[1]

        EPS = 1e-15

        # Regularization losses
        size_loss = torch.sum(edge_mask) * size_reg
        mask_ent_reg = -edge_mask * torch.log(edge_mask + EPS) - (1 - edge_mask) * torch.log(1 - edge_mask + EPS)
        mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)

        # Explanation loss
        cce_loss = torch.nn.functional.cross_entropy(masked_pred, original_pred)

        return cce_loss + size_loss + mask_ent_loss

    def prepare(self, args):
        """Nothing is done to prepare the GNNExplainer, this happens at every index"""
        return

    def explain(self, index):
        """
        Main method to construct the explanation for a given sample. This is done by training a mask such that the masked graph still gives
        the same prediction as the original graph using an optimization approach
        :param index: index of the graph that we wish to explain
        :return: explanation weights
        """
        graph = self.graphs[int(index)]

        # Prepare model for new explanation run
        self.model_to_explain.eval()

        (N, F), E = graph.x.size(), graph.edge_index.size(1)
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)

        with torch.no_grad():
            _, _, logits = self.model_to_explain(graph.to(self.device))
            pred_label = logits.argmax(dim=-1).detach()

        optimizer = Adam([self.edge_mask], lr=self.lr)

        # Start training loop
        for e in range(0, self.epochs):
            optimizer.zero_grad()
            _, _, masked_logits = self.model_to_explain(graph.to(self.device), edge_weight=torch.sigmoid(self.edge_mask).to(self.device))
            loss = self._loss(masked_logits, pred_label, torch.sigmoid(self.edge_mask), self.reg_coefs)
            loss.backward()
            optimizer.step()

        mask = torch.sigmoid(self.edge_mask)
        return mask
