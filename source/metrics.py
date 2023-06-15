from torchmetrics.functional import accuracy as torch_accuracy
from torchmetrics.functional import auroc as torch_auc
from torchmetrics.functional import average_precision as torch_ap
from torchmetrics.functional import r2_score as torch_r2_score
from torchmetrics.functional import mean_absolute_error as torch_mae
from torchmetrics.functional import mean_squared_error as torch_mse

import torch
import numpy as np

from torch_geometric.data import Data
from torch_geometric.transforms import RemoveIsolatedNodes, ToUndirected


def auc(ground, pred):
    return torch_auc(pred, ground, num_classes=pred.shape[1]).item()


def ap(ground, pred):
    return torch_ap(pred, ground, num_classes=pred.shape[1]).item()


def accuracy(ground, pred):
    return torch_accuracy(pred, ground, num_classes=pred.shape[1]).item()


def r_squared(ground, pred):
    return torch_r2_score(pred, ground).item()


def mse(ground, pred):
    return torch_mse(pred, ground).item()


def mae(ground, pred):
    return torch_mae(pred, ground).item()


def prediction_similarity(original_predictions, explanation_predictions, metric='correlation'):
    """
    Calculate similarity between two predictions
    :param original_predictions:
    :param explanation_predictions:
    :param metric: sufficiency or necessity
    :return: similarity score of two prediction distribution
    """
    if original_predictions.sum() != original_predictions.shape[0]:
        original_predictions = torch.softmax(original_predictions, axis=1)
        explanation_predictions = torch.softmax(explanation_predictions, axis=1)

    if metric == 'sufficiency':
        return ((original_predictions.argmax(axis=1) == explanation_predictions.argmax(axis=1)).sum() / original_predictions.shape[0]).item()
    elif metric == 'necessity':
        return ((original_predictions.argmax(axis=1) != explanation_predictions.argmax(axis=1)).sum() / original_predictions.shape[0]).item()
    else:
        raise NotImplementedError


def faithfulness(gnn_model, original_graphs, explanations, k, metric_names, device):
    """
    Calculates the faithfulness of explanations on gnn model, under continuous explanations.
    :param gnn_model: PyTorch Geometric GNN model
    :param original_graphs: PyTorch Geometric Data, original graphs that gnn_model has been trained
    :param explanations: PyTorch Geometric Data where edge_weight is assigned as explanations
    :param k: selecting top k edges as explanations
    :param metric_names: list of metrics that will be checked
    :param device: device to run the model
    :return: faithfulness degree of explanations on gnn model
    """

    assert len(explanations) == len(original_graphs)

    explanations_out, original_graphs_out = [], []

    for i in range(len(explanations)):
        if explanations[i].edge_index.shape[1] > 0 and not explanations[i].edge_weight.sum().isnan().item():
            directed_edge_weight = explanations[i].edge_weight[explanations[i].edge_index[0] <= explanations[i].edge_index[1]]
            directed_edge_index = explanations[i].edge_index[:, explanations[i].edge_index[0] <= explanations[i].edge_index[1]]
            threshold = directed_edge_weight.topk(min(k, directed_edge_weight.shape[0]))[0][-1]
            idx = directed_edge_weight >= threshold
            directed_edge_index = directed_edge_index[:, idx]

            new_data = Data(
                edge_index=directed_edge_index.clone(),
                x=explanations[i].x.clone(),
            )
            # remove isolated nodes
            new_data = ToUndirected()(new_data)
            new_data = RemoveIsolatedNodes()(new_data)

            with torch.no_grad():
                _, _, explanations_out_i = gnn_model(new_data.to(device))
                _, _, original_graphs_out_i = gnn_model(original_graphs[i].to(device))

            explanations_out.append(explanations_out_i)
            original_graphs_out.append(original_graphs_out_i)
    explanations_out = torch.vstack(explanations_out)
    original_graphs_out = torch.vstack(original_graphs_out)

    faithfulness_scores = []
    for metric in metric_names:
        faithfulness_scores.append(prediction_similarity(original_graphs_out, explanations_out, metric))

    return faithfulness_scores


def similarity_of_explanations(original_matrix_elements, noisy_matrix_elements, top_k=10, metric='jaccard'):
    """
    Calculate similarity between two explanations
    :param original_matrix_elements:
    :param noisy_matrix_elements:
    :param top_k: top k edges that will be checked
    :param metric: jaccard
    :return: similarity score of two matrix from explanations
    """
    if original_matrix_elements.sum() == 0 or noisy_matrix_elements.sum() == 0:
        return 0
    else:
        if metric == 'jaccard':
            # checks top_k edges that is available in original matrix and also in noisy matrix
            k = min(top_k, original_matrix_elements.shape[0])
            idx1 = torch.topk(original_matrix_elements, k)[1]
            idx2 = torch.topk(noisy_matrix_elements, k)[1]
            mask = torch.logical_not(torch.isin(idx1, idx2))
            diff = torch.masked_select(idx1, mask)
            score = 1 - len(diff) / k
            return score
        else:
            raise NotImplementedError


def robustness(explanations, explanations_under_noise, top_k, metric_names):
    """
    Calculates the robustness of explanations under noise
    :param explanations: PyTorch Geometric Data where edge_weight is assigned as explanations
    :param explanations_under_noise: PyTorch Geometric Data where edge_weight is assigned as explanations under noise
    :param top_k: top k edges that will be checked
    :param metric_names: list of metrics that will be checked
    :return:
    """
    assert len(explanations) == len(explanations_under_noise)

    scores = {metric_name: [] for metric_name in metric_names}
    for i in range(len(explanations)):
        if explanations[i].edge_index.shape[1] > 0 and not explanations[i].edge_weight.sum().isnan().item():
            edge_index = explanations[i].edge_index.t().detach().cpu()
            edge_index_under_noise = explanations_under_noise[i].edge_index.t().detach().cpu()
            edge_weight = explanations[i].edge_weight.detach().cpu()
            edge_weight_under_noise = explanations_under_noise[i].edge_weight.detach().cpu()

            idx = edge_index[:, 0] <= edge_index[:, 1]
            idx_under_noise = edge_index_under_noise[:, 0] <= edge_index_under_noise[:, 1]

            edge_index = edge_index[idx]
            edge_index_under_noise = edge_index_under_noise[idx_under_noise]
            edge_weight = edge_weight[idx]
            edge_weight_under_noise = edge_weight_under_noise[idx_under_noise]

            # Convert the edge indexes to sets for efficient comparison
            set_edge_index_1 = set([tuple(edge.tolist()) for edge in edge_index])
            set_edge_index_2 = set([tuple(edge.tolist()) for edge in edge_index_under_noise])

            # Find the edges that exist in both edge_index_1 and edge_index_2
            common_edges = [edge for edge in set_edge_index_1 if edge in set_edge_index_2]

            # Find the indices of common edges in edge_index_1
            common_edge_indices = [torch.where((edge_index_under_noise == torch.tensor(edge)).sum(dim=1) == 2)[0] for edge in common_edges]
            common_edge_indices = torch.cat(common_edge_indices, dim=0).sort()[0]

            explanations_elements = edge_weight.detach().cpu()
            explanations_under_noise_elements = edge_weight_under_noise[common_edge_indices].detach().cpu()

            for metric_name in metric_names:
                score = similarity_of_explanations(explanations_elements, explanations_under_noise_elements, top_k, metric_name)
                if not np.isnan(score):
                    scores[metric_name].append(score)
    return [sum(scores[metric_name]) / len(scores[metric_name]) for metric_name in metric_names]


def faithfulness_with_removal(gnn_model, original_graphs, explanations, k, metric_names, device):
    """
    Calculate faithfulness of explanations by removing top k edges from explanations
    :param gnn_model: PyTorch Geometric GNN model
    :param original_graphs: PyTorch Geometric Data, original graphs that gnn_model has been trained
    :param explanations: PyTorch Geometric Data where edge_weight is assigned as explanations
    :param k: removing top k edges from the explanations
    :param metric_names: list of metrics that will be checked
    :param device: device to run the model
    :return: faithfulness degree of explanations on gnn model
    """
    assert len(explanations) == len(original_graphs)

    explanations_out, original_graphs_out = [], []

    for i in range(len(explanations)):
        if explanations[i].edge_index.shape[1] > 0 and not explanations[i].edge_weight.sum().isnan().item():
            directed_edge_weight = explanations[i].edge_weight[explanations[i].edge_index[0] <= explanations[i].edge_index[1]]
            directed_edge_index = explanations[i].edge_index[:, explanations[i].edge_index[0] <= explanations[i].edge_index[1]]

            threshold = directed_edge_weight.topk(min(k, directed_edge_weight.shape[0]))[0][-1]
            idx = directed_edge_weight < threshold
            directed_edge_index = directed_edge_index[:, idx]

            new_data = Data(
                edge_index=directed_edge_index.clone(),
                x=explanations[i].x.clone(),
            )
            # remove isolated nodes
            new_data = ToUndirected()(new_data)
            new_data = RemoveIsolatedNodes()(new_data)

            if new_data.edge_index.shape[1] > 0:
                _, _, explanations_out_i = gnn_model(new_data.to(device))
                _, _, original_graphs_out_i = gnn_model(original_graphs[i].to(device))

                explanations_out.append(explanations_out_i)
                original_graphs_out.append(original_graphs_out_i)
    explanations_out = torch.vstack(explanations_out)
    original_graphs_out = torch.vstack(original_graphs_out)

    faithfulness_scores = []
    for metric in metric_names:
        faithfulness_scores.append(prediction_similarity(original_graphs_out, explanations_out, metric))

    return faithfulness_scores
