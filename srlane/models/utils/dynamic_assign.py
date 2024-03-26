# Modified from https://github.com/hirotomusiker/CLRerNet/blob/main/libs/core/bbox/assigners/dynamic_topk_assigner.py # noqa: E501


import torch
from torch import Tensor

from srlane.models.losses.lineiou_loss import line_iou


def focal_cost(cls_pred: Tensor,
               gt_labels: Tensor,
               alpha: float = 0.25,
               gamma: float = 2.,
               eps: float = 1e-12):
    """
    Args:
        cls_pred: Predicted classification score with shape (n_query, n_class).
        gt_labels: Label of ground truth with shape (n_gt,).
        alpha: Factor to adjust the weight of background samples.
        gamma: Factor to adjust the weight of simple samples.
        eps: Factor to avoid divide-by-zero.
    Returns:
        Tensor: Classification cost with shape(n_query, n_gt).
    """
    cls_pred = cls_pred.sigmoid()
    neg_cost = -(1 - cls_pred + eps).log() * (1 - alpha) * cls_pred.pow(gamma)
    pos_cost = -(cls_pred + eps).log() * alpha * (1 - cls_pred).pow(gamma)
    cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
    return cls_cost


def dynamic_k_assign(cost: Tensor,
                     pair_wise_ious: Tensor,
                     n_candidate: int = 4):
    """Assign ground truth with priors dynamically.

    Args:
        cost: Assign cost with shape(n_query, n_gt).
        pair_wise_ious: Iou with shape(n_query, n_gt).
        n_candidate: Maximum number of priors that a ground truth can match.
    Returns:
        Tensor: Indices of assigned prior.
        Tensor: Corresponding ground truth indices.
    """
    matching_matrix = torch.zeros_like(cost)
    ious_matrix = pair_wise_ious

    topk_ious, _ = torch.topk(ious_matrix, n_candidate, dim=0)
    dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
    for gt_idx in range(len(dynamic_ks)):
        _, pos_idx = torch.topk(cost[:, gt_idx],
                                k=dynamic_ks[gt_idx].item(),
                                largest=False)
        matching_matrix[pos_idx, gt_idx] = 1.0
    del topk_ious, dynamic_ks

    matched_gt = matching_matrix.sum(1)
    if (matched_gt > 1).sum() > 0:
        _, cost_argmin = torch.min(cost[matched_gt > 1, :], dim=1)
        matching_matrix[matched_gt > 1, :] *= 0.0
        matching_matrix[matched_gt > 1, cost_argmin] = 1.0

    prior_idx = matching_matrix.sum(1).nonzero()
    gt_idx = matching_matrix[prior_idx].argmax(-1)
    return prior_idx.flatten(), gt_idx.flatten()


def assign(predictions: Tensor,
           targets: Tensor,
           img_w: int,
           iou_weight: float = 2.,
           k: int = 4, ):
    """Computes dynamicly matching based on the cost, including cls cost and
    iou cost.

    Args:
        predictions: Predictions with shape: (n_priors, 76).
        targets: Targets with shape: (n_gt, 76).
        img_w: As implied by name.
        iou_weight: Iou weight of the overall cost.
        k:  Maximum number of priors that a ground truth can match.
    return:
        Tensor: Indices of matched prior.
        Tensor: Indices of matched ground truth.
    """
    predictions = predictions.detach().clone()
    targets = targets.detach().clone()

    # classification cost
    cls_score = focal_cost(predictions[:, :2], targets[:, 1].long())
    # iou cost
    iou_score = line_iou(
        predictions[..., 4:], targets[..., 4:], img_w, width=30, aligned=False
    )
    iou_score = iou_score / torch.max(iou_score)

    cost = -iou_score * iou_weight + cls_score
    iou = line_iou(
        predictions[..., 4:], targets[..., 4:], img_w, width=7.5, aligned=False
    )
    iou[iou < 0.] = 0.
    return dynamic_k_assign(cost, iou, n_candidate=k)
