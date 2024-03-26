# Modified from https://github.com/Turoad/CLRNet/blob/7269e9d1c1c650343b6c7febb8e764be538b1aed/clrnet/models/losses/lineiou_loss.py # noqa: E501

import torch
from torch import Tensor


def line_iou(pred: Tensor,
             target: Tensor,
             img_w: int,
             width: float = 7.5,
             aligned: bool = True,
             delta_y: float = 320 / 71):
    """
    Calculate the line iou value between predictions and targets
    Args:
        pred: Lane predictions, shape: (num_pred, 72)
        target: Ground truth, shape: (num_target, 72)
        img_w: Image width
        width: Extended radius
        aligned: Whether the dimensions of 'pred' and 'target' is aligned.
        delta_y: Coordinate-Y interval.

     Returns:
        Tensor: Labels in one hot tensor of shape :math:`(N, C, *)`,
    """

    with torch.no_grad():
        pred_width = ((pred[:, 2:] - pred[:, :-2]) ** 2 + delta_y ** 2
                      ) ** 0.5 / delta_y * width
        pred_width = torch.cat(
            [pred_width[:, 0:1], pred_width, pred_width[:, -1:]], dim=1
        )

        valid_mask = (target >= 0) & (target < img_w)
        valid_mask = valid_mask[:, 2:] & valid_mask[:, :-2]
        valid_mask = torch.cat(
            [valid_mask[:, 0:1], valid_mask, valid_mask[:, -1:]], dim=1
        )
        target_width = ((target[:, 2:] - target[:, :-2]) ** 2 + delta_y ** 2
                        ) ** 0.5 / delta_y * width
        target_width = torch.cat(
            [target_width[:, 0:1], target_width, target_width[:, -1:]], dim=1
        )
        target_width[~valid_mask] = width

    px1 = pred - pred_width
    px2 = pred + pred_width
    tx1 = target - target_width
    tx2 = target + target_width
    if aligned:
        invalid_mask = target
        ovr = torch.min(px2, tx2) - torch.max(px1, tx1)
        union = torch.max(px2, tx2) - torch.min(px1, tx1)
    else:
        num_pred = pred.shape[0]
        invalid_mask = target.repeat(num_pred, 1, 1)
        ovr = (torch.min(px2[:, None, :], tx2[None, ...]) -
               torch.max(px1[:, None, :], tx1[None, ...]))
        union = (torch.max(px2[:, None, :], tx2[None, ...]) -
                 torch.min(px1[:, None, :], tx1[None, ...]))

    invalid_masks = (invalid_mask < 0) | (invalid_mask >= img_w)
    ovr[invalid_masks] = 0.
    union[invalid_masks] = 0.
    iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)
    return iou


def liou_loss(pred: Tensor, target: Tensor, img_w: int, width: float = 7.5):
    return (1 - line_iou(pred, target, img_w, width)).mean()
