from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultiSegmentAttention(nn.Module):
    """Multi-Segment Attention (MSA) for lane detection.

    Args:
        embed_dim: Channel dimension.
        num_groups: Number of lane segment groups.
        dropout: dropout ratio.
    """

    def __init__(self,
                 embed_dim: int,
                 num_groups: int = 1,
                 dropout: float = 0.0, ):
        super(MultiSegmentAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_groups = num_groups
        self.dropout = dropout
        if embed_dim % num_groups != 0:
            raise ValueError(f"Embed_dim ({embed_dim}) must be "
                             f"divisible by num_groups ({num_groups})")
        self.head_dim = embed_dim // num_groups
        self.scale = 1 / (self.head_dim ** 0.5)

        self.q_proj = nn.Linear(embed_dim, self.head_dim)
        self.k_proj = nn.Conv1d(embed_dim, embed_dim, kernel_size=1,
                                groups=num_groups)

    def forward(self,
                x: Tensor,
                attn_mask: Optional[Tensor] = None,
                tau: float = 1.0):
        """The forward function of MSA.

        Args:
            x: The input data with shape (B, N, C).
            attn_mask: Attention mask. Defaults to None.
            tau: Temperature in Softmax. Default: 1.0
        Returns:
            Tensor: The updated feature with shape (B, N, C).
            Tensor: The attention map.
        """
        bs, n_q, _ = x.shape
        kv = x.flatten(0, 1).unsqueeze(-1)
        k = self.k_proj(kv)
        q = self.q_proj(x).unsqueeze(1)
        v = x.view(bs, n_q, self.num_groups, -1).permute(0, 2, 1, 3)
        k = k.view(bs, n_q, self.num_groups, -1).permute(0, 2, 3, 1)
        attn_weight = (q @ k) * self.scale
        if attn_mask is not None:
            attn_weight = attn_weight + attn_mask.view(*attn_weight.shape)
        attn_weight = attn_weight.div(tau).softmax(-1)
        context = attn_weight @ v
        context = context.permute(0, 2, 1, 3).contiguous()
        return context.flatten(-2, -1), attn_weight

    @staticmethod
    def loss(pred_lanes: Tensor,
             target_lanes: Tensor,
             pred_attn_weight: Tensor):
        """Loss function.

        Args:
            pred_lanes: Predicted lane xs with shape (n_prior, 72).
            target_lanes: Ground-truth lane xs with shape (n_pos, 72).
            pred_attn_weight: Atten map with shape (groups, n_pos, n_prior).
        Returns:
            Tensor: Cross entropy loss of attention map.
        """
        if len(target_lanes) == 0:
            return 0
        target_lanes = target_lanes.detach().clone()
        target_lanes = target_lanes.flip(-1)  # (n_pos, 72)
        pred_lanes = pred_lanes.clone()
        pred_lanes = pred_lanes.flip(-1)
        groups, n_pos, n_prior = pred_attn_weight.shape
        target_lanes = target_lanes.reshape(n_pos, groups, -1).permute(1, 0, 2)  # (groups, n_pos, 72//groups) # noqa: E501
        pred_lanes = pred_lanes.reshape(n_prior, groups, -1).permute(1, 0, 2)  # (groups, n_prior, 72//groups) # noqa: E501
        valid_mask = (0 <= target_lanes) & (target_lanes < 1)
        dist = ((pred_lanes.unsqueeze(1) - target_lanes.unsqueeze(2)).abs()
                ) * valid_mask.unsqueeze(2)  # (groups, n_pos, n_prior, 72//groups) # noqa: E501
        dist = dist.sum(-1) / (valid_mask.sum(-1).unsqueeze(2) + 1e-6)  # (groups, n_pos, n_prior) # noqa: E501
        _, indices = dist.min(-1)  # (groups, n_pos)
        valid_mask = valid_mask.any(-1)  # (groups, n_pos)
        indices[~valid_mask] = 255
        pred_attn_weight = torch.clamp(pred_attn_weight, 1e-6, 1 - 1e-6)
        loss = F.nll_loss(torch.log(pred_attn_weight).transpose(1, 2),
                          indices.long(),
                          ignore_index=255)
        return loss
