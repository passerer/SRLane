import math
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from srlane.models.registry import HEADS
from srlane.models.losses.seg_loss import SegLoss


@HEADS.register_module
class LocalAngleHead(nn.Module):
    """Local angle prediction head.

    Args:
        num_points: Number of lane points.
        in_channel: Input channels.
        cfg: Model config.
    """

    def __init__(self,
                 num_points: int = 72,
                 in_channel: int = 64,
                 cfg=None,
                 ):
        super(LocalAngleHead, self).__init__()
        self.n_offsets = num_points
        self.cfg = cfg
        self.img_w = cfg.img_w
        self.img_h = cfg.img_h
        self.aux_seg = self.cfg.get("seg_loss_weight", 0.) > 0.
        self.feat_h, self.feat_w = self.cfg.angle_map_size
        # Cartesian coordinates
        self.register_buffer(name="prior_ys",
                             tensor=torch.linspace(0, self.feat_h,
                                                   steps=self.n_offsets,
                                                   dtype=torch.float32))
        grid_y, grid_x = torch.meshgrid(torch.arange(self.feat_h - 0.5, 0,
                                                     -1, dtype=torch.float32),
                                        torch.arange(0.5, self.feat_w,
                                                     1, dtype=torch.float32),
                                        indexing="ij")
        grid = torch.stack((grid_x, grid_y), 0)
        grid.unsqueeze_(0)  # (1, 2, h, w)
        self.register_buffer(name="grid", tensor=grid)

        self.angle_conv = nn.ModuleList()
        for _ in range(self.cfg.n_fpn):
            self.angle_conv.append(nn.Conv2d(in_channel, 1,
                                             1, 1, 0, bias=False))

        if self.aux_seg:
            num_classes = self.cfg.max_lanes + 1
            self.seg_conv = nn.ModuleList()
            for _ in range(self.cfg.n_fpn):
                self.seg_conv.append(nn.Conv2d(in_channel, num_classes,
                                               1, 1, 0))
            self.seg_criterion = SegLoss(num_classes=num_classes)
        self.init_weights()

    def init_weights(self):
        for m in self.angle_conv.parameters():
            nn.init.normal_(m, 0., 1e-3)

    def forward(self,
                feats: List[Tensor], ):
        """This method performs the forward propagation process.

        Args:
        - feats: List of feature maps.

        Returns:
        - Tensor: Lane proposals.
        - Optional[List[Tensor]]: predicted angle map, used for training.
        """
        theta_list = []
        # In testing mode, only the deepest feature is used.
        if not self.training:
            feats = feats[-1:]
        for i, feat in enumerate(feats, 1):
            theta = self.angle_conv[len(feats) - i](feat).sigmoid()
            theta_list.append(theta)
        if self.aux_seg:
            seg_list = []
            for i, feat in enumerate(feats, 1):
                seg = self.seg_conv[len(feats) - i](feat)
                seg_list.append(seg)
        angle = F.interpolate(theta_list[-1],
                              size=[self.feat_h, self.feat_w],
                              mode="bilinear",
                              align_corners=True).squeeze(1)
        angle = angle.detach()
        # Remove excessively tilted angles, optional
        angle.clamp_(min=0.05, max=0.95)
        # Build lane proposals
        k = (angle * math.pi).tan()
        bs, h, w = angle.shape
        grid = self.grid
        ws = ((self.prior_ys.view(1, 1, self.n_offsets)
               - grid[:, 1].view(1, h * w, 1)) / k.view(bs, h * w, 1)
              + grid[:, 0].view(1, h * w, 1))  # (bs, h*w, n_offsets)
        ws = ws / w
        valid_mask = (0 <= ws) & (ws < 1)
        _, indices = valid_mask.max(-1)
        start_y = indices / (self.n_offsets - 1)  # (bs, h*w)
        priors = ws.new_zeros(
            (bs, h * w, 2 + 2 + self.n_offsets), device=ws.device)
        priors[..., 2] = start_y
        priors[..., 4:] = ws

        return dict(priors=priors,
                    pred_angle=[theta.squeeze(1) for theta in theta_list]
                    if self.training else None,
                    pred_seg=seg_list
                    if (self.training and self.aux_seg) else None)

    def loss(self,
             pred_angle: List[Tensor],
             pred_seg: List[Tensor],
             gt_angle: List[Tensor],
             gt_seg: List[Tensor],
             loss_weight: Tuple[float] = [0.2, 0.2, 1.],
             ignore_value: float = 0.,
             **ignore_kwargs):
        """ L1 loss for local angle estimation over multi-level features.

        Args:
        - pred_angle: List of estimated angle maps.
        - gt_angle: List of target angle maps.
        - loss_weight: Loss weights of each map.
        - ignore_value: Placeholder value for non-target.

        Returns:
        - Tensor: The calculated angle loss.
        """
        angle_loss = 0
        for pred, target, weight in zip(pred_angle, gt_angle, loss_weight):
            valid_mask = target > ignore_value
            angle_loss = (angle_loss
                          + ((pred - target).abs() * valid_mask).sum()
                          / (valid_mask.sum() + 1e-4)) * weight
        if self.aux_seg:
            seg_loss = 0
            for pred, target, weight in zip(pred_seg, gt_seg, loss_weight):
                seg_loss = seg_loss + self.seg_criterion(pred, target) * weight
            return {"angle_loss": angle_loss,
                    "seg_loss": seg_loss, }

        return {"angle_loss": angle_loss}

    def __repr__(self):
        num_params = sum(map(lambda x: x.numel(), self.parameters()))
        return f"#Params of {self._get_name()}: {num_params / 10 ** 3:<.2f}[K]"
