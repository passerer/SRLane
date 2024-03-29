from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .multi_segment_attention import MultiSegmentAttention
from srlane.ops import nms
from srlane.utils.lane import Lane
from srlane.models.losses.focal_loss import FocalLoss
from srlane.models.utils.dynamic_assign import assign
from srlane.models.utils.a3d_sample import sampling_3d
from srlane.models.losses.lineiou_loss import liou_loss
from srlane.models.registry import HEADS


class RefineHead(nn.Module):
    """Refine head.

    Args:
        stage: Refinement stage index.
        num_points: Number of points to describe a lane.
        prior_feat_channels: Input channel.
        in_channel: Input channels.
        fc_hidden_dim: Hidden channels.
        refine_layers: Total number of refinement stage.
        sample_points: Number of points for sampling lane feature.
        num_groups: Number of lane segment groups.
        cfg: Model config.
    """

    def __init__(self,
                 stage: int,
                 num_points: int,
                 prior_feat_channels: int,
                 fc_hidden_dim: int,
                 refine_layers: int,
                 sample_points: int,
                 num_groups: int,
                 cfg=None):
        super(RefineHead, self).__init__()
        self.stage = stage
        self.cfg = cfg
        self.img_w = self.cfg.img_w
        self.img_h = self.cfg.img_h
        self.n_strips = num_points - 1
        self.n_offsets = num_points
        self.sample_points = sample_points
        self.fc_hidden_dim = fc_hidden_dim
        self.num_groups = num_groups
        self.num_level = cfg.n_fpn
        self.last_stage = stage == refine_layers - 1

        self.register_buffer(name="sample_x_indexs", tensor=(
                torch.linspace(0, 1,
                               steps=self.sample_points,
                               dtype=torch.float32) * self.n_strips).long())
        self.register_buffer(name="prior_feat_ys", tensor=torch.flip(
            (1 - self.sample_x_indexs.float() / self.n_strips), dims=[-1]))

        self.prior_feat_channels = prior_feat_channels
        self.z_embeddings = nn.Parameter(torch.zeros(self.sample_points),
                                         requires_grad=True)

        self.gather_fc = nn.Conv1d(sample_points, fc_hidden_dim,
                                   kernel_size=prior_feat_channels,
                                   groups=self.num_groups)
        self.shuffle_fc = nn.Linear(fc_hidden_dim, fc_hidden_dim)
        self.channel_fc = nn.ModuleList()
        self.segment_attn = nn.ModuleList()
        for i in range(1):
            self.segment_attn.append(
                MultiSegmentAttention(fc_hidden_dim, num_groups=num_groups))
            self.channel_fc.append(
                nn.Sequential(nn.Linear(fc_hidden_dim, 2 * fc_hidden_dim),
                              nn.ReLU(),
                              nn.Linear(2 * fc_hidden_dim, fc_hidden_dim)))
        reg_modules = list()
        cls_modules = list()
        for _ in range(1):
            reg_modules += [nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim),
                            nn.ReLU()]
            cls_modules += [nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim),
                            nn.ReLU()]

        self.reg_modules = nn.ModuleList(reg_modules)
        self.cls_modules = nn.ModuleList(cls_modules)
        self.reg_layers = nn.Linear(
            fc_hidden_dim,
            self.n_offsets + 1 + 1)
        self.cls_layers = nn.Linear(fc_hidden_dim, 2)
        self.init_weights()

    def init_weights(self):
        for m in self.cls_layers.parameters():
            nn.init.normal_(m, mean=0., std=1e-3)
        for m in self.reg_layers.parameters():
            nn.init.normal_(m, mean=0., std=1e-3)
        nn.init.normal_(self.z_embeddings, mean=self.cfg.z_mean[self.stage],
                        std=self.cfg.z_std[self.stage])


    def translate_to_linear_weight(self,
                                   ref: Tensor,
                                   num_total: int = 3,
                                   tau: int = 2.0):
        grid = torch.arange(num_total, device=ref.device,
                            dtype=ref.dtype).view(
            *[len(ref.shape) * [1, ] + [-1, ]])
        ref = ref.unsqueeze(-1).clone()
        l2 = (ref - grid).pow(2.0).div(tau).abs().neg()
        weight = torch.softmax(l2, dim=-1)

        return weight  # (1, 36, 3)

    def pool_prior_features(self,
                            batch_features: List[Tensor],
                            num_priors: int,
                            prior_feat_xs: Tensor, ):
        """Pool prior feature from feature map.
        Args:
            batch_features: Input feature maps.
        """
        batch_size = batch_features[0].shape[0]

        prior_feat_xs = prior_feat_xs.view(batch_size, num_priors, -1, 1)
        prior_feat_ys = self.prior_feat_ys.unsqueeze(0).expand(
            batch_size * num_priors,
            self.sample_points).view(
            batch_size, num_priors, -1, 1)

        grid = torch.cat((prior_feat_xs, prior_feat_ys), dim=-1)
        if self.training or not hasattr(self, "z_weight"):
            z_weight = self.translate_to_linear_weight(self.z_embeddings)
            z_weight = z_weight.view(1, 1, self.sample_points, -1).expand(
                batch_size,
                num_priors,
                self.sample_points,
                self.num_level)
        else:
            z_weight = self.z_weight.view(1, 1, self.sample_points, -1).expand(
                batch_size,
                num_priors,
                self.sample_points,
                self.num_level)

        feature = sampling_3d(grid, z_weight,
                              batch_features)  # (b, n_prior, n_point, c)
        feature = feature.view(batch_size * num_priors, -1,
                               self.prior_feat_channels)
        feature = self.gather_fc(feature).reshape(batch_size, num_priors, -1)
        for i in range(1):
            res_feature, attn = self.segment_attn[i](feature, attn_mask=None)
            feature = feature + self.channel_fc[i](res_feature)
        return feature, attn

    def forward(self, batch_features, priors, pre_feature=None):
        batch_size = batch_features[-1].shape[0]
        num_priors = priors.shape[1]
        prior_feat_xs = (priors[..., 4 + self.sample_x_indexs]).flip(
            dims=[2])  # top to bottom

        batch_prior_features, attn = self.pool_prior_features(
            batch_features, num_priors, prior_feat_xs)

        fc_features = batch_prior_features
        fc_features = fc_features.reshape(batch_size * num_priors,
                                          self.fc_hidden_dim)

        if pre_feature is not None:
            fc_features = fc_features + pre_feature.view(*fc_features.shape)

        cls_features = fc_features
        reg_features = fc_features
        predictions = priors.clone()
        if self.training or self.last_stage:
            for cls_layer in self.cls_modules:
                cls_features = cls_layer(cls_features)
            cls_logits = self.cls_layers(cls_features)
            cls_logits = cls_logits.reshape(
                batch_size, -1, cls_logits.shape[1])  # (B, num_priors, 2)
            predictions[:, :, :2] = cls_logits
        for reg_layer in self.reg_modules:
            reg_features = reg_layer(reg_features)
        reg = self.reg_layers(reg_features)
        reg = reg.reshape(batch_size, -1, reg.shape[1])

        #  predictions[:, :, 2] += reg[:, :, 0]
        # predictions[:, :, 3] = reg[:, :, 1]
        # predictions[..., 4:] += reg[..., 2:]
        predictions[:, :, 2:] += reg

        return predictions, fc_features, attn


@HEADS.register_module
class CascadeRefineHead(nn.Module):
    def __init__(self,
                 num_points: int = 72,
                 prior_feat_channels: int = 64,
                 fc_hidden_dim: int = 64,
                 refine_layers: int = 1,
                 sample_points: int= 36 ,
                 num_groups: int = 6,
                 cfg=None):
        super(CascadeRefineHead, self).__init__()
        self.cfg = cfg
        self.img_w = self.cfg.img_w
        self.img_h = self.cfg.img_h
        self.n_strips = num_points - 1
        self.n_offsets = num_points
        self.sample_points = sample_points
        self.refine_layers = refine_layers
        self.fc_hidden_dim = fc_hidden_dim
        self.num_groups = num_groups
        self.prior_feat_channels = prior_feat_channels

        self.register_buffer(name="prior_ys",
                             tensor=torch.linspace(1, 0, steps=self.n_offsets,
                                                   dtype=torch.float32))

        self.stage_heads = nn.ModuleList()
        for i in range(refine_layers):
            self.stage_heads.append(
                RefineHead(stage=i,
                           num_points=num_points,
                           prior_feat_channels=prior_feat_channels,
                           fc_hidden_dim=fc_hidden_dim,
                           refine_layers=refine_layers,
                           sample_points=sample_points,
                           num_groups=num_groups,
                           cfg=cfg))

        self.cls_criterion = FocalLoss(alpha=0.25, gamma=2.)

    def forward(self, x, **kwargs):
        batch_features = list(x)
        batch_features.reverse()
        priors = kwargs["priors"]
        pre_feature = None
        predictions_lists = []
        attn_lists = []

        # iterative refine
        for stage in range(self.refine_layers):
            predictions, pre_feature, attn = self.stage_heads[stage](
                batch_features, priors,
                pre_feature)
            predictions_lists.append(predictions)
            attn_lists.append(attn)

            if stage != self.refine_layers - 1:
                priors = predictions.clone().detach()

        if self.training:
            output = {"predictions_lists": predictions_lists,
                      "attn_lists": attn_lists}
            return output
        return predictions_lists[-1]

    def loss(self,
             output,
             batch):
        predictions_lists = output["predictions_lists"]
        attn_lists = output["attn_lists"]
        targets = batch["gt_lane"].clone()

        cls_loss = 0
        l1_loss = 0
        iou_loss = 0
        attn_loss = 0

        for stage in range(0, self.refine_layers):
            predictions_list = predictions_lists[stage]
            attn_list = attn_lists[stage]
            for idx, (predictions, target, attn) in enumerate(
                    zip(predictions_list, targets, attn_list)):
                target = target[target[:, 1] == 1]
                if len(target) == 0:
                    cls_target = predictions.new_zeros(
                        predictions.shape[0]).long()
                    cls_pred = predictions[:, :2]
                    cls_loss = cls_loss + self.cls_criterion(
                        cls_pred, cls_target).sum()
                    continue

                predictions = torch.cat((predictions[:, :2],
                                         predictions[:, 2:4] * self.n_strips,
                                         predictions[:, 4:] * self.img_w),
                                        dim=1)

                with torch.no_grad():
                    (matched_row_inds, matched_col_inds) = assign(
                        predictions, target, self.img_w,
                        k=self.cfg.angle_map_size[0])

                attn_loss += MultiSegmentAttention.loss(
                    predictions[:, 4:] / self.img_w,
                    target[matched_col_inds, 4:] / self.img_w,
                    attn[:, matched_row_inds])

                # classification targets
                cls_target = predictions.new_zeros(predictions.shape[0]).long()
                cls_target[matched_row_inds] = 1
                cls_pred = predictions[:, :2]

                # regression targets
                reg_yl = predictions[matched_row_inds, 2:4]
                target_yl = target[matched_col_inds, 2:4].clone()
                with torch.no_grad():
                    reg_start_y = torch.clamp(
                        (reg_yl[:, 0]).round().long(), 0,
                        self.n_strips)
                    target_start_y = target_yl[:, 0].round().long()
                    target_yl[:, 1] -= reg_start_y - target_start_y

                reg_pred = predictions[matched_row_inds, 4:]
                reg_targets = target[matched_col_inds, 4:].clone()

                # Loss calculation
                cls_loss = cls_loss + self.cls_criterion(cls_pred, cls_target).sum(
                ) / target.shape[0]

                l1_loss = l1_loss + F.smooth_l1_loss(reg_yl, target_yl,
                                                       reduction="mean")

                iou_loss = iou_loss + liou_loss(reg_pred, reg_targets,
                                                self.img_w)

        cls_loss /= (len(targets) * self.refine_layers)
        l1_loss /= (len(targets) * self.refine_layers)
        iou_loss /= (len(targets) * self.refine_layers)
        attn_loss /= (len(targets) * self.refine_layers)

        return_value = {"cls_loss": cls_loss,
                        "l1_loss": l1_loss,
                        "iou_loss": iou_loss,
                        "attn_loss": attn_loss}

        return return_value

    def predictions_to_pred(self, predictions, img_meta):
        """
        Convert predictions to internal Lane structure for evaluation.
        """
        prior_ys = self.prior_ys.to(predictions.device)
        prior_ys = prior_ys.double()
        lanes = []

        for lane in predictions:
            lane_xs = lane[4:]  # normalized value
            start = min(max(0, int(round(lane[2].item() * self.n_strips))),
                        self.n_strips)
            length = int(round(lane[3].item()))
            end = start + length - 1
            end = min(end, self.n_strips)
            # extend its prediction until the x is outside the image
            mask = ~((((lane_xs[:start] >= 0.) & (lane_xs[:start] <= 1.)
                       ).cpu().numpy()[::-1].cumprod()[::-1]).astype(bool))
            lane_xs[end + 1:] = -2
            lane_xs[:start][mask] = -2
            lane_ys = prior_ys[(lane_xs >= 0.) & (lane_xs <= 1.)]
            lane_xs = lane_xs[(lane_xs >= 0.) & (lane_xs <= 1.)]
            if len(lane_xs) <= 1:
                continue
            lane_xs = lane_xs.flip(0).double()
            lane_ys = lane_ys.flip(0)

            if "img_cut_height" in img_meta:
                cut_height = img_meta["img_cut_height"]
                ori_img_h = img_meta["img_size"][0]
                lane_ys = (lane_ys * (ori_img_h - cut_height) +
                           cut_height) / ori_img_h
            points = torch.stack(
                (lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)),
                dim=1).squeeze(2)
            lane = Lane(points=points.cpu().numpy(),
                        metadata={})
            lanes.append(lane)
        return lanes

    def get_lanes(self, output, img_metas, as_lanes=True):
        """
        Convert model output to lanes.
        """
        softmax = nn.Softmax(dim=1)

        decoded = []
        img_metas = [item for img_meta in img_metas.data for item in img_meta]
        for predictions, img_meta in zip(output, img_metas):
            # filter out the conf lower than conf threshold
            threshold = self.cfg.test_parameters.conf_threshold
            scores = softmax(predictions[:, :2])[:, 1]
            keep_inds = scores >= threshold
            predictions = predictions[keep_inds]
            scores = scores[keep_inds]

            if predictions.shape[0] == 0:
                decoded.append([])
                continue

            nms_preds = predictions.detach().clone()
            nms_preds[..., 2:4] *= self.n_strips
            nms_preds[..., 3] = nms_preds[..., 2] + nms_preds[..., 3] - 1
            nms_preds[..., 4:] *= self.img_w

            keep, num_to_keep, _ = nms(
                nms_preds,
                scores,
                overlap=self.cfg.test_parameters.nms_thres,
                top_k=self.cfg.max_lanes)
            keep = keep[:num_to_keep]
            predictions = predictions[keep]

            if predictions.shape[0] == 0:
                decoded.append([])
                continue
            predictions[:, 3] = torch.round(predictions[:, 3] * self.n_strips)
            if as_lanes:
                pred = self.predictions_to_pred(predictions, img_meta)
            else:
                pred = predictions
            decoded.append(pred)

        return decoded

    def __repr__(self):
        num_params = sum(map(lambda x: x.numel(), self.parameters()))
        return f"#Params of {self._get_name()}: {num_params / 10 ** 3:<.2f}[K]"
