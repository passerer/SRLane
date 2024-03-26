import torch.nn as nn

from srlane.models.registry import NETS
from srlane.models.registry import build_backbones, build_head, build_neck


@NETS.register_module
class TwoStageDetector(nn.Module):
    """Base class for two-stage detector.
    Usually includes backbone, neck, rpn head and roi head.

    Args:
        cfg: Model config.
    """
    def __init__(self, cfg):
        super(TwoStageDetector, self).__init__()
        self.cfg = cfg
        self.backbone = build_backbones(cfg)
        self.neck = build_neck(cfg)
        self.rpn_head = build_head(cfg.rpn_head, cfg)
        self.roi_head = build_head(cfg.roi_head, cfg)

    def extract_feat(self, batch):
        feat = self.backbone(batch["img"]
                             if isinstance(batch, dict) else batch)
        feat = self.neck(feat)
        return feat

    def _forward_test(self, batch):
        feat = self.extract_feat(batch)
        rpn_result_dict = self.rpn_head(feat)
        return self.roi_head(feat, **rpn_result_dict)

    def _forward_train(self, batch):
        feat = self.extract_feat(batch)
        loss_dic = dict()

        rpn_result_dict = self.rpn_head(feat)
        rpn_loss = self.rpn_head.loss(**rpn_result_dict, **batch)
        loss_dic.update(rpn_loss)
        roi_result_dict = self.roi_head(feat, **rpn_result_dict)
        roi_loss = self.roi_head.loss(roi_result_dict, batch=batch)
        loss_dic.update(roi_loss)

        for loss_k, loss_v in loss_dic.items():
            loss_dic[loss_k] = loss_v * self.cfg.get(f"{loss_k}_weight", 1.)
        all_loss = sum(loss_dic.values())
        loss_dic["loss"] = all_loss

        return {"loss": all_loss,
                "loss_status": loss_dic}

    def forward(self, batch):
        if self.training:
            return self._forward_train(batch)
        return self._forward_test(batch)

    def __repr__(self):
        separator_info = "======== Param. Info. ========"
        num_params = sum(map(lambda x: x.numel(), self.parameters()))
        info = f"#Params of {self._get_name()}: "
        info += f"{num_params / 10 ** 6:<.2f}[M]"
        return '\n'.join([separator_info, repr(self.backbone), repr(self.neck),
                          repr(self.rpn_head), repr(self.roi_head), info])
