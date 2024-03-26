import torch
import torch.nn.functional as F


class SegLoss(torch.nn.Module):
    def __init__(self, num_classes=2, ignore_label=255, bg_weight=0.4):
        super(SegLoss, self).__init__()
        weights = torch.ones(num_classes)
        weights[0] = bg_weight
        self.criterion = torch.nn.NLLLoss(ignore_index=ignore_label,
                                          weight=weights)

    def forward(self, preds, targets):
        loss = self.criterion(F.log_softmax(preds, dim=1), targets.long())
        return loss
