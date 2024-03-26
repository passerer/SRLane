from typing import List

import torch.nn as nn
from torch import Tensor

from ..registry import NECKS


@NECKS.register_module
class ChannelMapper(nn.Module):
    """Channel Mapper to reduce/increase channels of backbone features.

    Args:
        in_channels: Number of input channels per scale.
        out_channels: Number of output channels (used at each scale).
        num_outs: Number of output feature maps.
    """
    def __init__(self,
                 in_channels: List[int],
                 out_channels: int,
                 num_outs: int = None,
                 **kwargs,
                 ):
        super(ChannelMapper, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs if num_outs is not None else len(in_channels)
        assert self.num_outs <= self.num_ins

        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_ins - self.num_outs, self.num_ins):
            l_conv = nn.Conv2d(in_channels[i], out_channels, 1, 1, 0)
            self.lateral_convs.append(l_conv)

    def forward(self,
                inputs: List[Tensor]):
        assert len(inputs) >= len(self.in_channels)
        if len(inputs) > len(self.in_channels):
            for _ in range(len(inputs) - len(self.in_channels)):
                del inputs[0]
        outs = [
            lateral_conv(inputs[i]) for i, lateral_conv in
            enumerate(self.lateral_convs)
        ]
        return tuple(outs)

    def __repr__(self):
        num_params = sum(map(lambda x: x.numel(), self.parameters()))
        return f"#Params of {self._get_name()}: {num_params / 10 ** 3:<.2f}[K]"
