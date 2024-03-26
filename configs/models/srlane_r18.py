"""
SRLane model settings.
"""

angle_map_size = (4, 10)
hidden_dim = 64
z_mean = [0.9269, 1.1524, 1.2609]
z_std = [0.2379, 0.4805, 0.5709]
n_fpn = 3
feat_ds_strides = [8, 16, 32]
num_points = 72

net = dict(type="TwoStageDetector")

backbone = dict(type="ResNetWrapper",
                resnet="resnet18",
                pretrained=True,
                replace_stride_with_dilation=[False, False, False],
                out_conv=False,)

neck = dict(type="ChannelMapper",
            in_channels=[128, 256, 512],
            out_channels=hidden_dim,
            num_outs=3,)

rpn_head = dict(type="LocalAngleHead",
                num_points=num_points,
                in_channel=hidden_dim,)

roi_head = dict(type="CascadeRefineHead",
                refine_layers=1,
                fc_hidden_dim=hidden_dim * 3,
                prior_feat_channels=hidden_dim,
                sample_points=36,  # 36
                num_groups=6,)
