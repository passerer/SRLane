import torch
import torch.nn.functional as F


def sampling_each_level(sample_points: torch.Tensor,
                        value: torch.Tensor,
                        weight=None):
    B, n_queries, n_points, _ = sample_points.shape
    _, C, H_feat, W_feat = value.shape

    # `sampling_points` now has the shape [B*n_groups, n_queries, n_points, 2]
    out = F.grid_sample(
        value, sample_points.float(),
        mode="bilinear", padding_mode="zeros", align_corners=True,
    )

    if weight is not None:
        weight = weight.view(B, n_queries, n_points).unsqueeze(1)
        out *= weight

    return out.permute(0, 2, 3, 1)


def sampling_3d(
        sample_points: torch.Tensor,
        weight: torch.Tensor,
        multi_lvl_values,
):
    B, n_queries, n_points, _ = sample_points.shape
    B, C, _, _ = multi_lvl_values[0].shape

    num_levels = len(multi_lvl_values)

    sample_points_xy = sample_points * 2.0 - 1.0

    sample_points_lvl_weight_list = weight.unbind(-1)

    out = sample_points.new_zeros(
        B, n_queries, n_points, C)

    for i in range(num_levels):
        value = multi_lvl_values[i]
        lvl_weights = sample_points_lvl_weight_list[i]

        out += sampling_each_level(sample_points_xy, value,
                                   weight=lvl_weights)

    return out
