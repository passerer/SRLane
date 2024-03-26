"""
CULane dataset settings.
"""

dataset_type = "CULane"
dataset_path = "data/CULane"

ori_img_w = 1640
ori_img_h = 590
img_w = 800
img_h = 320
cut_height = 270
max_lanes = 4

train_process = [
    dict(
        type="GenerateLaneLine",
        transforms=[
            dict(name="Resize",
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
            dict(name="HorizontalFlip", parameters=dict(p=1.0), p=0.5),
            dict(name="ChannelShuffle", parameters=dict(p=1.0), p=0.1),
            dict(name="MultiplyAndAddToBrightness",
                 parameters=dict(mul=(0.85, 1.15), add=(-10, 10)),
                 p=0.6),
            dict(name="AddToHueAndSaturation",
                 parameters=dict(value=(-10, 10)),  # -10-10
                 p=0.7),
            dict(name="OneOf",
                 transforms=[
                     dict(name="MotionBlur", parameters=dict(k=(3, 5))),
                     dict(name="MedianBlur", parameters=dict(k=(3, 5))),
                     dict(name="JpegCompression",
                          parameters=dict(compression=(85, 95))),
                 ],
                 p=0.2),  # 0.2
            dict(name="Affine",
                 parameters=dict(
                     translate_percent=dict(x=(-0.1, 0.1), y=(-0.1, 0.1)),
                     rotate=(-10, 10),
                     scale=(0.8, 1.2),
                     shear=dict(x=(-10, 10))),
                 p=0.7),
            dict(name="Resize",
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
        ],
    ),
    dict(type="ToTensor", keys=["img", "gt_lane", "gt_angle", "gt_seg"]),
]
val_process = [
    dict(type="GenerateLaneLine",
         transforms=[
             dict(name="Resize",
                  parameters=dict(size=dict(height=img_h, width=img_w)),
                  p=1.0),
         ],
         training=False),
    dict(type="ToTensor", keys=["img"]),
]

dataset = dict(train=dict(
    type=dataset_type,
    data_root=dataset_path,
    split="train",
    processes=train_process,
    ),
    val=dict(
        type=dataset_type,
        data_root=dataset_path,
        split="test",
        processes=val_process,
    ),
    test=dict(
        type=dataset_type,
        data_root=dataset_path,
        split="test",
        processes=val_process,
    ))

test_parameters = dict(conf_threshold=0.4, nms_thres=50, nms_topk=max_lanes)
