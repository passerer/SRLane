# flake8: noqa

"""
Tusimple dataset settings.
"""

dataset_type = "TuSimple"
dataset_path = "data/tusimple"
test_json_file = "data/tusimple/test_label.json"

max_lanes = 5
ori_img_w = 1280
ori_img_h = 720
img_h = 320
img_w = 800
cut_height = 160

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
                 parameters=dict(value=(-10, 10)),
                 p=0.7),
            dict(name="OneOf",
                 transforms=[
                     dict(name="MotionBlur", parameters=dict(k=(3, 5))),
                     dict(name="MedianBlur", parameters=dict(k=(3, 5))),
                     dict(name="CoarseDropout", parameters=dict(p=(0.0, 0.05),
                                                                size_percent=(0.02, 0.05)))
                 ],
                 p=0.2),
            dict(name="OneOf",
                 transforms=[
                     dict(name="Affine",
                          parameters=dict(translate_percent=dict(x=(-0.1, 0.1),
                                                                 y=(-0.1, 0.1)),
                                          rotate=(-10, 10),
                                          scale=(0.8, 1.2),
                                          shear=dict(x=(-10, 10)),
                                          ), ),
                 ],
                 p=0.7),
            dict(name="Resize",
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
        ],
    ),
    dict(type="ToTensor", keys=["img", "gt_lane", "gt_angle"]),
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
    split="trainval",
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
