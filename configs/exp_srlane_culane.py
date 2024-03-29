_base_ = [
    "./datasets/culane.py",
    "./models/srlane_r18.py"
]

work_dirs = "work_dirs/sr_cu"

iou_loss_weight = 2.
cls_loss_weight = 2.
xyt_loss_weight = 0.2
angle_loss_weight = 5
attn_loss_weight = 0.  # 0.05
seg_loss_weight = 0.  # 0.5

total_iter = 44440
batch_size = 40
eval_ep = 3
workers = 8
log_interval = 500

precision = "16-mixed"  # "32"

optimizer = dict(type="AdamW", lr=6e-4)

scheduler = dict(type="warmup", warm_up_iters=800, total_iters=total_iter)
