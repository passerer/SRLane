import math

import torch


def build_scheduler(cfg, optimizer):
    cfg_cp = cfg.scheduler.copy()
    cfg_type = cfg_cp.pop("type")

    if cfg_type != "warmup" and cfg_type not in dir(torch.optim.lr_scheduler):
        raise ValueError(f"{cfg_type} is not defined.")

    if cfg_type == "warmup":
        def warm_up_cosine_lr(iteration):
            warm_up = cfg_cp["warm_up_iters"]
            if iteration <= warm_up:
                return iteration / warm_up
            else:
                return 0.5 * (math.cos((iteration - warm_up) / (
                       cfg_cp["total_iters"] - warm_up) * math.pi) + 1)
        return torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                 warm_up_cosine_lr)
    _scheduler = getattr(torch.optim.lr_scheduler, cfg_type)

    return _scheduler(optimizer, **cfg_cp)
