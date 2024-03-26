import random
from functools import partial

import torch
import numpy as np
from mmcv.parallel import collate
from srlane.registry import Registry, build_from_cfg

DATASETS = Registry("datasets")
PROCESS = Registry("process")


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return torch.nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_dataset(split_cfg, cfg):
    return build(split_cfg, DATASETS, default_args=dict(cfg=cfg))


def worker_init_fn(worker_id, seed):
    worker_seed = worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(split_cfg, cfg, is_train=True):
    dataset = build_dataset(split_cfg, cfg)

    init_fn = partial(worker_init_fn, seed=cfg.seed)

    samples_per_gpu = cfg.batch_size // cfg.gpus
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=is_train,
        num_workers=cfg.workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        worker_init_fn=init_fn)

    return data_loader
