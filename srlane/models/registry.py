import torch.nn as nn
from srlane.registry import Registry, build_from_cfg

BACKBONES = Registry("backbones")
HEADS = Registry("heads")
NECKS = Registry("necks")
NETS = Registry("nets")


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbones(cfg):
    return build(cfg.backbone, BACKBONES, default_args=dict(cfg=cfg))


def build_neck(cfg):
    return build(cfg.neck, NECKS, default_args=dict(cfg=cfg))


def build_head(split_cfg, cfg):
    return build(split_cfg, HEADS, default_args=dict(cfg=cfg))


def build_net(cfg):
    return build(cfg.net, NETS, default_args=dict(cfg=cfg))
