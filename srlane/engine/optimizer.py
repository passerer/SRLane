import torch


def build_optimizer(cfg, net):
    cfg_cp = cfg.optimizer.copy()
    cfg_type = cfg_cp.pop("type")

    if cfg_type not in dir(torch.optim):
        raise ValueError(f"{cfg_type} is not defined.")

    _optim = getattr(torch.optim, cfg_type)
    return _optim(net.parameters(), **cfg_cp)
