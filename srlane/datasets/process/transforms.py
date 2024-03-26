import torch
import numpy as np

from ..registry import PROCESS


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f"type {type(data)} cannot be converted to tensor.")


@PROCESS.register_module
class ToTensor(object):
    """Convert some results to :obj:`torch.Tensor` by given keys.

    Args:
        keys (Sequence[str]): Keys that need to be converted to Tensor.
    """

    def __init__(self, keys=["img", "mask"], cfg=None):
        self.keys = keys

    def __call__(self, sample):
        data = {}
        if len(sample["img"].shape) < 3:
            sample["img"] = np.expand_dims(sample["img"], -1)
        for key in self.keys:
            if isinstance(sample[key], list) or isinstance(sample[key], dict):
                data[key] = sample[key]
                continue
            data[key] = to_tensor(sample[key])
        data["img"] = data["img"].permute(2, 0, 1)
        return data

    def __repr__(self):
        return self.__class__.__name__ + f"(keys={self.keys})"
