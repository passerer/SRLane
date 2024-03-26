import logging
import os.path as osp

import cv2
from torch.utils.data import Dataset
from mmcv.parallel import DataContainer as DC

from .registry import DATASETS
from .process import Process
from srlane.utils.visualization import imshow_lanes


@DATASETS.register_module
class BaseDataset(Dataset):
    def __init__(self, data_root, split, processes=None, cfg=None):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        self.data_root = data_root
        self.training = "train" in split
        self.processes = Process(processes, cfg)

    def view(self, predictions, img_metas):
        img_metas = [item for img_meta in img_metas.data for item in img_meta]
        for lanes, img_meta in zip(predictions, img_metas):
            img_name = img_meta["img_name"]
            img = cv2.imread(osp.join(self.data_root, img_name))
            out_file = osp.join(self.cfg.work_dir, "visualization",
                                img_name.replace('/', '_'))
            lanes = [lane.to_array(img_meta["img_size"]) for lane in lanes]
            imshow_lanes(img, lanes, out_file=out_file)

    def __len__(self):
        return len(self.data_infos)

    @staticmethod
    def imread(path, rgb=True):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        data_info = self.data_infos[idx]
        img = self.imread(data_info["img_path"])
        img = img[self.cfg.cut_height:, :, :]
        sample = data_info.copy()
        sample.update({"img": img})

        if self.training:
            if self.cfg.cut_height != 0:
                new_lanes = []
                for i in sample["lanes"]:
                    lanes = []
                    for p in i:
                        lanes.append((p[0], p[1] - self.cfg.cut_height))
                    new_lanes.append(lanes)
                sample.update({"lanes": new_lanes})

        sample = self.processes(sample)
        meta = {"full_img_path": data_info["img_path"],
                "img_name": data_info["img_name"],
                "img_size": data_info.get("img_size",
                                          (self.cfg.ori_img_h,
                                           self.cfg.ori_img_w)),
                "img_cut_height": self.cfg.cut_height}
        meta = DC(meta, cpu_only=True)
        sample.update({"meta": meta})

        return sample
