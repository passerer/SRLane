import os
import os.path as osp
from os.path import join

import numpy as np
import pickle as pkl
from tqdm import tqdm

import srlane.evaluation.culane_metric as culane_metric
from .base_dataset import BaseDataset
from .registry import DATASETS

LIST_FILE = {
    "train": "list/train_gt.txt",
    "val": "list/val.txt",
    "test": "list/test.txt",
}

CATEGORYS = {
    "normal": "list/test_split/test0_normal.txt",
    "crowd": "list/test_split/test1_crowd.txt",
    "hlight": "list/test_split/test2_hlight.txt",
    "shadow": "list/test_split/test3_shadow.txt",
    "noline": "list/test_split/test4_noline.txt",
    "arrow": "list/test_split/test5_arrow.txt",
    "curve": "list/test_split/test6_curve.txt",
    "cross": "list/test_split/test7_cross.txt",
    "night": "list/test_split/test8_night.txt",
}


@DATASETS.register_module
class CULane(BaseDataset):
    def __init__(self, data_root, split, processes=None, cfg=None):
        super().__init__(data_root, split, processes=processes, cfg=cfg)
        self.list_path = join(data_root, LIST_FILE[split])
        self.split = split
        self.load_annotations()
        self.h_samples = np.arange(270, 590, 8) / 590

    def load_annotations(self, diff_thr=15):
        self.logger.info("Loading CULane annotations...")
        os.makedirs(".cache", exist_ok=True)
        cache_path = f".cache/culane_{self.split}.pkl"
        if osp.exists(cache_path):
            with open(cache_path, "rb") as cache_file:
                self.data_infos = pkl.load(cache_file)
                self.max_lanes = max(
                    len(anno["lanes"]) for anno in self.data_infos)
                return

        self.data_infos = []
        with open(self.list_path) as list_file:
            prev_img = np.zeros(1)
            for i, line in tqdm(enumerate(list_file)):
                infos = {}
                line = line.split()
                img_line = line[0]
                img_line = img_line[1 if img_line[0] == '/' else 0::]
                img_path = join(self.data_root, img_line)
                if self.split == "train":
                    img = self.imread(img_path)
                    diff = np.abs(img.astype(np.float32) -
                                  prev_img.astype(np.float32)).sum()
                    diff /= (img.shape[0] * img.shape[1] * img.shape[2])
                    prev_img = img
                    if diff < diff_thr:
                        continue
                infos["img_name"] = img_line
                infos["img_path"] = img_path

                if len(line) > 1:
                    mask_line = line[1]
                    mask_line = mask_line[1 if mask_line[0] == '/' else 0::]
                    mask_path = join(self.data_root, mask_line)
                    infos["mask_path"] = mask_path

                if len(line) > 2:
                    exist_list = [int(marker) for marker in line[2:]]
                    infos["lane_exist"] = np.array(exist_list)

                anno_path = img_path[:-3] + "lines.txt"
                with open(anno_path, 'r') as anno_file:
                    data = [
                        list(map(float, line.split()))
                        for line in anno_file.readlines()
                    ]
                lanes = [
                    [(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)
                     if lane[i] >= 0 and lane[i + 1] >= 0] for lane in data]
                lanes = [list(set(lane)) for lane in
                         lanes]  # remove duplicated points
                lanes = [lane for lane in lanes
                         if
                         len(lane) > 2]  # remove lanes with less than 2 points

                lanes = [sorted(lane, key=lambda x: x[1])
                         for lane in lanes]  # sort by y
                infos["lanes"] = lanes
                self.data_infos.append(infos)

        with open(cache_path, "wb") as cache_file:
            pkl.dump(self.data_infos, cache_file)

    def get_prediction_string(self, pred):
        ys = self.h_samples
        out = []
        for lane in pred:
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            lane_xs = xs[valid_mask] * self.cfg.ori_img_w
            lane_ys = ys[valid_mask] * self.cfg.ori_img_h
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            lane_str = ' '.join([
                f"{x:.5f} {y:.5f}" for x, y in zip(lane_xs, lane_ys)
            ])
            if lane_str != '':
                out.append(lane_str)

        return '\n'.join(out)

    def evaluate(self, predictions, output_basedir):
        self.logger.info("Generating CULane prediction output...")
        for idx, pred in enumerate(predictions):
            output_dir = join(
                output_basedir,
                osp.dirname(self.data_infos[idx]["img_name"]))
            output_filename = osp.basename(
                self.data_infos[idx]["img_name"])[:-3] + "lines.txt"
            os.makedirs(output_dir, exist_ok=True)
            output = self.get_prediction_string(pred)

            with open(join(output_dir, output_filename),
                      'w') as out_file:
                out_file.write(output)
        if self.split == "test":
            for cate, cate_file in CATEGORYS.items():
                culane_metric.eval_predictions(output_basedir,
                                               self.data_root,
                                               join(self.data_root, cate_file),
                                               iou_thresholds=[0.5],
                                               official=True)

        result = culane_metric.eval_predictions(output_basedir,
                                                self.data_root,
                                                self.list_path,
                                                iou_thresholds=[0.5],
                                                official=True)

        return result[0.5]["F1"]
