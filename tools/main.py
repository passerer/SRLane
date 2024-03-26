import os
import argparse

import torch.backends.cudnn as cudnn
from mmengine.config import Config

from srlane.engine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config",
                        help="Config file path")
    parser.add_argument("--work_dirs", type=str, default=None,
                        help="Dirs for log and saving ckpts")
    parser.add_argument("--load_from", default=None,
                        help="The checkpoint file to load from")
    parser.add_argument("--view", action="store_true",
                        help="Whether to visualize results during validation")
    parser.add_argument("--validate", action="store_true",
                        help="Whether to evaluate the checkpoint")
    parser.add_argument("--gpus", nargs='+', type=int, default=[0, ],
                        help="Used GPU indices")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(gpu) for gpu in args.gpus)

    cfg = Config.fromfile(args.config)
    cfg.gpus = len(args.gpus)
    cfg.load_from = args.load_from
    cfg.view = args.view
    cfg.seed = args.seed
    cfg.work_dirs = args.work_dirs if args.work_dirs else cfg.work_dirs

    cudnn.benchmark = True

    runner = Runner(cfg)
    if args.validate:
        runner.validate()
    else:
        runner.train()


if __name__ == "__main__":
    main()
