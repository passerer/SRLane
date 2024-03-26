"""
Speed Measurement of a lane detector. It takes a configuration file as input.
The average time taken for inference is calculated and printed as output.
"""

import time
import argparse

import torch
from mmengine.config import Config

from srlane.models.registry import build_net


def parse_args():
    parser = argparse.ArgumentParser(description="Speed measure")
    parser.add_argument("config",
                        help="Config file path")
    parser.add_argument("--repetitions", default=1000,
                        help="Repeat times")
    parser.add_argument("--warmup", default=200,
                        help="Trigger GPU initialization")
    args = parser.parse_args()
    args.cuda = True
    return args


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    cfg = Config.fromfile(args.config)
    torch.backends.cudnn.benchmark = False

    net = build_net(cfg)
    print(net)
    net = net.to(device)
    net.eval()

    for i in range(args.warmup):
        input = torch.zeros(1, 3, cfg.img_h, cfg.img_w)
        input = input.to(device)
        net(input)
    input = torch.zeros(1, 3, cfg.img_h, cfg.img_w)
    input = input.to(device)
    if args.cuda:
        torch.cuda.current_stream(device).synchronize()
    start = time.perf_counter()
    for i in range(args.repetitions):
        net(input)
    if args.cuda:
        torch.cuda.current_stream(device).synchronize()
    sum_time = (time.perf_counter() - start) * 1000

    print(f"avg time = {sum_time / args.repetitions:.2f}ms")


if __name__ == "__main__":
    main()
