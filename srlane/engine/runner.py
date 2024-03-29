import time
import random

import torch
import numpy as np
from tqdm import tqdm
from lightning.fabric import Fabric

from .optimizer import build_optimizer
from .scheduler import build_scheduler
from srlane.models.registry import build_net
from srlane.datasets import build_dataloader
from srlane.utils.recorder import build_recorder
from srlane.utils.net_utils import save_model, load_network


class Runner(object):
    def __init__(self, cfg):
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        self.cfg = cfg
        self.recorder = build_recorder(self.cfg)
        self.net = build_net(cfg)
        self.load_network()
        self.optimizer = build_optimizer(self.cfg, self.net)
        self.scheduler = build_scheduler(self.cfg, self.optimizer)
        self.fabric = Fabric(accelerator="cuda",
                             devices=cfg.gpus,
                             strategy="dp",
                             precision=cfg.precision)
        self.fabric.launch()
        self.net, self.optimizer = self.fabric.setup(self.net, self.optimizer)

        self.val_loader = None
        self.test_loader = None
        self.metric = 0

    def load_network(self):
        if not self.cfg.load_from:
            return
        load_network(self.net, self.cfg.load_from, strict=False)

    def train_epoch(self, train_loader):
        self.net.train()
        end = time.time()
        for i, data in enumerate(train_loader):
            if self.recorder.step >= self.cfg.total_iter:
                break
            date_time = time.time() - end
            self.recorder.step += 1
            output = self.net(data)
            self.optimizer.zero_grad()
            loss = output["loss"].sum()
            self.fabric.backward(loss)
            self.optimizer.step()
            self.scheduler.step()
            batch_time = time.time() - end
            end = time.time()
            self.recorder.update_loss_status(output["loss_status"])
            self.recorder.batch_time.update(batch_time)
            self.recorder.data_time.update(date_time)

            if i % self.cfg.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                self.recorder.lr = lr
                self.recorder.record("train")

    def train(self):
        self.recorder.logger.info("Build train_loader...")
        train_loader = build_dataloader(self.cfg.dataset.train,
                                        self.cfg,
                                        is_train=True)
        train_loader = self.fabric.setup_dataloaders(train_loader)
        self.recorder.logger.info("Start training...")
        epoch = 0
        while self.recorder.step < self.cfg.total_iter:
            self.recorder.epoch = epoch
            self.train_epoch(train_loader)
            if (self.recorder.step >= self.cfg.total_iter
                    or (epoch + 1) % self.cfg.eval_ep == 0):
                self.validate()
            epoch += 1

    @torch.no_grad()
    def validate(self):
        if not self.val_loader:
            self.val_loader = build_dataloader(self.cfg.dataset.val,
                                               self.cfg,
                                               is_train=False)
        net = self.net
        net.eval()
        predictions = []
        for i, data in enumerate(tqdm(self.val_loader, desc="Validate")):
            output = net(data)
            output = net.module.roi_head.get_lanes(output, data["meta"])
            predictions.extend(output)
            if self.cfg.view:
                self.val_loader.dataset.view(output, data["meta"])

        metric = self.val_loader.dataset.evaluate(predictions,
                                                  self.cfg.work_dir)
        self.recorder.logger.info("metric: " + str(metric))
        self.recorder.tb_logger.add_scalar("val/metric", metric,
                                           self.recorder.step)
        if metric > self.metric:
            self.metric = metric
            save_model(net, self.recorder)
