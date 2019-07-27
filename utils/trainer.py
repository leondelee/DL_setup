#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

File Name : Trainer,py
File Description : Define the Trainer class for model training
Author : llw

"""
import os
import time

from sklearn.metrics import accuracy_score, mean_squared_error

from utils.tools import *


class Trainer:
    def __init__(self, model, criterion, scheduler, train_data, test_data, cfg, logger):
        self.model = model
        self.criterion = criterion
        self.scheduler = scheduler
        self.train_data = train_data
        self.test_data = test_data
        self.max_epoch = int(cfg["max_epoch"])
        self.eval_step = int(cfg["eval_step"])
        self.loss = 0
        self.metric = 0
        self.best_score = 0
        self.cfg = cfg
        self.logger = logger
        self.iteration = 0

    def run(self):
        self.logger.info('-' * 20 + "New Training Starts" + '-' * 20)
        for epoch in range(self.max_epoch):
            tic = time.time()
            self.logger.info("Epoch {}/{}:".format(epoch, self.max_epoch))
            self.step()
            if (epoch + 1) % self.eval_step == 0:
                log_info = evaluate(self.model, self.cfg['metric'], self.test_data)
                log_content = "Average {} is {}.".format(log_info['metric_name'], log_info['value'])
                self.metric = log_info['value']
                self.logger.info(log_content)
                if self.best_score > self.metric:
                    self.best_score = self.metric
                    self.logger.info("Saving checkpoint to {}.".format(os.path.join(self.cfg["root_path"],
                                                                               self.cfg["checkpoint_path"],
                                                                               self.cfg["name"] +
                                                                               self.cfg["checkpoint_name"]))
                                )
                    t.save(
                        self.model.state_dict(),
                        os.path.join(self.cfg["root_path"], self.cfg["checkpoint_path"], self.cfg["name"] + self.cfg["checkpoint_name"])
                    )
            self.logger.info("loss: {}".format(self.loss))
            toc = time.time()
            self.logger.info("Elapsed time is {}s".format(toc - tic))

    def step(self):
        self.loss = 0
        self.iteration += 1
        if self.iteration % self.cfg["lr_step"] == 0:
            for group in self.scheduler.param_groups:
                group['lr'] *= self.cfg["lr_decay"]
        cnt = 0
        self.model.train()
        for batch_data in tqdm(self.train_data):
            self.scheduler.zero_grad()
            output = self.model(batch_data)
            loss = self.criterion(output) # + feature_transform_reguliarzer(trans_feat) * 0.001
            loss.backward()
            self.scheduler.step()
            self.loss += loss.data
            cnt += 1
        self.loss = self.loss / cnt


if __name__ == '__main__':
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.FileHandler("test.log"))
    logger.info("test")