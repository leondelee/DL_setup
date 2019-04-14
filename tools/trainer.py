#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

File Name : Trainer,py
File Description : Define the Trainer class for model training
Author : llw

"""
import heapq
import time

import torch as t

from tools.tools import *


class Trainer:
    """
    define a trainer class to train model
    """
    def __init__(self, model, criterion, scheduler, optimizer, dataset, val_dataset, metric):
        self.model = model
        self.criterion = criterion
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.max_epoch = MAX_EPOCH
        self.batch_size = BATCH_SIZE
        self.model = model
        self.learning_rate = LEARNING_RATE
        self.iterations = 0
        self.stats = {}
        self.plugin_queues = {
            'iteration': [],
            'epoch': [],
            'batch': [],
            'update': [],
        }
        self.loss = 0
        self.metric = metric
        self.current_time = time.strftime(TIME_FORMAT)
        self.last_high = 0

    def register_plugin(self, plugin):
        plugin.register(self)

        intervals = plugin.trigger_interval
        if not isinstance(intervals, list):
            intervals = [intervals]
        for duration, unit in intervals:
            queue = self.plugin_queues[unit]
            queue.append((duration, len(queue), plugin))

    def call_plugins(self, queue_name, time, *args):
        args = (time,) + args
        queue = self.plugin_queues[queue_name]
        if len(queue) == 0:
            return
        while queue[0][0] <= time:
            plugin = queue[0][2]
            getattr(plugin, queue_name)(*args)
            for trigger in plugin.trigger_interval:
                if trigger[1] == queue_name:
                    interval = trigger[0]
            new_item = (time + interval, queue[0][1], plugin)
            heapq.heappushpop(queue, new_item)

    def run(self):
        print('Start training...')
        for q in self.plugin_queues.values():
            heapq.heapify(q)

        for epoch in range(self.max_epoch):
            print('Epoch', epoch)
            self.scheduler.step()
            self.step_one_epoch()
            self.call_plugins('epoch', epoch)
            if (epoch + 1) % UPDATE_FREQ == 0:
                log_content = 'loss at epoch {epoch} is {loss}.\n'.format(
                    epoch=epoch,
                    loss=self.loss,
                )
                val_log, val_score = evaluate(self.model, self.metric, self.val_dataset)
                train_log, train_score = evaluate(self.model, self.metric, self.dataset)
                if val_score > self.last_high:
                    if MODEl_SAVE:
                        self.model.save()
                    self.last_high = val_score
                log_content += "On training set:\n"
                log_content += train_log
                log_content += "On validation set:\n"
                log_content += val_log
                print(log_content)
                mylog(self.model.model_name, self.current_time, log_content)
        log_content = "Training finished!\nHighest {name} is {value}.".format(
            name=self.metric.__name__,
            value=self.last_high
        )
        mylog(self.model.model_name, self.current_time, log_content)
        print(log_content)

    def step_one_epoch(self):
        self.loss = 0
        for iteration, data in enumerate(tqdm(self.dataset)):
            batch_input, batch_label = data
            batch_input = batch_input.to(DEVICE)
            batch_label = batch_label.to(DEVICE)
            self.call_plugins('batch', iteration, batch_input, batch_label)
            input_var = t.autograd.Variable(batch_input).float()
            label_var = t.autograd.Variable(batch_label).long()
            plugin_data = [None, None]
            def closure():
                output_var = self.model(input_var)
                if len(output_var) > 0:
                    output_var = output_var[0]
                output_var = output_var.float()
                self.loss = self.criterion(output_var.view(-1, NUM_CLASSES), label_var.view(-1, 1)[:, 0])
                self.loss.backward()
                if plugin_data[0] is None:
                    plugin_data[0] = output_var.data
                    plugin_data[1] = self.loss
                return plugin_data[1]
            self.optimizer.zero_grad()
            self.optimizer.step(closure)
            self.call_plugins('iteration', iteration, batch_input, batch_label, *plugin_data)
            self.call_plugins('update', iteration, self.model)
        self.iterations += iteration




