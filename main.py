#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

File Name : main.py
File Description : This is the file where we do the training, testing, validating job
Author : Liangwei Li

"""
import torchvision as tv
from sklearn.metrics import accuracy_score, f1_score

from tools.trainer import Trainer
from tools.tools import check_previous_models
from model.transfer_model import TransferModel
from data.data_loader import *
from config import *



def train(trainer):
    trainer.run()


if __name__ == '__main__':
    # define model
    # model = ResNet34(in_channels=NUM_CHANNELS, out_classes=NUM_CLASSES).to(DEVICE)
    # model = resnet18(pretrained=False).to(DEVICE)
    from torchvision.models import resnet18
    model = TransferModel(resnet18, pretrained=True, pre_out=1000)
    model = model.to(DEVICE)
    if MODEl_SAVE:
        model_flag = check_previous_models(model.model_name)                       # check if there exist previous models
        if model_flag != None:
            model.load(model_flag)
    else:
        log_path = os.path.join(LOG_PATH, model.model_name)
        previous_logs = os.listdir(log_path)
        for log in previous_logs:
            os.unlink(log_path + log)

    # load_data
    type_name = ["vinegar"]
    train_dic = {}
    val_dic = {}
    test_dic = {}
    for idx, name in enumerate(type_name):
        train_dic[name] = [DataSet(data_type="train", label=k, annotation_type=idx) for k in range(NUM_CLASSES)]
        val_dic[name] = [DataSet(data_type="validation", label=k, annotation_type=idx) for k in range(NUM_CLASSES)]
        test_dic[name] = [DataSet(data_type="test", label=k, annotation_type=idx) for k in range(NUM_CLASSES)]

    train_data = load_data([d for d in [name_data for name_data in train_dic[name] for name in type_name]])
    val_data = load_data([d for d in [name_data for name_data in val_dic[name] for name in type_name]])
    test_data = load_data([d for d in [name_data for name_data in test_dic[name] for name in type_name]])

    # define training details
    # balanced_weight = t.zeros(NUM_CLASSES, ).to(DEVICE)
    # balanced_weight[0] = 0.5
    # balanced_weight[1] = 0.5
    # balanced_weight[2] = 1
    # balanced_weight[3] = 2
    # balanced_weight = balanced_weight.to(DEVICE)
    balanced_weight = None
    criterion = t.nn.CrossEntropyLoss(weight=balanced_weight)
    optimizer = t.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP, gamma=GAMMA)
    trainer = Trainer(
        model=model,
        criterion=criterion,
        scheduler=scheduler,
        optimizer=optimizer,
        dataset=train_data,
        val_dataset=test_data,
        metric=accuracy_score
    )

    train(trainer)
    # evaluate(model=model, metric=accuracy_score, eval_data=val_data)
    # evaluate(model, test_data)
