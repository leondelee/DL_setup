#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File Name : BasicModel.py
File Description : Packaging some common functions which will be used frequently
Author : Liangwei Li
"""
import time
import os

import torch as t
from torch import nn

from config import TIME_FORMAT, CHECK_POINT_PATH


class BasicModel(nn.Module):
    """
    Usage: All the childs model should inherit this module, and use straightly model.save() or model.load(path)
    """

    def __init__(self):
        super(BasicModel, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        """
        Save model in the format of "model_name+time'
        :param name:
        :return:
        """
        checkpoint_path = os.path.join(CHECK_POINT_PATH, self.model_name)
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
        if name is None:
            name = time.strftime(TIME_FORMAT + '.pth')
        t.save(self.state_dict(), os.path.join(checkpoint_path, name))
        return name


if __name__ == '__main__':
    test = BasicModel()
    print(test.save())