#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

File Name :
File Description :
Author : Liangwei Li

"""
import os

import torch as t

from config import *

def dense_to_one_hot(origin_tensor, num_of_classes):
    """
    Convert a dense representation of one vector to one-hot representation
    :param origin_tensor:
    :param num_of_classes:
    :return:
    """
    origin_tensor = origin_tensor.view(-1, 1)
    origin_tensor = origin_tensor.long()
    return t.zeros(origin_tensor.shape[0], num_of_classes).scatter(1, origin_tensor, 1).long()


def check_previous_models():
    """
    check whether there exist previous models, if true, then let user choose whether to use it.
    :return:
    """
    available_models = os.listdir(CHECK_POINT_PATH)
    available_models.sort(key=lambda x: get_time_stamp(x))
    if available_models :
        print('Do you want to keep and load previous models ?')
        key = input('Please type in k(keep) / d(delete): ')
        if key == 'k':
            model_name = 'checkpoints/' + available_models[-1]
            return model_name
        elif key == 'd':
            for model in available_models:
                os.unlink(CHECK_POINT_PATH + model)
            return None
        else:
            print('Please type k or d !')
    else:
        return None


def mylog(file_name, log_content):
    """
    define a logger function
    :param file_name:  the name of the log file
    :param log_content:  the content to be logged
    :return:
    """
    with open(LOG_PATH + file_name + '.log', 'a+') as file:
        file.write(log_content + '\n')
        file.close()


def get_time_stamp(str, time_format=TIME_FORMAT):
    """
    given a time string in a particular format, get the time stamp represented by this string
    :param str: given time string
    :param time_format: time format used
    :return:  time stamp
    """
    import time
    import datetime
    import re
    timestr = re.findall('>_(.*)\.', str)[0]
    return time.mktime(datetime.datetime.strptime(timestr, time_format).timetuple())


if __name__ == '__main__':
    import os
    av = os.listdir(CHECK_POINT_PATH)
    av.sort(key=lambda x: get_time_stamp(x))
    print(av)