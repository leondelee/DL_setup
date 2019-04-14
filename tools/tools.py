#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

File Name :
File Description :
Author : Liangwei Li

"""
import cv2
import torch as t
from tqdm import  tqdm
import matplotlib.pyplot as plt
import pptk

from config import *


def show_point_clouds(pts, lbs):
    v = pptk.viewer(pts)
    v.attributes(lbs)


def rotate_image(image, degre, center=None, scale=1):
    (h, w) = image.shape[:2]
    if not center:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, degre, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def dense_to_one_hot(label, num_of_classes):
    """
    Convert a dense representation of one vector to one-hot representation
    :param origin_tensor:
    :param num_of_classes:
    :return:
    """
    res = t.zeros(num_of_classes, ).long()
    res[label] = 1
    return res


def check_previous_models(model_name):
    """
    check whether there exist previous models, if true, then let user choose whether to use it.
    :return:
    """
    checkpoint_path = os.path.join(CHECK_POINT_PATH, model_name)
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    available_models = os.listdir(checkpoint_path)
    available_models.sort(key=lambda x: get_time_stamp(x))
    log_path = os.path.join(LOG_PATH, model_name)
    previous_logs = os.listdir(log_path)
    while available_models:
        print('Do you want to keep and load previous models ?')
        key = input('Please type in k(keep) / d(delete): ')
        if key == 'k':
            model_name = os.path.join(checkpoint_path, available_models[-1])
            print("Loading model {}".format(available_models[-1]))
            return model_name
        elif key == 'd':
            for model in available_models:
                os.unlink(os.path.join(checkpoint_path, model))
            for log in previous_logs:
                os.unlink(os.path.join(log_path, log))
            return None
        else:
            print('Please type k or d !')


def mylog(model_name, time, log_content):
    """
    define a logger function
    :param file_name:  the name of the log file
    :param log_content:  the content to be logged
    :return:
    """
    if not os.path.exists(LOG_PATH):
        os.mkdir(LOG_PATH)
    with open(LOG_PATH + "/" + model_name + "/" + time + '.log', 'a+') as file:
        file.write(log_content + '\n')
        file.close()


def mytensor_writer(writer, arg_dic, iteration):
    for arg_name in arg_dic:
        writer.add_scalar(arg_name, arg_dic[arg_name], iteration)


def my_plot(logfile, key_words="loss"):
    import re
    with open(logfile, 'r') as file:
        lines = file.read()
        record = re.findall("{}.*is (.*)\.".format(key_words), lines)
        record = [float(r) for r in record]
        epochs = [i for i in range(len(record))]
        plt.plot(epochs, record)
        plt.title(key_words)
        plt.savefig(key_words + ".jpg")


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
    timestr = re.findall('(.*)\.', str)[0]
    return time.mktime(datetime.datetime.strptime(timestr, time_format).timetuple())


def evaluate(model, metric, eval_data):
    model.eval()
    print("Evaluating model...\n")
    log_content = ""
    res = 0
    for data in tqdm(eval_data):
        X, y = data
        X = X.float().to(DEVICE)
        y = y.long().to('cpu')
        out = model(X)
        if len(out) > 1:
            out = out[0]
        out = t.argmax(out, dim=2).cpu()
        # print("out: {}, y: {}".format(out, y))
        res += metric(out.view(-1, 1), y.view(-1, 1))
    res = res / len(eval_data)
    log_content += "average {metric_name} is {metric_value}.\n".format(metric_name=metric.__name__, metric_value=res)
    model.train()
    return log_content, res


if __name__ == '__main__':
    import os
    av = os.listdir(LOG_PATH)
    av = os.path.join(LOG_PATH, "<class 'model.transfer_model.TransferModel'>")
    avs = [os.path.join(av, log) for log in os.listdir(av)]
    my_plot(avs[-1], "loss")
