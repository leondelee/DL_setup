# Author: llw
import os
import sys

import torch as t

# path information
ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(ROOT_PATH)
DATA_PATH = os.path.join(ROOT_PATH, "data/")
MODEL_PATH = os.path.join(ROOT_PATH, "model/")
TOOLS_PATH = os.path.join(ROOT_PATH, "tools/")
CHECK_POINT_PATH = os.path.join(ROOT_PATH, "checkpoints/")
LOG_PATH = os.path.join(ROOT_PATH, "log/")
TENSORBOARD_LOG_PATH = os.path.join(ROOT_PATH, "runs/")
for pth in [ROOT_PATH, DATA_PATH, MODEL_PATH, TOOLS_PATH, CHECK_POINT_PATH, LOG_PATH]:
    if not os.path.exists(pth):
        os.mkdir(pth)

# device
DEVICE = 'cuda'

# image information
VINEGAR = 0
IODINE = 1
NUM_CHANNELS = 3
NUM_CLASSES = 3
RESIZE = False
HEIGHT = 224
WIDTH = 224
# disease level
"""
0 = 2
1 = 3
2 = cancer
"""
LEVELS = [l for l in range(NUM_CLASSES)]
LEVEL_FOLDER = os.listdir(os.path.join(DATA_PATH, "train/vinegar"))
LEVEL_FOLDER.sort()

# Training Details
BATCH_SIZE = 32
LOAD_DATA_WORKERS = 0
TIME_FORMAT = "%m_%d_%H:%M:%S"
MAX_EPOCH = 500
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.9
SCHEDULER_STEP = 50
GAMMA = 0.5
UPDATE_FREQ = 1
MODEl_SAVE = True


if __name__ == '__main__':
    print(LEVEL_FOLDER)
