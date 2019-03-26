# Author: llw
import os

# path information
ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
os.path.append(ROOT_PATH)
DATA_PATH = os.path.join(ROOT_PATH, "data/")
MODEL_PATH = os.path.join(ROOT_PATH, "model/")
TOOLS_PATH = os.path.join(ROOT_PATH, "tools/")
CHECK_POINT_PATH = os.path.join(ROOT_PATH, "checkpoints/")
LOG_PATH = os.path.join(CHECK_POINT_PATH, "log/")

# image information
VINEGAR = 0
IODINE = 1
# disease level
LEVEL0 = 0
LEVEL1 = 1
LEVEL2 = 2
LEVEL3 = 3

BATCH_SIZE = 8
LOAD_DATA_WORKERS = 4
TIME_FORMAT = "%m_%d_%H:%M:%S"
MAX_EPOCH = 200
USE_GPU = True
LEARNING_RATE = 1e-4

if __name__ == '__main__':
    print(os.path.dirname(os.path.realpath(__file__)))

