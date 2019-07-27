# Author: llw
import os
import urllib.request

components = ["data/", "model/", "utils/", "cfg/"]
for c in components:
    if not os.path.exists(c):
        os.mkdir(c)

urllib.request.urlretrieve("https://raw.githubusercontent.com/leondelee/DL_setup/master/data/data_loader.py", "data_loader.py")
os.rename("data_loader.py", "data/data_loader.py")

urllib.request.urlretrieve("https://raw.githubusercontent.com/leondelee/DL_setup/master/model/smoothnet3d.py.py", "BasicModel.py")
os.rename("BasicModel.py", "model/BasicModel.py")


urllib.request.urlretrieve("https://raw.githubusercontent.com/leondelee/DL_setup/master/utils/tools.py", "tools.py")
os.rename("tools.py", "utils/tools.py")

urllib.request.urlretrieve("https://raw.githubusercontent.com/leondelee/DL_setup/master/utils/trainer.py", "trainer.py")
os.rename("trainer.py", "utils/trainer.py")

urllib.request.urlretrieve("https://raw.githubusercontent.com/leondelee/DL_setup/master/cfg/demo.yml", "demo.yml")
os.rename("demo.yml", "cfg/demo.yml")

urllib.request.urlretrieve("https://raw.githubusercontent.com/leondelee/DL_setup/master/main.py", "main.py")