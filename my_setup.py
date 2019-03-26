# Author: llw
import os
import urllib.request

components = ["data/", "model/", "tools/"]
for c in components:
    if not os.path.exists(c):
        os.mkdir(c)

urllib.request.urlretrieve("https://raw.githubusercontent.com/leondelee/DL_setup/master/data/data_loader.py", "data_loader.py")
os.rename("data_loader.py", "data/data_loader.py")

urllib.request.urlretrieve("https://raw.githubusercontent.com/leondelee/DL_setup/master/model/BasicModel.py", "BasicModel.py")
os.rename("BasicModel.py", "model/BasicModel.py")

urllib.request.urlretrieve("https://raw.githubusercontent.com/leondelee/DL_setup/master/model/ResNet34.py", "ResNet34.py")
os.rename("ResNet34.py", "model/ResNet34.py")

urllib.request.urlretrieve("https://raw.githubusercontent.com/leondelee/DL_setup/master/tools/tools.py", "tools.py")
os.rename("tools.py", "tools/tools.py")

urllib.request.urlretrieve("https://raw.githubusercontent.com/leondelee/DL_setup/master/tools/trainer.py", "trainer.py")
os.rename("trainer.py", "tools/trainer.py")

urllib.request.urlretrieve("https://raw.githubusercontent.com/leondelee/DL_setup/master/config.py", "config.py")