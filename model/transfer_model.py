# Author: llw
from config import *
from model.BasicModel import BasicModel


class TransferModel(BasicModel):

    def __init__(self, pre_model, pretrained=True, pre_out=1000, pre_grad=False):
        super(TransferModel, self).__init__()
        self.pre_grad = pre_grad
        self.pre_model = pre_model(pretrained=pretrained)
        self.fc = t.nn.Linear(pre_out, NUM_CLASSES)

    def forward(self, X):
        if self.pre_grad:
            self.pre_model.train()
        else:
            self.pre_model.eval()
        out = self.pre_model(X)
        out = t.nn.functional.relu(out)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    from torchvision.models import resnet18
    tm = TransferModel(resnet18, pretrained=True)
    tm.save()