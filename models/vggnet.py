import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg16

class Vggnet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base_vgg16 = vgg16(pretrained=pretrained)
        self.shared_cnn_layers = None
        self.shared_fc_layers = None
        self.old_layers = None
        self.new_layers = None

    def freeze(self, block="shared"):
        if block == "shared_cnn":
            for param in self.shared_cnn_layers.parameters():
                param.requires_grad = False
        if block == "shared_fc":
            for param in self.shared_fc_layers.parameters():
                param.requires_grad = False
        if block == "old":
            for param in self.old_layers.parameters():
                param.requires_grad = False

    def unfreeze(self, block="shared"):
        if block == "shared_cnn":
            for param in self.shared_cnn_layers.parameters():
                param.requires_grad = True
        if block == "shared_fc":
            for param in self.shared_fc_layers.parameters():
                param.requires_grad = True
        if block == "old":
            for param in self.old_layers.parameters():
                param.requires_grad = True

    def warmup(self):
        self.freeze(block="shared_cnn")
        self.freeze(block="shared_fc")
        self.freeze(block="old")

    def featext(self):
        self.freeze(block="shared_cnn")
        self.freeze(block="shared_fc")
        self.freeze(block="old")

    def finetune(self):
        self.unfreeze(block="shared_cnn")
        self.unfreeze(block="shared_fc")
        self.unfreeze(block="old")

    def lwf(self):
        self.unfreeze(block="shared_cnn")
        self.unfreeze(block="shared_fc")
        self.unfreeze(block="old")

    def finetune_fc(self):
        self.freeze(block="shared_cnn")
        self.unfreeze(block="shared_fc")
        self.unfreeze(block="old")

    def forward(self, input):
        pass
