import torch
from torch import nn
from abc import abstractmethod

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        
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
        self.freeze(block="old")

    def lwf(self):
        self.unfreeze(block="shared_cnn")
        self.unfreeze(block="shared_fc")
        self.unfreeze(block="old")

    def lwf_eq_prob(self):
        self.unfreeze(block="shared_cnn")
        self.unfreeze(block="shared_fc")
        self.unfreeze(block="old")

    def finetune_fc(self):
        self.freeze(block="shared_cnn")
        self.unfreeze(block="shared_fc")
        self.freeze(block="old")

    @abstractmethod
    def forward(self, _input):
        return NotImplementedError