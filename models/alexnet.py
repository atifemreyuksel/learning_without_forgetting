import torch
from torch import nn
from torchvision.models import alexnet
from models.base_model import BaseModel

class Alexnet(BaseModel):
    def __init__(self, pretrained=True, num_new_classes=10):
        super().__init__()
        base_alexnet = alexnet(pretrained=pretrained)
        self.shared_cnn_layers = base_alexnet.features
        self.adap_avg_pool = base_alexnet.avgpool
        self.shared_fc_layers = base_alexnet.classifier[:6]
        self.old_layers = base_alexnet.classifier[6:]
        self.new_layers = nn.Linear(self.old_layers[0].in_features, num_new_classes)

    def forward(self, _input):
        cnn_out = self.shared_cnn_layers(_input)
        cnn_out = self.adap_avg_pool(cnn_out)
        shared_fc_out = self.shared_fc_layers(cnn_out)
        # Old task branch
        old_task_outputs = self.old_layers(shared_fc_out)
        old_outputs = self.softmax(old_task_outputs)
        # New task branch
        new_task_outputs = self.new_layers(shared_fc_out)
        outputs = torch.cat((old_task_outputs, new_task_outputs), dim=1)
        outputs = self.softmax(outputs)
        return outputs, old_outputs
