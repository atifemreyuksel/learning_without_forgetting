import torch
from torch import nn
from torchvision.models import vgg16_bn
from models.base_model import BaseModel

class Vggnet(BaseModel):
    def __init__(self, pretrained=True, num_new_classes=10):
        super().__init__()
        base_vgg16 = vgg16_bn(pretrained=pretrained)
        self.shared_cnn_layers = base_vgg16.features
        self.adap_avg_pool = base_vgg16.avgpool
        self.shared_fc_layers = base_vgg16.classifier[:6]
        self.old_layers = base_vgg16.classifier[6:]
        self.new_layers = nn.Linear(self.old_layers[0].in_features, num_new_classes)

    def forward(self, _input):
        cnn_out = self.shared_cnn_layers(_input)
        cnn_out = self.adap_avg_pool(cnn_out)
        shared_fc_out = self.shared_fc_layers(cnn_out)
        # Old task branch
        old_task_outputs = self.old_layers(shared_fc_out)
        # New task branch
        new_task_outputs = self.new_layers(shared_fc_out)
        outputs = torch.cat((old_task_outputs, new_task_outputs), dim=1)
        return outputs, old_task_outputs
