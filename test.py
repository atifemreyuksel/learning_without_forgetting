import os
import json
import torch
import random
import argparse
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from datetime import datetime

from models.alexnet import Alexnet
from models.vggnet import Vggnet
from datasets.mnist_dataloader import MnistDataset
from torchvision.datasets import ImageNet

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def test(model, val_loader, task_type, num_new_classes):    
    model.eval()
    with torch.no_grad():
        epoch_val_accuracy = 0
        for data, label in val_loader:
            data = data.to(device)
            label = label.to(device)
            val_output, _ = model(data)
            if task_type == "new":
                acc = (val_output[:, -num_new_classes:].argmax(dim=1) == label).float().mean()
            elif task_type == "old":
                acc = (val_output[:, :-num_new_classes].argmax(dim=1) == label).float().mean()
            else:
                KeyError("Please give correct task type")    
            epoch_val_accuracy += acc / len(val_loader)
    return epoch_val_accuracy


parser = argparse.ArgumentParser()
parser.add_argument('--load_from', type=str, default='', help='load the pretrained model from the specified location')
args = parser.parse_args()

gpu_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu_id}'

checkpoint = torch.load(args.load_from, map_location=f'cuda:{gpu_id}')
config_file = checkpoint['config_file']
args = dotdict(json.load(open(config_file, 'r')))

is_multigpu = "0" in args.gpu_ids and "1" in args.gpu_ids

if args.dataset == 'mnist':
    test_dataset_new = MnistDataset(root=args.dataset_dir, phase="test")
    test_dataset_old = ImageNet("data/ImageNet", split="val")

test_loader_new = torch.utils.data.DataLoader(test_dataset_new, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers//2, pin_memory=True)
test_loader_old = torch.utils.data.DataLoader(test_dataset_old, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers//2, pin_memory=True)

if args.model_name == "alexnet":
    model = Alexnet(pretrained=args.pretrained, num_new_classes=args.num_classes)
elif args.model_name == "vgg16":
    model = Vggnet(pretrained=args.pretrained, num_new_classes=args.num_classes)
else:
    raise NotImplementedError('%s is not found' % args.model_name)

if is_multigpu:
    device = 'cuda:0'
    model = nn.DataParallel(model)
else:    
    device = f'cuda:{args.gpu_ids}'
model.to(device)

model.load_state_dict(checkpoint['model_state_dict'])

val_accuracy_old = test(model, test_loader_old, task_type="old", num_new_classes=args.num_classes)
print(f"Old task = accuracy : {val_accuracy_old:.4f}\n")
with open(os.path.join(args.checkpoint_dir, "inference_log.txt"), "w") as f:
    f.write(f"Old task = accuracy : {val_accuracy_old:.4f}\n")

val_accuracy_new = test(model, test_loader_new, task_type="new", num_new_classes=args.num_classes)
print(f"New task = accuracy : {val_accuracy_old:.4f}\n")
with open(os.path.join(args.checkpoint_dir, "inference_log.txt"), "w") as f:
    f.write(f"New task = accuracy : {val_accuracy_old:.4f}\n")
