import os
import json
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from datetime import datetime
from torch.optim.lr_scheduler import StepLR

from models.alexnet import Alexnet
from models.vggnet import Vggnet
from loss import TotalLoss
from datasets.mnist_dataloader import MnistDataset

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def _compute_output_of_old_tasks(init_model_name, train_loader):
    if args.model_name == "alexnet":
        from torchvision.models import alexnet
        model = alexnet(pretrained=True)
    elif args.model_name == "vgg16":
        from torchvision.models import vgg16
        model = vgg16(pretrained=True)
    else:
        raise NotImplementedError('%s is not found' % args.model_name)

    model.eval()
    with torch.no_grad():
        for data, _ in train_loader:
            data = data.to(device)
            old_outputs = model(data)
    return old_outputs

def warmup(model, train_loader, optimizer, criterion, warmup_epochs):
    model.warmup()
    for epoch in range(warmup_epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        model.train()    
        for data, label in tqdm(train_loader):
            data = data.to(device)
            label = label.to(device)

            output, _ = model(data)
            loss = criterion(output, label, is_warmup=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)
        
        model.eval()
        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in val_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(output, label, is_warmup=True)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(val_loader)
                epoch_val_loss += val_loss / len(val_loader)
        print(f"Warmup epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n")
    return model, optimizer

def select_training_strategy(model, train_method):
    if train_method == "featext":
        model.featext()
    elif train_method == "finetune":
        model.finetune()
    elif train_method == "finetune_fc":
        model.finetune_fc()
    elif train_method == "lwf":
        model.lwf()
    else:
        raise NotImplementedError("Choose valid training method")
    return model   

def train(model, train_loader, criterion, optimizer, prev_outputs):
    model.train()
    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)

        output, _ = model(data)
        loss = criterion(output, label, prev_outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)
    return model, optimizer, epoch_accuracy, epoch_loss

def evaluation(model, val_loader, criterion, prev_outputs):    
    model.eval()
    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in val_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label, prev_outputs)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(val_loader)
            epoch_val_loss += val_loss / len(val_loader)
    return epoch_val_accuracy, epoch_val_loss, epoch_val_loss


parser = argparse.ArgumentParser()

# experiment specifics
parser.add_argument('--name', type=str, default='imagenet2mnist', help='name of the experiment. It decides where to store samples and models')        
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints_disc', help='models are saved here')
parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist'], help='Dataset choice')

# training specifics       
parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
parser.add_argument('--num_workers', type=int, default=16, help='num workers for data prep')
parser.add_argument('--epochs', type=int, default=100, help='# of epochs in training')
parser.add_argument('--warmup_epochs', type=int, default=2, help='# of epochs in warmup step')
parser.add_argument('--weight_decay', type=int, default=5e-4, help='Coefficient of weight decay for optimizer')
parser.add_argument('--train_method', type=str, default="lwf", choices=["lwf", "finetune", "featext", "finetune_fc"], help='training strategy for new model')

# for setting inputs
parser.add_argument('--dataset_dir', type=str, default='./data/mnist/') 

# for displays
parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')    

# model and optimizer
parser.add_argument('--model_name', type=str, default='alexnet', choices=["vgg16", "alexnet"], help='create model with given name')
parser.add_argument('--load_from', type=str, default='', help='load the pretrained model from the specified location')
parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--lr_factor', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')

args = parser.parse_args()

checkpoint_dir = os.path.join(args.checkpoints_dir, args.name)
os.makedirs(checkpoint_dir, exist_ok=True)

config_file = os.path.join(checkpoint_dir, f'config_{datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]}.json')
json.dump(vars(args), open(config_file, 'w'))

seed_everything(args.seed)
is_multigpu = "0" in args.gpu_ids and "1" in args.gpu_ids

if args.dataset == 'mnist':
    train_dataset = MnistDataset(args.dataset_dir, phase="train")
    val_dataset = MnistDataset(args.dataset_dir, phase="val")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers//2, pin_memory=True)

if args.model_name == "alexnet":
    model = Alexnet(train_method=args.train_method)
elif args.model_name == "vgg16":
    model = Vggnet(train_method=args.train_method)
else:
    raise NotImplementedError('%s is not found' % args.model_name)

if is_multigpu:
    device = 'cuda:0'
    model = nn.DataParallel(model)
else:    
    device = f'cuda:{args.gpu_ids}'
model.to(device)

# loss function & optimizers
# TODO: Implement KDLoss (knowledge distillation) in loss.py
criterion = TotalLoss(strategy=args.train_method, temp=args.temp)

if args.optimizer_type == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
elif args.optimizer_type == "adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.9), weight_decay=args.weight_decay)
else:
    raise NotImplementedError("choose adam or sgd")

init_epoch = 0
if args.load_from != "":
    checkpoint = torch.load(args.load_from)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    init_epoch = checkpoint['epoch']
    criterion = checkpoint['criterion']

if args.train_method == "lwf":
    # Warm-up for fully connected layers of new task
    prev_outputs = _compute_output_of_old_tasks(args.model_name, train_loader)
    # Warm-up for fully connected layers of new task
    model = warmup(model, train_loader, optimizer, criterion, args.warmup_epochs)
else:
    prev_outputs = None
# Choose training strategy
model = select_training_strategy(model, args.train_method)

for epoch in range(init_epoch, init_epoch + args.epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    model, optimizer, epoch_accuracy, epoch_loss = train(model, train_loader, criterion, optimizer, prev_outputs)
    epoch_val_accuracy, epoch_val_loss, epoch_val_loss = evaluation(model, val_loader, criterion, prev_outputs)
    print(f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n")

    if epoch % args.save_epoch_freq == 0 and epoch:
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'criterion': criterion,
                    'config_file': config_file
                }, 
                os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
            )