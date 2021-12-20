import torch
import torch.nn as nn

import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torchvision.models import resnet18, resnet34
from tqdm import tqdm

# Defining global variables

data_dir = './data'

backbone = 'resnet18' 
projection_dim = 128 

# training parameters

seed =  42 
batch_size = 512
workers = 4
epochs = 800
log_interval = 200
image_size = 32
optimizer =  'sgd' 
learning_rate = 0.2 # lr = 0.1 * batch_size / 256, referenced from Section B.6 and B.7 of SimCLR paper
momentum = 0.9
weight_decay = 0.0005 

# finetune options
finetune_epochs = 100
load_epoch = 800  # checkpoint for finetune

class prediction_layer(nn.Module):
  def __init__(self, input_d=2048, hidden_d=512, output_d=2048):  
    super().__init__()
    self.layer1 = nn.Sequential(
        nn.Linear(input_d, hidden_d),
        nn.BatchNorm1d(hidden_d),
        nn.ReLU(inplace=True)
    )
    self.layer2 = nn.Linear(hidden_d, output_d)

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    return x

class projection_layer(nn.Module):
  def __init__(self, input_d, hidden_d=2048, output_d=2048): 
    super().__init__()
    self.layer1 = nn.Sequential(
        nn.Linear(input_d, hidden_d),
        nn.BatchNorm1d(hidden_d),
        nn.ReLU(inplace=True)
    )
    self.layer2 = nn.Identity() # for CIFAR datasets, we only use Identity for second layer
    self.layer3 = nn.Sequential(
        nn.Linear(hidden_d, output_d),
        nn.BatchNorm1d(output_d, affine=False)
    )

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    return x

class SimSiam(nn.Module):
  def __init__(self, base_encoder):
    super().__init__()

    self.backbone = base_encoder(pretrained=False)  
    self.feature_dim = self.backbone.fc.in_features
    out_dim = self.backbone.fc.in_features
    self.backbone.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    self.backbone.maxpool = nn.Identity()
    self.backbone.fc = nn.Identity()  
    self.projector = projection_layer(out_dim)
    self.predictor = prediction_layer()

  def forward(self, x1, x2):

    bb = self.backbone
    f = self.projector
    h =  self.predictor

    bb1, bb2 = bb(x1), bb(x2)
    z1, z2 = f(bb1), f(bb2)
    p1, p2 = h(z1), h(z2)
    
    return {'z1': z1, 'z2': z2, 'p1': p1, 'p2': p2}

class AverageMeter(object):
    """Utility to calculate and store average value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class LinearModel(nn.Module):
    def __init__(self, base_encoder: nn.Module, input_dim: int, n_classes: int):
        super().__init__()
        self.base_encoder = base_encoder
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.layer1 = nn.Linear(self.input_dim, self.n_classes)

    def forward(self, x):
        enc = self.base_encoder(x)
        return self.layer1(enc)

def train_epoch(cur_epoch, model, dataloader, optimizer=None):

    softmax_criterion = nn.CrossEntropyLoss().cuda()
    criterion = lambda output, target: softmax_criterion(output, target)

    if optimizer:
        model.train()
    else:
        model.eval()

    loss_meter = AverageMeter('loss')
    acc_meter = AverageMeter('acc')
    loader_bar = tqdm(dataloader)
    
    for i,data in enumerate(loader_bar):
        x, y = data[0].cuda(),data[1].long().squeeze().cuda()

        logits = model(x)
        loss = criterion(logits, y)

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = (logits.argmax(dim=1) == y).float().mean()
        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(acc.item(), x.size(0))
        if optimizer:
            loader_bar.set_description("Train epoch {}, loss: {:.4f}, acc: {:.4f}".format(cur_epoch, loss_meter.avg, acc_meter.avg))
        else:
            loader_bar.set_description("Test epoch {}, loss: {:.4f}, acc: {:.4f}".format(cur_epoch, loss_meter.avg, acc_meter.avg))

    return loss_meter.avg, acc_meter.avg

def train_linear() -> None:
    test_accuracies = []
    test_losses = []
    
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    test_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    train_set = CIFAR100(root=data_dir, train=True, transform=train_transform, download=False)
    test_set = CIFAR100(root=data_dir, train=False, transform=test_transform, download=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,num_workers=workers, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Load pre-trained model
    base_encoder = eval(backbone)
    pre_model = SimSiam(base_encoder).cuda()
    #pre_model.load_state_dict(torch.load('simsiam_best_{}.pt'.format(backbone)))
    pre_model.load_state_dict(torch.load('simsiam_resnet18_epoch2.pt'.format(backbone)))

    # Define Linear model
    model = LinearModel(pre_model.backbone, input_dim=pre_model.feature_dim, n_classes=len(train_set.targets))
    model = model.cuda()
    model.base_encoder.requires_grad = False
    parameters = [param for param in model.parameters() if param.requires_grad is True]  

    optimizer = torch.optim.SGD(
        parameters,
        learning_rate,   
        momentum=momentum,
        weight_decay=0.)

    optimal_loss, optimal_acc = 1e5, 0.
    for epoch in range(1, finetune_epochs + 1):
        train_loss, train_acc = train_epoch(epoch, model,train_loader, optimizer)
        test_loss, test_acc = train_epoch(epoch, model, test_loader)
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)
        if train_loss < optimal_loss:
            optimal_loss = train_loss
            optimal_acc = test_acc
            torch.save(model.state_dict(), 'simsiam_lin_{}_best.pth'.format(backbone))
        
    np.savetxt("test_accuracies_linear.txt", test_accuracies)
    np.savetxt("test_losses_linear.txt", test_losses)

if __name__ == '__main__':
    train_linear()

