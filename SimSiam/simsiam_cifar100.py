import torch
import torch.nn as nn

import numpy as np
from PIL import Image, ImageFilter

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision.models import resnet18
from torchvision import transforms

from tqdm import tqdm
import random

# Defining global variables

data_dir = './data'

backbone = 'resnet18' 
projection_dim = 128 

# training parameters
# Apart from lr all referenced from SimSiam paper

seed =  42 
batch_size = 512
workers = 4
epochs = 800
log_interval = 1
image_size = 32
optimizer =  'sgd' 
learning_rate = 0.1 # absolute lr
momentum = 0.9
weight_decay = 0.0005 


# Defining for classes required for SimSiam Model

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


class SimSiamLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z1, z2, p1, p2):

        l1 = -nn.functional.cosine_similarity(p1, z2.detach(), dim=-1).mean()
        l2 = -nn.functional.cosine_similarity(p2, z1.detach(), dim=-1).mean()
        return 0.5 * l1 + 0.5 * l2


class PairTransform:
    """Transform an image two times, one serving as the query and other as key."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        q = self.transform(x)
        k = self.transform(x)
        return [q, k]

def train() -> None:
    assert torch.cuda.is_available()
    cudnn.benchmark = True

    train_losses = []

    train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(*[[0.5071, 0.4867, 0.4408],[0.2675, 0.2565, 0.2761]])]) #cifar 100 mean and std
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]) # cifar 10 mean and std
            # reference: https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151

   
    train_set = CIFAR100(root=data_dir,
                                 train=True,
                                 download=True,
                                 transform=PairTransform(train_transforms))

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=workers,
                              drop_last=True,
                              pin_memory=True)

    criterion = SimSiamLoss()

    # Prepare model
    base_encoder = eval(backbone)
    model = SimSiam(base_encoder).cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        learning_rate,
        momentum=momentum,
        weight_decay=weight_decay)

    min_lr = 1e-3

    # cosine annealing scheduler
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: min_lr + (learning_rate-min_lr) * 0.5 * (1 + np.cos(step / epochs * len(train_loader) * np.pi))
        )

    # SimSiam training
    model.train()
    optimal_loss = 1e5
    for epoch in range(1, epochs + 1):
        loss_meter = AverageMeter("SimSiam_loss")
        train_bar = tqdm(train_loader)
        for idx, ((images), _) in enumerate(train_bar):
            optimizer.zero_grad()
            outs = model(images[0].cuda(non_blocking=True), images[1].cuda(non_blocking=True))
            loss = criterion(outs['z1'], outs['z2'], outs['p1'], outs['p2'])
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_meter.update(loss.item(),images[0].size(0)) #images[0].size(0) is the batch_size 512
            train_loss = loss_meter.avg
            train_bar.set_description("Train epoch {}, SimSiam loss: {:.4f}".format(epoch, train_loss))
            
        if train_loss < optimal_loss:
          optimal_loss = train_loss
          torch.save(model.state_dict(), 'simsiam_best_{}.pt'.format(backbone))
                
        train_losses.append(train_loss)

        # save checkpoint according to log_interval 
        if epoch >= log_interval and epoch % log_interval == 0:
            torch.save(model.state_dict(), 'simsiam_{}_epoch{}.pt'.format(backbone, epoch))

    np.savetxt("train_losses.txt", train_losses)

if __name__ == '__main__':
    train()
