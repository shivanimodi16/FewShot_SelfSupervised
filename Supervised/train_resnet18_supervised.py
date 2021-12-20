#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torchvision.utils import make_grid
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_url
#from google.colab import drive
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
#from google.colab import files
from torchvision import models
from torchsummary import summary
from glob import glob
import re

torch.manual_seed(0)


# In[ ]:


#drive.mount('/content/gdrive')
#files.view('/content/gdrive/My Drive/Colab Notebooks/train_checkpoints/')

from modeling import *
from dataprep import *


# In[ ]:


dataset = 'CIFAR100'

img_size = 32
train_batch_size = 128
test_batch_size = 100
num_workers = 2
val_size = 5000
full_train_size = 50000 #Could be automatic
num_epochs = 200
resume = True
lr = 0.1
root = f'/content/gdrive/My Drive/Colab Notebooks/train_checkpoints/resnet18_{dataset.lower()}'


# In[ ]:


dataset_class = getattr(torchvision.datasets,dataset)
train_idx, train_sampler, valid_idx, val_sampler = train_val_samplers(full_train_size,val_size)

mean, std = get_torchvision_dataset_stats(dataset,train_sampler)#Train mean and std

np.save(os.path.join(root,f'{dataset.lower()}_train_mean.npy'),mean)
np.save(os.path.join(root,f'{dataset.lower()}_train_std.npy'),std)

#Train/test transforms
transform_train = transforms.Compose([
    transforms.RandomCrop(img_size, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean,std),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean,std),
])

#Read train/val/test
trainset = dataset_class(  
    root='data',
    train=True,
    download=True,
    transform=transform_train
    )

valset = dataset_class(
    root='data',
    train=True,
    download=True,
    transform=transform_test
    )


testset = dataset_class(
    root='data',
    train=False,
    download=True,
    transform=transform_test
    )


# In[ ]:


#Create data loaders
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=train_batch_size,
    num_workers=num_workers,
    sampler = train_sampler)#No need to shuffle b/c of sampler


valloader = torch.utils.data.DataLoader(
    valset,
    batch_size=test_batch_size,
    num_workers=num_workers,
    sampler = val_sampler)


testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=test_batch_size,
    shuffle=False,
    num_workers=num_workers)


# In[ ]:


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# In[ ]:


num_classes = len(trainset.classes)
net = resnet18_small(num_classes=num_classes)
net = net.to(device)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(),
                      lr=lr,
                      momentum=0.9,
                      weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                      T_max=200)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


# In[ ]:



if resume:
    # Load checkpoint.
    net, optimizer, scheduler,best_acc, start_epoch = load_ckpt(net,optimizer,scheduler,root = root)


# Training
def train(epoch,dataloader,net,optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
        if batch_idx % 100 == 0:
              loss, current = loss.item(), batch_idx * len(inputs)
              print(f"loss: {loss:>7f}  [{current:>5d}/{total:>5d}]")

def test(epoch,dataloader,net,optimizer,save_ckpts = True):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss /= num_batches
    acc = 100.*correct/total 

    print(f"Test Error: \n Accuracy: {(acc):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    if save_ckpts:
      # Save checkpoint.
      if acc > best_acc:
        best_acc = save_ckpt(epoch,net,acc,optimizer,scheduler,file_name = 'ckpt.pth',root = root)
      
      if epoch % 10 == 0:
        _ = save_ckpt(epoch,net,acc,optimizer,scheduler,file_name = f'ckpt_epoch_{epoch}.pth',root = root)
      

      


for epoch in range(start_epoch, num_epochs):
    train(epoch,trainloader,net,optimizer)
    test(epoch,valloader,net,optimizer)
    scheduler.step()


# #Test set performance

# In[ ]:


net, optimizer, scheduler,best_acc, start_epoch = load_ckpt(net,optimizer,scheduler,specific_model = 'best',root = root)
print(best_acc,start_epoch)


# In[ ]:


test(start_epoch,
     testloader,
     net,
     optimizer,
     save_ckpts = False)

