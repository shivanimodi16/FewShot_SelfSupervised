#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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
from google.colab import drive
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from google.colab import files
from torchvision import models
from torchsummary import summary
from glob import glob
import re
from itertools import compress
import pandas as pd
from torchvision.models.feature_extraction import create_feature_extractor

torch.manual_seed(0)


# In[ ]:


drive.mount('/content/gdrive')
files.view('/content/gdrive/My Drive/Colab Notebooks/train_checkpoints/')
get_ipython().system('git clone https://github.com/samirchar/selfSupervised_fewShot.git')
from selfSupervised_fewShot.dataprep import *


# In[ ]:


target_dataset = 'STL10'
source_dataset = 'CIFAR100'

img_size = 32
train_batch_size = 32
test_batch_size = 64
num_workers = 2

source_root = f'/content/gdrive/My Drive/Colab Notebooks/train_checkpoints/resnet18_{source_dataset.lower()}'
lincls_path = f'{source_root}/lincls_on_{target_dataset.lower()}'
if not os.path.exists(lincls_path):
  os.mkdir(lincls_path)


# In[ ]:


#Get mean and std from training set of source data
mean = np.load(os.path.join(source_root,f'{source_dataset.lower()}_train_mean.npy'))
std = np.load(os.path.join(source_root,f'{source_dataset.lower()}_train_std.npy'))


# In[ ]:


dataset_class = getattr(torchvision.datasets,target_dataset)
#train_idx, train_sampler, valid_idx, val_sampler = train_val_samplers(full_train_size,val_size)

#Train/test transforms
transform_train = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomCrop(img_size, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean,std),
])

transform_test = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(mean,std),
])

if target_dataset=='STL10':
  #Read train/val/test
  trainset = dataset_class(  
      root='data',
      split='train',
      download=True,
      transform=transform_train
      )
  
  testset = dataset_class(
      root='data',
      split='test',
      download=True,
      transform=transform_test
      )
else:
  #Read train/val/test
  trainset = dataset_class(  
      root='data',
      train=True,
      download=True,
      transform=transform_train
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
    shuffle = True)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=test_batch_size,
    shuffle=False,
    num_workers=num_workers)


# In[ ]:


#Load pretrained model on CIFAR100
device = "cuda" if torch.cuda.is_available() else "cpu"
resnet18 = resnet18_small(num_classes=100)

criterion = nn.CrossEntropyLoss()

if device == 'cuda':
    resnet18 = torch.nn.DataParallel(resnet18)
    cudnn.benchmark = True

resnet18,_,_,_,_ = load_ckpt(resnet18,None,None,specific_model = 'best',root = source_root)


# In[ ]:


X_train,y_train = feature_extractor(resnet18.module,'backbone.avgpool',trainloader,device)
X_test,y_test = feature_extractor(resnet18.module,'backbone.avgpool',testloader,device)


# In[ ]:


np.save('X_train.npy',X_train)
np.save('X_test.npy',X_test)

np.save('y_train.npy',y_train)
np.save('y_test.npy',y_test)

