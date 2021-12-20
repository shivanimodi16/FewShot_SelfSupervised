import os
import torch
import torch.nn as nn
from torchvision import models
import re
from torchvision.models.feature_extraction import create_feature_extractor

torch.manual_seed(0)

class resnet18_small(nn.Module):
    def __init__(self,num_classes = 10):
        super(resnet18_small, self).__init__()
        self.num_classes = num_classes
        self.backbone = models.resnet18(pretrained = False)

        self.backbone.conv1 = nn.Conv2d(in_channels=self.backbone.conv1.in_channels,
                            out_channels=self.backbone.conv1.out_channels,
                            kernel_size=(3,3),
                            stride = (1,1),
                            padding=(1,1),
                            bias=False)

        self.backbone.maxpool = nn.Identity()#Remove Pooling
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features,self.num_classes) #Change output layer

    def forward(self, x):
        z = self.backbone(x)
        return z

def save_ckpt(epoch,
              net,
              acc,
              optimizer,
              scheduler,
              file_name = 'ckpt.pth',
              root='/content/gdrive/My Drive/Colab Notebooks/train_checkpoints/resnet18_cifar10'):
  print('Saving..')
  state = {
      'net': net.state_dict(),
      'acc': acc,
      'epoch': epoch,
      'opt': optimizer.state_dict(),
      'schd':scheduler.state_dict()
  }
  if not os.path.isdir(root):
      os.mkdir(root)
      
  torch.save(state, f'{root}/{file_name}')
  best_acc = acc

  return best_acc
  
def load_ckpt(net,
              optimizer,
              scheduler,
              specific_model = 'latest',
              root='/content/gdrive/My Drive/Colab Notebooks/train_checkpoints/resnet18_cifar10'):
  
  print('==> Resuming from checkpoint..')
  assert os.path.isdir(root), 'Error: no checkpoint directory found!'
  assert (specific_model == 'latest')|(specific_model == 'best')|(isinstance(specific_model,int)), f'{specific_model} not implemented'
  
  file_name = None
  
  if specific_model == 'latest':
    available_ckpts = next(os.walk(root))[-1]
    nums = [re.findall('[0-9]+',i) for i in available_ckpts]
    nums = [int(j) for i in nums for j in i if j is not None]
    latest_ckpt_number = max(nums)
    file_name = f'ckpt_epoch_{latest_ckpt_number}.pth'
    
  elif specific_model =='best':
    file_name = 'ckpt.pth'
  
  elif isinstance(specific_model,int):
    file_name = f'ckpt_epoch_{specific_model}.pth'


  checkpoint = torch.load(f'{root}/{file_name}')
  net.load_state_dict(checkpoint['net'])
  
  if optimizer is not None:
    optimizer.load_state_dict(checkpoint['opt'])

  if scheduler is not None:
    scheduler.load_state_dict(checkpoint['schd'])

  best_acc = checkpoint['acc']
  start_epoch = checkpoint['epoch']

  return net, optimizer, scheduler,best_acc, start_epoch


def feature_extractor(model,layer_name,dataset,device,return_target = True):
  return_nodes = {layer_name:'output'}
  extractor = create_feature_extractor(model,return_nodes)
  extracted_features = []
  targets_list = []
  with torch.no_grad():
    for inputs, targets in dataset:
      inputs, targets = inputs.to(device), targets.to(device)
      features = extractor(inputs)
      extracted_features.append(features['output'].squeeze())
      targets_list.append(targets)

  extracted_features = torch.concat(extracted_features,dim=0)
  targets = torch.concat(targets_list,dim=0)
  
  if return_target:
    return extracted_features.cpu().numpy(),targets.cpu().numpy()
  
  return extracted_features.cpu().numpy()