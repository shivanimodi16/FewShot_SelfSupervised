import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pickle 

torch.manual_seed(0)


def write_pickle(obj,file_name):
    with open(file_name, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

def read_pickle(file_name):
    with open(file_name, 'rb') as handle:
        obj = pickle.load(handle)
    return obj


def get_data_stats(data_loader):
  mean = 0.
  std = 0.
  nb_samples = 0.
  for data,_ in data_loader:
      batch_samples = data.size(0)
      data = data.view(batch_samples, data.size(1), -1)
      mean += data.mean(2).sum(0)
      std += data.std(2).sum(0)
      nb_samples += batch_samples

  mean /= nb_samples
  std /= nb_samples

  mean,std = list(mean.numpy().round(4)),list(std.numpy().round(4))
  return mean,std

def get_torchvision_dataset_stats(dataset_name: str, sampler = None ):
  
  data_class = getattr(torchvision.datasets,dataset_name)

  dummy_train = data_class(
      root='data',
      train=True,
      download=True,
      transform=transforms.ToTensor())

  if sampler is not None:
    #Trainset without transforms
    dummy_trainloader = torch.utils.data.DataLoader(
        dummy_train,
        batch_size=100,
        num_workers=2,
        sampler = sampler
        )
  else:
    #Trainset without transforms
    dummy_trainloader = torch.utils.data.DataLoader(
        dummy_train,
        batch_size=100,
        num_workers=2,
        )

  return get_data_stats(dummy_trainloader)

#Create Train/Val samplers

def train_val_samplers(full_train_size,val_size):

  indices = list(range(full_train_size))

  np.random.seed(43)
  np.random.shuffle(indices)

  train_idx, valid_idx = indices[val_size:], indices[:val_size]
  train_sampler = SubsetRandomSampler(train_idx)
  val_sampler = SubsetRandomSampler(valid_idx)

  return train_idx, train_sampler, valid_idx, val_sampler