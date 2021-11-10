# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


#

import numpy as np
import torch.utils.data as tordata
import os.path as osp
import ops
from ops import age2group
from torchvision.datasets.folder import pil_loader
import random
import torch


class data_prefetcher():
    
    def __init__(self, loader, *norm_index):
        
        self.loader = iter(loader) #built in function to iterate over dataset
        self.normlize = lambda x: x.sub_(0.5).div_(0.5) #to normalize image to [-1,1]
        self.norm_index=norm_index #?
        self.stream = torch.cuda.Stream() #helps in parallel execution on the device
        self.preload()  #member function defined below
        
        
    #Preloading data using cude stream and performing normalization on each batch
    
    def preload(self):
        
        try:
            self.next_input = next(self.loader) #Iterator for next batch of data.
        except StopIteration:
            self.next_input = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = [
                self.normlize(x.cuda(non_blocking=True)) if i in self.norm_index else x.cuda(non_blocking=True)
                for i, x in enumerate(self.next_input)]
                
                
    #to go to the next batch of the input        
    def next(self):
        
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        self.preload()
        return input
    
    
def load_source(dataset_name, train=True, age_group=4):
    
    
    data_root = osp.join(osp.dirname(osp.abspath(__file__)), 'materials', dataset_name)
    with open(osp.join(data_root, '{}.txt'.format('train' if train else 'test')), 'r') as f:
        source = np.array([x.strip().split() for x in f.readlines()])
    
    #Creating the image path
    path = np.array([osp.join(data_root, x[0]) for x in source])
    #Age array
    age = np.array([int(x[1]) for x in source])
    #Group array
    group = age2group(age, age_group)
    return {'path': path, 'age': age, 'group': group}

class BaseDataset(tordata.Dataset):
    
    
    def __init__(self,
                 dataset_name,
                 age_group,
                 train=False,
                 max_iter=0,
                 batch_size=0,
                 transforms=None):
        
        
        self.dataset_name = dataset_name
        self.age_group = age_group
        self.train = train
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.total_pairs = batch_size * max_iter
        self.transforms = transforms

        data = load_source(train=train, dataset_name=dataset_name, age_group=age_group)
        
        self.image_list, self.ages, self.groups = data['path'], data['age'], data['group']
        

        self.mean_ages = np.array([np.mean(self.ages[self.groups == i])
                                   for i in range(self.age_group)]).astype(np.float32)
        
        self.label_group_images = []
        self.label_group_ages = []
        for i in range(self.age_group):
            self.label_group_images.append(
                self.image_list[self.groups == i].tolist())
            self.label_group_ages.append(
                self.ages[self.groups == i].astype(np.float32).tolist())
            
            
    def __len__(self):
        return self.total_pairs

        
    
        
        
                
        
                
        
        
        