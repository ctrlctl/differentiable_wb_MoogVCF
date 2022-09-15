import os
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms

class MLP(nn.Module):
    '''
      Multilayer Perceptron.
    '''
    def __init__(self):
        super().__init__()
        self.layer_dict = nn.ModuleDict()
        self.layer_dict['Linear_0'] = nn.Linear(4, 16)
        self.layer_dict['Tanh_0'] = nn.Tanh()
        self.layer_dict['Linear_1'] = nn.Linear(16, 4)
        self.layer_dict['Tanh_1'] = nn.Tanh()

    def forward(self, x):
        '''Forward pass'''
        out = x
        out = self.layer_dict['Linear_0'].forward(out)
        out = self.layer_dict['Tanh_0'].forward(out)
        out = self.layer_dict['Linear_1'].forward(out)
        out = self.layer_dict['Tanh_1'].forward(out)

        return out