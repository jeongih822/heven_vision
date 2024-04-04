#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn

class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()

        channel = 1
        self.inFC = -1
            
        self.firstConv = nn.Conv2d(1, 10, padding=1, padding_mode='zeros', kernel_size=3) #26*26 10
        
        self.conv1 = nn.Conv2d(10, 20, padding=1, padding_mode='zeros', kernel_size=3) #12*12 20
        self.conv2 = nn.Conv2d(20, 40, padding=1, padding_mode='zeros', kernel_size=3) #5*5 40
        
        
        self.fc1 = nn.Linear(7*7*40, 400)
        self.fc2 = nn.Linear(400, 20)
        
        self.lastLayer = nn.Linear(20,10)

        self.mp = nn.MaxPool2d(2)
        
        self.relu = nn.LeakyReLU()

        

    def forward(self, x):
        x = self.firstConv(x)

        x = self.relu(x)
        x = self.relu(self.mp(self.conv1(x)))
        x = self.relu(self.mp(self.conv2(x)))
        """
        
        convolution
        
        """
        # x.size(0) : batch_size, self.inFC : input size of FC layer
        x = x.view(x.size(0), self.inFC)
        """

        Fully Connected

        """

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.lastLayer(x)
        
        return x
