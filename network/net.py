#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn

class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()

        channel = 1
        self.inFC = 0

    def forward(self, x):
        
        """
        
        convolution
        
        """
        x = x.view(0, self.inFC)
        """

        Fully Connected

        """
        
        return x
