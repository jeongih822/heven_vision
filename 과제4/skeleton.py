#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

# ResNet block
class NET_ResNet(nn.Module):
    """
    kernel size, stride, padding 값을 넣으면 됩니다
    conv1: 3*1 kernel 사용
    conv2: 1*3 kernel 사용
    conv1, conv2에서 피쳐 크기 유지
    """
    def __init__(self, input, output):
        super(NET_ResNet, self).__init__()

        self.conv1 = nn.Conv2d(input, output, kernel_size=0, stride=0, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(output)

        self.conv2 = nn.Conv2d(output, output, kernel_size=0, stride=0, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(output)
        
    def forward(self, x):
        """
        ResNet block 설계
        """
        
        return out

# ResNet Network architecture
class ResNet(nn.Module):
    """
    num_classes: Mnist 데이터셋 분류해야되는 클래스 개수
    num_layer: ResNet Block 반복 횟수
    conv1: 1 channel -> 32 channel
    conv2: 32 channel -> 1 channel
    conv1, conv2의 경우 피쳐 크기 유지
    input, output, kernel size, stride, padding 값을 넣으면 됩니다
    """
    def __init__(self, block=NET_ResNet, num_classes=10, num_layer=10):
        super(ResNet, self).__init__()
        self.input = 32
        self.output = 32

        self.conv1 = nn.Conv2d(0, 0, kernel_size=0, stride=0, padding=0, bias=False)
        self.conv2 = nn.Conv2d(0, 0, kernel_size=0, stride=0, padding=0, bias=False)

        self.layer = self.make_layer(block, self.input, self.output, num_layer)

        self.linear = nn.Linear(0, num_classes)

    def make_layer(self, block, input, output, num_layer):
        layers = []
        for i in range(num_layer):
            layers.append(block(input, output))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        전체 네트워크 설계
        """

        return out