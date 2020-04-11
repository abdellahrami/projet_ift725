# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License:
Other: Suggestions are welcome
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.CNNBaseModel import CNNBaseModel

class ResidualBlock(nn.Module):
    """
    this block is the building block of the residual network. it takes an
    input with in_channels, applies some blocks of convolutional layers
    to reduce it to out_channels and sum it up to the original input,
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        output += self.shortcut(x)
        output = F.relu(output)
        return output

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out
class DenseNet(nn.Module):
    def __init__(self, growthRate, depth,  nClasses):
        super(DenseNet, self).__init__()

        nDenseBlocks = (depth-4) // 3
        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks)
        nChannels += nDenseBlocks*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)


    def _make_dense(self, nChannels, growthRate, nDenseBlocks):
        layers = []
        for i in range(int(nDenseBlocks)):
            layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.dense1(out)
        #out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 32))
        return out

class ResNet(CNNBaseModel):
    """
    Class that implements the ResNet 18 layers model.
    Reference:
    [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
    """

    def __init__(self, init_weights=True):
        """
        Builds ResNet-18 model.
        Args:
            num_classes(int): number of classes. default 200(tiny imagenet)
    
            init_weights(bool): when true uses _initialize_weights function to initialize
            network's weights.
        """
        super(ResNet, self).__init__(init_weights)

        self.in_channels = 64

        self.conv1 = nn.Conv2d(408, 64, kernel_size=3, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_resnet18_layer(64, stride=1)
        self.layer2 = self._make_resnet18_layer(128, stride=2)
        self.layer3 = self._make_resnet18_layer(256, stride=2)
        self.layer4 = self._make_resnet18_layer(512, stride=2)


    def _make_resnet18_layer(self, out_channels, stride):
        """
        Building ResNet layer
        """
        strides = [stride] + [1]
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x: Tensor
        """
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        #output = F.avg_pool2d(output, 2)
        return output


class Bottleneck(nn.Module):
    def __init__(self, nChannels):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, 64  , kernel_size=1, padding = 1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3
                            , bias=False)

        self.bn3 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,256,kernel_size = 1,bias = False)
        self.pool =  nn.MaxPool2d(kernel_size=2, stride=2),
        self.fc_layers = nn.Sequential(
            nn.Linear(3072, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out = torch.cat((x, out), 1) 
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)
        return out
