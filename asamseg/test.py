import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from functools import partial
import asamseg.utils as myut
from typing import Callable, Tuple, List
import pytorch_lightning as pl
from torchvision import models
from typing import Type, Any, Callable, Union, List, Optional


class base_resnet(nn.Module):
    def __init__(self):
        super(base_resnet, self).__init__()
        self.model = models.resnet34(pretrained=False)
        # self.model.load_state_dict(torch.load('./model/resnet50-19c8e357.pth'))
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        for n, m in self.model.layer3.named_modules():
            if 'conv1' in n:
                m.dilation, m.padding, m.stride = (1, 1), (1, 1), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.model.layer4.named_modules():
            if 'conv1' in n:
                m.dilation, m.padding, m.stride = (1, 1), (1, 1), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x1 = self.model.layer1(x)
        x2 = self.model.layer2(x1)
        x3 = self.model.layer3(x2)
        x4 = self.model.layer4(x3)

        return x4, x3, x2, x1


print(base_resnet().modules)
