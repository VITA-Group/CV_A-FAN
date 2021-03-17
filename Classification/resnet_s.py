'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from advertorch.utils import NormalizeByChannelMeanStd

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, init_weight=1):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.all_layers = 9

        all_layer = []
        all_layer.append(NormalizeByChannelMeanStd(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]))
        all_layer.append(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False))
        all_layer.append(nn.BatchNorm2d(16))
        all_layer.append(nn.ReLU())

        strides = [1] + [1]*(num_blocks[0]-1)
        for stride in strides:
            all_layer.append(block(self.in_planes, 16, stride))
            self.in_planes = 16 * block.expansion

        strides = [2] + [1]*(num_blocks[1]-1)
        for stride in strides:
            all_layer.append(block(self.in_planes, 32, stride))
            self.in_planes = 32 * block.expansion

        strides = [2] + [1]*(num_blocks[2]-1)
        for stride in strides:
            all_layer.append(block(self.in_planes, 64, stride))
            self.in_planes = 64 * block.expansion

        all_layer.append(nn.AdaptiveAvgPool2d((1, 1)))
        all_layer.append(nn.Flatten())
        all_layer.append(nn.Linear(64, num_classes))

        self.sequential_model = nn.Sequential(*all_layer)

        self.w = nn.Parameter(torch.FloatTensor(self.all_layers), requires_grad=True)
        self.w.data.fill_(init_weight)

        self.apply(_weights_init)


    def forward(self, x, end_point=34, start_point=0):

        return self.sequential_model[start_point:end_point](x)

def resnet56(init_weight_eta=1):
    return ResNet(BasicBlock, [9, 9, 9], init_weight=init_weight_eta)

