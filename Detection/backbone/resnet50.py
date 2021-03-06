from typing import Tuple
from backbone.resnet50_ori import resnet50_ori
import torchvision
from torch import nn

import backbone.base
import pdb


class ResNet50(backbone.base.Base):

    def __init__(self, pretrained: bool):
        super().__init__(pretrained)

    def features(self) -> Tuple[nn.Module, nn.Module, int, int]:
        
        resnet50 = resnet50_ori(pretrained=self._pretrained)
        # list(resnet50.children()) consists of following modules
        #   [0] = Conv2d, [1] = BatchNorm2d, [2] = ReLU,
        #   [3] = MaxPool2d, [4] = Sequential(Bottleneck...),
        #   [5] = Sequential(Bottleneck...),
        #   [6] = Sequential(Bottleneck...),
        #   [7] = Sequential(Bottleneck...),
        #   [8] = AvgPool2d, [9] = Linear
        children = list(resnet50.children())
        # features = children[:-3] # only feature exclude final
        num_features_out = 1024
        hidden = children[-3] # final bottleneck
        num_hidden_out = 2048
        # fix
        # resnet50.conv1
        # resnet50.bn1
        # resnet50.relu
        # resnet50.maxpool
        # resnet50.layer1
        for parameters in [feature.parameters() for feature in [resnet50.conv1, resnet50.bn1, resnet50.relu, resnet50.maxpool, resnet50.layer1]]:
            for parameter in parameters:
                parameter.requires_grad = False

        return resnet50, hidden, num_features_out, num_hidden_out


# class ResNet50(backbone.base.Base):

#     def __init__(self, pretrained: bool):
#         super().__init__(pretrained)

#     def features(self) -> Tuple[nn.Module, nn.Module, int, int]:
        
#         resnet50 = torchvision.models.resnet50(pretrained=self._pretrained)

#         # list(resnet50.children()) consists of following modules
#         #   [0] = Conv2d, [1] = BatchNorm2d, [2] = ReLU,
#         #   [3] = MaxPool2d, [4] = Sequential(Bottleneck...),
#         #   [5] = Sequential(Bottleneck...),
#         #   [6] = Sequential(Bottleneck...),
#         #   [7] = Sequential(Bottleneck...),
#         #   [8] = AvgPool2d, [9] = Linear
#         children = list(resnet50.children())
#         features = children[:-3]
#         num_features_out = 1024

#         hidden = children[-3]
#         num_hidden_out = 2048

#         for parameters in [feature.parameters() for i, feature in enumerate(features) if i <= 4]:
#             for parameter in parameters:
#                 parameter.requires_grad = False

#         features = nn.Sequential(*features)

#         return features, hidden, num_features_out, num_hidden_out
