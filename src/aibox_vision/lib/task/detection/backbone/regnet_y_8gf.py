from typing import Union, Tuple

import torchvision
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from . import Backbone


class RegNet_Y_8GF(Backbone):

    def __init__(self, pretrained: bool, num_frozen_levels: int):
        super().__init__(pretrained, num_frozen_levels)

    def _build_component(self) -> Backbone.Component:
        regnet_y_8gf = torchvision.models.regnet_y_8gf(pretrained=self.pretrained)
        regnet_y_8gf.fc = nn.Linear(in_features=regnet_y_8gf.fc.in_features, out_features=self.num_classes)

        # x = torch.randn(1, 3, 224, 224)
        # for i in range(len(regnet_y_400mf.stem)):
        #     x = regnet_y_400mf.stem[i](x)
        #     print(i, x.shape)
        # for i in range(len(regnet_y_400mf.trunk_output)):
        #     x = regnet_y_400mf.trunk_output[i](x)
        #     print(i, x.shape)
        conv1 = regnet_y_8gf.stem
        conv2 = regnet_y_8gf.trunk_output[:1]
        conv3 = regnet_y_8gf.trunk_output[1:2]
        conv4 = regnet_y_8gf.trunk_output[2:3]
        conv5 = regnet_y_8gf.trunk_output[3:]

        num_conv1_out = 128
        num_conv2_out = 256
        num_conv3_out = 512
        num_conv4_out = 1024
        num_conv5_out = 2048

        return Backbone.Component(
            conv1, conv2, conv3, conv4, conv5,
            num_conv1_out, num_conv2_out, num_conv3_out, num_conv4_out, num_conv5_out
        )

    @staticmethod
    def normalization_means() -> Tuple[float, float, float]:
        return 0.485, 0.456, 0.406

    @staticmethod
    def normalization_stds() -> Tuple[float, float, float]:
        return 0.229, 0.224, 0.225
