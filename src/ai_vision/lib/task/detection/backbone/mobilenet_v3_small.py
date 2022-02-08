from typing import Tuple

import torchvision

from . import Backbone


class MobileNet_v3_Small(Backbone):

    def __init__(self, pretrained: bool, num_frozen_levels: int):
        super().__init__(pretrained, num_frozen_levels)

    def _build_component(self) -> Backbone.Component:
        mobilenet_v3_small = torchvision.models.mobilenet_v3_small(pretrained=self.pretrained)

        # >>> x = torch.randn(1, 3, 240, 360)
        # >>> for i, feature in enumerate(mobilenet_v3_small.features):
        # >>>     x = feature(x)
        # >>>     print(i, x.shape)
        #
        # 0 torch.Size([1, 16, 120, 180])
        # 1 torch.Size([1, 16, 60, 90])
        # 2 torch.Size([1, 24, 30, 45])
        # 3 torch.Size([1, 24, 30, 45])
        # 4 torch.Size([1, 40, 15, 23])
        # 5 torch.Size([1, 40, 15, 23])
        # 6 torch.Size([1, 40, 15, 23])
        # 7 torch.Size([1, 48, 15, 23])
        # 8 torch.Size([1, 48, 15, 23])
        # 9 torch.Size([1, 96, 8, 12])
        # 10 torch.Size([1, 96, 8, 12])
        # 11 torch.Size([1, 96, 8, 12])
        # 12 torch.Size([1, 576, 8, 12])

        conv1 = mobilenet_v3_small.features[:1]
        conv2 = mobilenet_v3_small.features[1:2]
        conv3 = mobilenet_v3_small.features[2:4]
        conv4 = mobilenet_v3_small.features[4:9]
        conv5 = mobilenet_v3_small.features[9:]

        num_conv1_out = 16
        num_conv2_out = 24
        num_conv3_out = 40
        num_conv4_out = 48
        num_conv5_out = 576

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
