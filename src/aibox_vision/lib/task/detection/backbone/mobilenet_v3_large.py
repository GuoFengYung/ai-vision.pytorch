from typing import Tuple

import torchvision

from . import Backbone


class MobileNet_v3_Large(Backbone):

    def __init__(self, pretrained: bool, num_frozen_levels: int):
        super().__init__(pretrained, num_frozen_levels)

    def _build_component(self) -> Backbone.Component:
        mobilenet_v3_large = torchvision.models.mobilenet_v3_large(pretrained=self.pretrained)

        # >>> x = torch.randn(1, 3, 240, 360)
        # >>> for i, feature in enumerate(mobilenet_v3_large.features):
        # >>>     x = feature(x)
        # >>>     print(i, x.shape)
        #
        # 0 torch.Size([1, 16, 120, 180])
        # 1 torch.Size([1, 16, 120, 180])
        # 2 torch.Size([1, 24, 60, 90])
        # 3 torch.Size([1, 24, 60, 90])
        # 4 torch.Size([1, 40, 30, 45])
        # 5 torch.Size([1, 40, 30, 45])
        # 6 torch.Size([1, 40, 30, 45])
        # 7 torch.Size([1, 80, 15, 23])
        # 8 torch.Size([1, 80, 15, 23])
        # 9 torch.Size([1, 80, 15, 23])
        # 10 torch.Size([1, 80, 15, 23])
        # 11 torch.Size([1, 112, 15, 23])
        # 12 torch.Size([1, 112, 15, 23])
        # 13 torch.Size([1, 160, 8, 12])
        # 14 torch.Size([1, 160, 8, 12])
        # 15 torch.Size([1, 160, 8, 12])
        # 16 torch.Size([1, 960, 8, 12])

        conv1 = mobilenet_v3_large.features[:2]
        conv2 = mobilenet_v3_large.features[2:4]
        conv3 = mobilenet_v3_large.features[4:7]
        conv4 = mobilenet_v3_large.features[7:13]
        conv5 = mobilenet_v3_large.features[13:]

        num_conv1_out = 16
        num_conv2_out = 24
        num_conv3_out = 40
        num_conv4_out = 112
        num_conv5_out = 960

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
