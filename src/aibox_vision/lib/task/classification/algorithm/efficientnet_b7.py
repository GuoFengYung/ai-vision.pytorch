from typing import Union, Tuple

from efficientnet_pytorch import EfficientNet
from torch import nn, Tensor
from torch.nn import functional as F

from . import Algorithm


class EfficientNet_B7(Algorithm):

    def __init__(self, num_classes: int,
                 pretrained: bool, num_frozen_levels: int,
                 eval_center_crop_ratio: float):
        super().__init__(num_classes,
                         pretrained, num_frozen_levels,
                         eval_center_crop_ratio)

    def _build_net(self) -> nn.Module:
        if self.pretrained:
            efficientnet_b7 = EfficientNet.from_pretrained(model_name='efficientnet-b7')
        else:
            efficientnet_b7 = EfficientNet.from_name(model_name='efficientnet-b7')

        efficientnet_b7._fc = nn.Linear(in_features=efficientnet_b7._fc.in_features, out_features=self.num_classes)

        # x = torch.randn(1, 3, 600, 1000)
        # x = efficientnet_b7._conv_stem(x); print(x.shape)
        # x = efficientnet_b7._bn0(x); print(x.shape)
        # x = efficientnet_b7._swish(x); print(x.shape)
        # for i in range(len(efficientnet_b7._blocks)):
        #     x = efficientnet_b7._blocks[i](x); print(i, x.shape)
        conv1 = nn.ModuleList([efficientnet_b7._conv_stem, efficientnet_b7._bn0, efficientnet_b7._swish] +
                              list(efficientnet_b7._blocks[:4]))
        conv2 = efficientnet_b7._blocks[4:11]
        conv3 = efficientnet_b7._blocks[11:18]
        conv4 = efficientnet_b7._blocks[18:38]
        conv5 = nn.ModuleList(list(efficientnet_b7._blocks[38:]) +
                              [efficientnet_b7._conv_head, efficientnet_b7._bn1, efficientnet_b7._swish])

        modules = [conv1, conv2, conv3, conv4, conv5]
        assert 0 <= self.num_frozen_levels <= len(modules)

        freezing_modules = modules[:self.num_frozen_levels]

        for module in freezing_modules:
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False

        return efficientnet_b7

    def forward(self,
                padded_image_batch: Tensor,
                gt_classes_batch: Tensor = None) -> Union[Tensor,
                                                          Tuple[Tensor, Tensor]]:
        batch_size, _, padded_image_height, padded_image_width = padded_image_batch.shape
        logit_batch = self.net.forward(padded_image_batch)

        if self.training:
            loss_batch = self.loss(logit_batch, gt_classes_batch)
            return loss_batch
        else:
            pred_prob_batch, pred_class_batch = F.softmax(input=logit_batch, dim=1).max(dim=1)
            return pred_prob_batch, pred_class_batch

    def loss(self, logit_batch: Tensor, gt_classes_batch: Tensor) -> Tensor:
        loss_batch = F.cross_entropy(input=logit_batch, target=gt_classes_batch, reduction='none')
        return loss_batch

    def remove_output_module(self):
        del self.net._fc

    def to_onnx_compatible(self):
        self.net.set_swish(memory_efficient=False)

    @property
    def output_module_weight(self) -> Tensor:
        return self.net._fc.weight.detach()

    @property
    def last_features_module(self) -> nn.Module:
        return self.net._conv_head

    @staticmethod
    def normalization_means() -> Tuple[float, float, float]:
        return 0.485, 0.456, 0.406

    @staticmethod
    def normalization_stds() -> Tuple[float, float, float]:
        return 0.229, 0.224, 0.225
