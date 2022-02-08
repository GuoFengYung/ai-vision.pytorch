from typing import Union, Tuple, List, Dict, Optional

import torch
import torchvision.models.detection
import torchvision.models.detection.transform
from torch import Tensor, nn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import mobilenet_backbone, resnet_fpn_backbone
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.transform import resize_boxes

from . import Algorithm
from ..backbone import Backbone
from ..backbone.mobilenet_v3_small import MobileNet_v3_Small
from ..backbone.mobilenet_v3_large import MobileNet_v3_Large
from ..backbone.resnet50 import ResNet50


class TorchFPN(Algorithm):

    def __init__(self, num_classes: int,
                 backbone: Backbone,
                 anchor_ratios: List[Tuple[int, int]], anchor_sizes: List[int],
                 train_rpn_pre_nms_top_n: int, train_rpn_post_nms_top_n: int,
                 eval_rpn_pre_nms_top_n: int, eval_rpn_post_nms_top_n: int,
                 num_anchor_samples_per_batch: int, num_proposal_samples_per_batch: int, num_detections_per_image: int,
                 anchor_smooth_l1_loss_beta: float, proposal_smooth_l1_loss_beta: float,
                 proposal_nms_threshold: float, detection_nms_threshold: float):
        # NOTE: Arguments `anchor_smooth_l1_loss_beta` and `proposal_smooth_l1_loss_beta` are ignored
        super().__init__(num_classes,
                         backbone,
                         anchor_ratios, anchor_sizes,
                         train_rpn_pre_nms_top_n, train_rpn_post_nms_top_n,
                         eval_rpn_pre_nms_top_n, eval_rpn_post_nms_top_n,
                         num_anchor_samples_per_batch, num_proposal_samples_per_batch, num_detections_per_image,
                         anchor_smooth_l1_loss_beta, proposal_smooth_l1_loss_beta,
                         proposal_nms_threshold, detection_nms_threshold)
        if isinstance(backbone, MobileNet_v3_Small):
            backbone_name = Backbone.Name.MOBILENET_V3_SMALL.value
            backbone = mobilenet_backbone(backbone_name, pretrained=backbone.pretrained, fpn=True,
                                          trainable_layers=5 - backbone.num_frozen_levels)
            anchor_sizes = (tuple(it for it in anchor_sizes),) * len(anchor_ratios)
        elif isinstance(backbone, MobileNet_v3_Large):
            backbone_name = Backbone.Name.MOBILENET_V3_LARGE.value
            backbone = mobilenet_backbone(backbone_name, pretrained=backbone.pretrained, fpn=True,
                                          trainable_layers=5 - backbone.num_frozen_levels)
            anchor_sizes = (tuple(it for it in anchor_sizes),) * len(anchor_ratios)
        elif isinstance(backbone, ResNet50):
            backbone_name = Backbone.Name.RESNET50.value
            backbone = resnet_fpn_backbone(backbone_name, pretrained=backbone.pretrained,
                                           trainable_layers=5 - backbone.num_frozen_levels)
            anchor_sizes = tuple((it,) for it in anchor_sizes)  # format like ((128,), (256,), (512,)) is expected
        else:
            raise ValueError(f'Unsupported backbone for this algorithm')

        # NOTE: These arguments have no effect since we will replace
        #       default `GeneralizedRCNNTransform` with our `IdentityTransform`
        min_size, max_size, image_mean, image_std = None, None, None, None

        aspect_ratios = ((tuple(it[0] / it[1] for it in anchor_ratios)),) * len(anchor_sizes)  # format like ((0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0)) is expected
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        self.net = torchvision.models.detection.FasterRCNN(
            backbone, num_classes, min_size, max_size, image_mean, image_std, rpn_anchor_generator,
            rpn_pre_nms_top_n_train=train_rpn_pre_nms_top_n, rpn_pre_nms_top_n_test=eval_rpn_pre_nms_top_n,
            rpn_post_nms_top_n_train=train_rpn_post_nms_top_n, rpn_post_nms_top_n_test=eval_rpn_post_nms_top_n,
            rpn_nms_thresh=proposal_nms_threshold, rpn_batch_size_per_image=num_anchor_samples_per_batch,
            box_nms_thresh=detection_nms_threshold, box_detections_per_img=num_detections_per_image,
            box_batch_size_per_image=num_proposal_samples_per_batch
        )
        self.net.transform = self.IdentityTransform()

    def forward(
            self, padded_image_batch: Tensor,
            gt_bboxes_batch: List[Tensor] = None, gt_classes_batch: List[Tensor] = None
    ) -> Union[Tuple[Tensor, Tensor, Tensor, Tensor],
               Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor]]]:
        batch_size = len(padded_image_batch)
        padded_image_batch = [it for it in padded_image_batch]

        if self.training:
            targets = []
            for gt_bboxes, gt_classes in zip(gt_bboxes_batch, gt_classes_batch):
                target = {
                    'boxes': gt_bboxes,
                    'labels': gt_classes
                }
                targets.append(target)

            loss_dict = self.net(padded_image_batch, targets)

            # For compatibility with the interface
            anchor_objectness_loss_batch = torch.stack([loss_dict['loss_objectness'] for _ in range(batch_size)], dim=0)
            anchor_transformer_loss_batch = torch.stack([loss_dict['loss_rpn_box_reg'] for _ in range(batch_size)], dim=0)
            proposal_class_loss_batch = torch.stack([loss_dict['loss_classifier'] for _ in range(batch_size)], dim=0)
            proposal_transformer_loss_batch = torch.stack([loss_dict['loss_box_reg'] for _ in range(batch_size)], dim=0)

            return (anchor_objectness_loss_batch, anchor_transformer_loss_batch,
                    proposal_class_loss_batch, proposal_transformer_loss_batch)
        else:
            out_list = self.net(padded_image_batch)

            detection_bboxes_batch = [out['boxes'] for out in out_list]
            detection_classes_batch = [out['labels'] for out in out_list]
            detection_probs_batch = [out['scores'] for out in out_list]

            # For compatibility with the interface
            anchor_bboxes_batch, proposal_bboxes_batch, proposal_probs_batch = [], [], []

            return (anchor_bboxes_batch, proposal_bboxes_batch, proposal_probs_batch,
                    detection_bboxes_batch, detection_classes_batch, detection_probs_batch)

    def remove_output_modules(self):
        del self.net.rpn.head.cls_logits
        del self.net.rpn.head.bbox_pred
        del self.net.roi_heads.box_predictor.cls_score
        del self.net.roi_heads.box_predictor.bbox_pred

    class IdentityTransform(nn.Module):

        def forward(self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None
                    ) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
            image_sizes = [img.shape[-2:] for img in images]
            images = torch.stack(images, dim=0)

            image_sizes_list: List[Tuple[int, int]] = []
            for image_size in image_sizes:
                assert len(image_size) == 2
                image_sizes_list.append((image_size[0], image_size[1]))

            image_list = ImageList(images, image_sizes_list)
            return image_list, targets

        def postprocess(self, result: List[Dict[str, Tensor]],
                        image_shapes: List[Tuple[int, int]],
                        original_image_sizes: List[Tuple[int, int]]
                        ) -> List[Dict[str, Tensor]]:
            if self.training:
                return result
            for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
                boxes = pred["boxes"]
                boxes = resize_boxes(boxes, im_s, o_im_s)
                result[i]["boxes"] = boxes
            return result
