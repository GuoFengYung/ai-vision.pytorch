import os
import random
import sys

from aibox_vision.lib.task.detection.backbone import Backbone

path_to_python = sys.executable

num_workers = random.choice([0, 1, 2])
visible_devices = random.choice([[0], [1], [0, 1]])
num_visible_devices = len(visible_devices)
backbone = random.choice(Backbone.OPTIONS)

max_batch_size_per_device_to_backbones_dict = {
    4: [
        Backbone.Name.RESNET18.value,
        Backbone.Name.RESNET34.value,
        Backbone.Name.RESNET50.value,
        Backbone.Name.RESNET101.value,
        Backbone.Name.RESNEST50.value,
        Backbone.Name.RESNEST101.value,
        Backbone.Name.WIDE_RESNET50_2.value,
        Backbone.Name.WIDE_RESNET101_2.value,
        Backbone.Name.MOBILENET_V3_SMALL.value,
        Backbone.Name.MOBILENET_V3_LARGE.value
    ],
    2: [
        Backbone.Name.RESNET152.value,
        Backbone.Name.RESNEXT50_32X4D.value,
        Backbone.Name.RESNEXT101_32X8D.value,
        Backbone.Name.SE_RESNEXT50_32X4D.value,
        Backbone.Name.SE_RESNEXT101_32X4D.value,
        Backbone.Name.RESNEST200.value
    ],
    1: [
        Backbone.Name.SENET154.value,
        Backbone.Name.NASNET_A_LARGE.value,
        Backbone.Name.PNASNET_5_LARGE.value,
        Backbone.Name.RESNEST269.value
    ]
}

for max_batch_size_per_device, backbones in max_batch_size_per_device_to_backbones_dict.items():
    if backbone in backbones:
        max_batch_size = max_batch_size_per_device * num_visible_devices
        break
else:
    raise ValueError(f'Unknown backbone: {backbone}')

batch_size = random.randrange(1, max_batch_size + 1)
learning_rate = batch_size * 0.001

script = f'''{path_to_python} ./src/aibox_vision/api/train.py \
-o=./outputs/tests \
-d=./examples/CatDog \
--num_workers={num_workers} \
--visible_devices="{str(visible_devices)}" \
--batch_size={batch_size} \
--learning_rate={learning_rate} \
--step_lr_sizes="[10, 14]" \
--num_epochs_to_finish=16 \
detection \
--algorithm=faster_rcnn \
--backbone={backbone}
'''

print(script)
exit_code = os.system(script)
exit(int(exit_code != 0))
