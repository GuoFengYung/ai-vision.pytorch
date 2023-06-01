import os
import random
import sys

from aibox_vision.lib.task.classification.algorithm import Algorithm

path_to_python = sys.executable

num_workers = random.choice([0, 1, 2])
visible_devices = random.choice([[0], [1], [0, 1]])
num_visible_devices = len(visible_devices)
algorithm = random.choice(Algorithm.OPTIONS)

max_batch_size_per_device_to_algorithms_dict = {
    4: [
        Algorithm.Name.MOBILENET_V2.value,
        Algorithm.Name.GOOGLENET.value,
        Algorithm.Name.INCEPTION_V3.value,
        Algorithm.Name.RESNET18.value,
        Algorithm.Name.RESNET34.value,
        Algorithm.Name.RESNET50.value,
        Algorithm.Name.RESNET101.value,
        Algorithm.Name.EFFICIENTNET_B0.value,
        Algorithm.Name.EFFICIENTNET_B1.value,
        Algorithm.Name.EFFICIENTNET_B2.value,
        Algorithm.Name.RESNEST50.value,
        Algorithm.Name.RESNEST101.value,
        Algorithm.Name.REGNET_Y_400MF.value
    ],
    2: [
        Algorithm.Name.EFFICIENTNET_B3.value,
        Algorithm.Name.EFFICIENTNET_B4.value,
        Algorithm.Name.EFFICIENTNET_B5.value,
        Algorithm.Name.RESNEST200.value
    ],
    1: [
        Algorithm.Name.EFFICIENTNET_B6.value,
        Algorithm.Name.EFFICIENTNET_B7.value,
        Algorithm.Name.RESNEST269.value
    ]
}

for max_batch_size_per_device, algorithms in max_batch_size_per_device_to_algorithms_dict.items():
    if algorithm in algorithms:
        max_batch_size = max_batch_size_per_device * num_visible_devices
        break
else:
    raise ValueError(f'Unknown algorithm: {algorithm}')

batch_size = random.randrange(1, max_batch_size + 1)
learning_rate = batch_size * 0.001
needs_freeze_bn = (batch_size / num_visible_devices) < 16

script = f'''{path_to_python} ./src/aibox_vision/api/train.py \
-o=./outputs/tests \
-d=./examples/CatDog \
--num_workers={num_workers} \
--visible_devices="{str(visible_devices)}" \
--needs_freeze_bn={needs_freeze_bn} \
--batch_size={batch_size} \
--learning_rate={learning_rate} \
--step_lr_sizes="[10, 14]" \
--num_epochs_to_finish=16 \
classification \
--algorithm={algorithm}
'''

print(script)
exit_code = os.system(script)
exit(int(exit_code != 0))
