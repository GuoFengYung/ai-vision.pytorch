import os
import random
import sys


path_to_python = sys.executable

num_workers = random.choice([0, 1, 2])
visible_devices = random.choice([[0], [1], [0, 1]])
batch_size = random.randrange(1, 5)
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
instance_segmentation \
--algorithm=mask_rcnn
'''

print(script)
exit_code = os.system(script)
exit(int(exit_code != 0))
