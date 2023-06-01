import os
import sys

path_to_python = sys.executable

assert len(sys.argv) == 2
path_to_cifar100_data_dir = sys.argv[1]

num_workers = 2
visible_devices = [0, 1]
needs_freeze_bn = True
batch_size = 32
learning_rate = batch_size * 0.001

script = f'''{path_to_python} ./src/aibox_vision/api/train.py \
-o=./outputs/tests \
-d={path_to_cifar100_data_dir} \
--num_workers={num_workers} \
--visible_devices="{str(visible_devices)}" \
--needs_freeze_bn={needs_freeze_bn} \
--batch_size={batch_size} \
--learning_rate={learning_rate} \
--step_lr_sizes="[6, 8]" \
--num_batches_to_display=100 \
--num_epochs_to_validate=10 \
--num_epochs_to_finish=10 \
classification \
--algorithm=resnet101
'''

print(script)
exit_code = os.system(script)
exit(int(exit_code != 0))
