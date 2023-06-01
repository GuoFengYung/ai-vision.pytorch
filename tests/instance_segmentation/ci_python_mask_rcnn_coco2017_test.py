import os
import sys

path_to_python = sys.executable

assert len(sys.argv) == 2
path_to_coco2017_data_dir = sys.argv[1]

num_workers = 2
visible_devices = [0, 1]
batch_size = 4
learning_rate = batch_size * 0.00125

script = f'''{path_to_python} ./src/aibox_vision/api/train.py \
-o=./outputs/tests \
-d={path_to_coco2017_data_dir} \
--num_workers={num_workers} \
--visible_devices="{str(visible_devices)}" \
--batch_size={batch_size} \
--learning_rate={learning_rate} \
--weight_decay=0.0001 \
--step_lr_sizes="[8, 11]" \
--num_batches_to_display=100 \
--num_epochs_to_validate=13 \
--num_epochs_to_finish=13 \
instance_segmentation \
--algorithm=mask_rcnn
'''

print(script)
exit_code = os.system(script)
exit(int(exit_code != 0))
