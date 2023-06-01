#!/usr/bin/env bash
python ./src/aibox_vision/api/train.py \
            -o=./outputs/tests \
            -d=./data/CIFAR-10 \
            --visible_devices="[0]" \
            --needs_freeze_bn=True \
            --batch_size=8 \
            --learning_rate=0.008 \
            --step_lr_sizes="[6, 8]" \
            --num_epochs_to_validate=10 \
            --num_epochs_to_finish=10 \
            classification \
            --algorithm=efficientnet_b7