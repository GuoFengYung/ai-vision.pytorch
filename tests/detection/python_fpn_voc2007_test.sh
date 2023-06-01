#!/usr/bin/env bash
python ./src/aibox_vision/api/train.py \
            -o=./outputs/tests \
            -d=./data/VOC2007 \
            --visible_devices="[0]" \
            --batch_size=2 \
            --learning_rate=0.002 \
            --step_lr_sizes="[10, 14]" \
            --num_epochs_to_validate=16 \
            --num_epochs_to_finish=16 \
            detection \
            --algorithm=fpn \
            --backbone=resnet50