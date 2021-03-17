#!/usr/bin/env bash
GPU=${1}
BACKBONE=resnet101
OUTPUTS_DIR=./coco101_baseline

CUDA_VISIBLE_DEVICES=${GPU} python -u train_baseline.py \
-s=coco2017 \
-b=${BACKBONE} \
-o=${OUTPUTS_DIR} \
--image_min_side=800 \
--image_max_side=1333 \
--anchor_sizes="[64, 128, 256, 512]" \
--anchor_smooth_l1_loss_beta=0.1111 \
--batch_size=8 \
--learning_rate=0.01 \
--weight_decay=0.0001 \
--step_lr_sizes="[120000, 160000]" \
--num_steps_to_snapshot=40000 \
--num_steps_to_finish=180000

# eval
CUDA_VISIBLE_DEVICES=${GPU} python eval.py -s=coco2017 -b=resnet50 \
--image_min_side=800 \
--image_max_side=1333 \
--anchor_sizes="[64, 128, 256, 512]" \
--rpn_post_nms_top_n=1000 \
${OUTPUTS_DIR}/ckpt-coco2017-resnet101_baseline/model-180000.pth