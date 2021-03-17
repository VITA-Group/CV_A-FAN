GPU=0,1
OUTPUTS_DIR=final_ckpt05

# SE setting : Layer:2, MIX:0011,  Gamma:1.0
# SD setting1: AdvWeight:0.3, ROI, Gamma:0.1
# SD setting2: AdvWeight:0.3, ROI, Gamma:0.05
# SD setting3: AdvWeight:0.3, MIX, Gamma:0.2
# SD setting4: AdvWeight:0.5, MIX, Gamma:0.3, Noise:0.01

# SE:
GAMMA_se=0.1
LAYER=2
MASK=1100
# SD:
GAMMA_sd=0.2
AdvWeight=0.3
# --mix_sd
# --noise_sd 0.01
# --only_roi_sd
# --sd_adv_loss_weight 0.3
# train
CUDA_VISIBLE_DEVICES=${GPU} python -u train_aug_final.py -s=coco2017 -b=resnet50 -o=${OUTPUTS_DIR} \
--image_min_side=800 \
--image_max_side=1333 \
--anchor_sizes="[64, 128, 256, 512]" \
--anchor_smooth_l1_loss_beta=0.1111 \
--batch_size=8 \
--learning_rate=0.01 \
--weight_decay=0.0001 \
--step_lr_sizes="[120000, 160000]" \
--num_steps_to_snapshot=40000 \
--num_steps_to_finish=180000 \
--mix_layer ${MASK} \
--pertub_idx_se ${LAYER} \
--gamma_se ${GAMMA_se} \
--gamma_sd ${GAMMA_sd} \
--sd_adv_loss_weight ${AdvWeight} \
--mix_sd

# eval
CUDA_VISIBLE_DEVICES=${GPU} python eval.py -s=coco2017 -b=resnet50 \
--image_min_side=800 \
--image_max_side=1333 \
--anchor_sizes="[64, 128, 256, 512]" \
--rpn_post_nms_top_n=1000 \
${OUTPUTS_DIR}/ckpt-s1se_${LAYER}_sd_roi_g${GAMMA_se}e2_MIX${MASK}-coco2017-resnet50/model-180000.pth