GPU=${1}
OUTPUTS_DIR=voc_final101_ckpt02

# SE setting : Layer:2, MIX:0011,  Gamma:1.0
# SD setting1: AdvWeight:0.3, ROI, Gamma:0.1
# SD setting2: AdvWeight:0.3, ROI, Gamma:0.05
# SD setting3: AdvWeight:0.3, MIX, Gamma:0.2
# SD setting4: AdvWeight:0.5, MIX, Noise:0.01

# SE:
GAMMA_se=1.0
LAYER=2
MASK=0011
# SD:
GAMMA_sd=0.05
AdvWeight=0.3
# --mix_sd
# --noise_sd 0.01
# --only_roi_sd
# --sd_adv_loss_weight 0.3
CUDA_VISIBLE_DEVICES=${GPU} python -u train_aug_final.py -s=voc2007 -b=resnet101 -o=${OUTPUTS_DIR} \
--batch_size=8 \
--learning_rate=0.008 \
--step_lr_sizes="[6250, 8750]" \
--num_steps_to_snapshot=1250 \
--num_steps_to_finish=11250 \
--mix_layer ${MASK} \
--pertub_idx_se ${LAYER} \
--gamma_se ${GAMMA_se} \
--gamma_sd ${GAMMA_sd} \
--sd_adv_loss_weight ${AdvWeight} \
--only_roi_sd

CUDA_VISIBLE_DEVICES=${GPU} python eval.py -s=voc2007 -b=resnet50 \
./${OUTPUTS_DIR}/ckpt-s1se_${LAYER}_sd_roi_g${GAMMA_se}e2_MIX${MASK}-voc2007-resnet101/model-11250.pth