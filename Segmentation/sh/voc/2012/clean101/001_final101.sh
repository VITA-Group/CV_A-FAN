GPU=${1}
EXP=FINAL_101_110401
SELAYER=3
SDLAYER=concat
GAMMASD=0.4
AdvWeight=0.1

# GAMMASE=0.01
# MIX=11
# python -u main_aug_final.py --year 2012 --crop_val --batch_size 4 \
# --model deeplabv3plus_resnet101 \
# --pertub_idx_sd ${SDLAYER} \
# --pertub_idx_se ${SELAYER} \
# --adv_loss_weight_sd ${AdvWeight} \
# --gamma_se ${GAMMASE} \
# --gamma_sd ${GAMMASD} \
# --gpu_id ${GPU} \
# --mix_layer ${MIX} \
# ${EXP}

EXP=FINAL_101_110402
GAMMASE=0.01
MIX=01
python -u main_aug_final.py --year 2012 --crop_val --batch_size 4 \
--model deeplabv3plus_resnet101 \
--pertub_idx_sd ${SDLAYER} \
--pertub_idx_se ${SELAYER} \
--adv_loss_weight_sd ${AdvWeight} \
--gamma_se ${GAMMASE} \
--gamma_sd ${GAMMASD} \
--gpu_id ${GPU} \
--mix_layer ${MIX} \
--continue_training \
--ckpt ./checkpoints/voc_FINAL_101_110402_selayer_3_sdlayer_concat_gamma_se0.01_gamma_sd0.4_advweight0.1MIX01/latest_deeplabv3plus_resnet101_voc_os16.pth \
${EXP}
