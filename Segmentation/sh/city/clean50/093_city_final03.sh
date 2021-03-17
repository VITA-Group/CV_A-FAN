GPU=${1}
EXP=FINAL05
SDLAYER=concat
SELAYER=2
AdvWeight=0.3
GAMMASE=0.01
GAMMASD=1.5
MIX=11

python -u main_aug_final.py \
--model deeplabv3plus_resnet50 \
--data_root ./datasets/data/cityscapes \
--dataset cityscapes \
--lr 0.1 \
--crop_size 768 \
--batch_size 4 \
--pertub_idx_sd ${SDLAYER} \
--pertub_idx_se ${SELAYER} \
--adv_loss_weight_sd ${AdvWeight} \
--gamma_se ${GAMMASE} \
--gamma_sd ${GAMMASD} \
--gpu_id ${GPU} \
--mix_layer ${MIX} \
--mix_sd \
${EXP}

EXP=FINAL06
SDLAYER=concat
SELAYER=2
AdvWeight=0.3
GAMMASE=0.01
GAMMASD=1.5
MIX=10

python -u main_aug_final.py \
--model deeplabv3plus_resnet50 \
--data_root ./datasets/data/cityscapes \
--dataset cityscapes \
--lr 0.1 \
--crop_size 768 \
--batch_size 4 \
--pertub_idx_sd ${SDLAYER} \
--pertub_idx_se ${SELAYER} \
--adv_loss_weight_sd ${AdvWeight} \
--gamma_se ${GAMMASE} \
--gamma_sd ${GAMMASD} \
--gpu_id ${GPU} \
--mix_layer ${MIX} \
--mix_sd \
${EXP}