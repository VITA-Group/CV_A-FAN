GPU=${1}
EXP=FINAL_110401
SDLAYER=concat
GAMMASD=0.4
AdvWeight=0.1

SELAYER=2
GAMMASE=0.03
MIX=01
python -u main_aug_final.py --year 2007 --crop_val --batch_size 4 \
--total_itrs 15000 \
--pertub_idx_sd ${SDLAYER} \
--pertub_idx_se ${SELAYER} \
--adv_loss_weight_sd ${AdvWeight} \
--gamma_se ${GAMMASE} \
--gamma_sd ${GAMMASD} \
--gpu_id ${GPU} \
--mix_layer ${MIX} \
${EXP}

SELAYER=2
GAMMASE=0.03
MIX=11
python -u main_aug_final.py --year 2007 --crop_val --batch_size 4 \
--total_itrs 15000 \
--pertub_idx_sd ${SDLAYER} \
--pertub_idx_se ${SELAYER} \
--adv_loss_weight_sd ${AdvWeight} \
--gamma_se ${GAMMASE} \
--gamma_sd ${GAMMASD} \
--gpu_id ${GPU} \
--mix_layer ${MIX} \
${EXP}

SELAYER=2
GAMMASE=0.03
MIX=00
python -u main_aug_final.py --year 2007 --crop_val --batch_size 4 \
--total_itrs 15000 \
--pertub_idx_sd ${SDLAYER} \
--pertub_idx_se ${SELAYER} \
--adv_loss_weight_sd ${AdvWeight} \
--gamma_se ${GAMMASE} \
--gamma_sd ${GAMMASD} \
--gpu_id ${GPU} \
--mix_layer ${MIX} \
${EXP}