GPU=${1}
python -u main_ori.py \
--year 2012 \
--crop_val \
--model deeplabv3plus_resnet101 \
--gpu_id ${GPU} \
--batch_size 4 \
--random_seed 66 \
baseline_voc2012_resnet101_bs4_2gpus

# python -u main_ori.py \
# --year 2012 \
# --crop_val \
# --model deeplabv3plus_resnet101 \
# --gpu_id ${GPU} \
# --batch_size 4 \
# --random_seed 37 \
# baseline_voc2012_resnet101_bs4_seed37