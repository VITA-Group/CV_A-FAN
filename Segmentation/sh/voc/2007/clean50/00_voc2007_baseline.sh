GPU=$1
python -u main_ori.py \
--year 2007 \
--crop_val \
--batch_size 4 \
--gpu_id ${GPU} \
--total_itrs 15000 \
--random_seed 66 \
baseline_voc2007_bs4_seed66 \

python -u main_ori.py \
--year 2007 \
--crop_val \
--batch_size 4 \
--gpu_id ${GPU} \
--total_itrs 15000 \
--random_seed 37 \
baseline_voc2007_bs4_seed37 \

python -u main_ori.py \
--year 2007 \
--crop_val \
--batch_size 4 \
--gpu_id ${GPU} \
--total_itrs 15000 \
--random_seed 17 \
baseline_voc2007_bs4_seed17 \