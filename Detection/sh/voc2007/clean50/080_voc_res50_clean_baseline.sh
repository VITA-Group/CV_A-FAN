GPU=${1}
OUTPUTS_DIR=voc50_clean_baseline_loss_4GPUS

# CUDA_VISIBLE_DEVICES=${GPU} python -u train_baseline.py -s=voc2007 -b=resnet50 -o=${OUTPUTS_DIR} \
#  --batch_size=8 \
#  --learning_rate=0.008 \
#  --step_lr_sizes="[6250, 8750]" \
#  --num_steps_to_snapshot=1250 \
#  --num_steps_to_finish=11250

CUDA_VISIBLE_DEVICES=${GPU} python eval.py -s=voc2007 -b=resnet50 \
voc50_clean_baseline_loss_4GPUS/checkpoints-voc2007-resnet50_baseline/model-11250.pth