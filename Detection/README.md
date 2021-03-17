## Faster-RCNN for Detection (w/w.o. A-FAN)

### Requirements

Environment reference: https://github.com/potterhsu/easy-faster-rcnn.pytorch

- Install

`conda install pytorch==1.0.0 torchvision==0.2.1 cuda100 -c pytorch`

`pip install tqdm`

`pip install tensorboardX`

`pip install advertorch`

`pip install pycocotools` 

`pip install opencv-python~=3.4`

`pip install websockets`

- Build

1. install

`python support/setup.py develop`

2. test

`python test/nms/test_nms.py`

If there is no bug, go next step.

- Dataset:

1. Download data, then:

`mkdir coco`

`mv annotations coco`

`mv train2017 coco`

`mv val2017 coco`

2. Put the COCO dataset dir link to **This Repo**

`mkdir data`

`cd data`

`ln -s path/to/coco COCO`

Please note that COCO is capital letter.

### How to Run?

- Resnet50 backbone on VOC dataset

`bash ./sh/voc2007/clean50/090_final_setting1.sh 0,1`

- Resnet50 backbone on COCO dataset

`bash ./sh/coco/clean50/090_final_setting1_gpu01.sh`

### The Detailed Commands

- Train

```
GPU=0
OUTPUTS_DIR=EXP01
GAMMA_se=1.0   # gamma in backbone
LAYER=2        # perturbation backbone layer
MASK=0011      # feature
GAMMA_sd=0.1   # gamma in detection head
AdvWeight=0.3  # adv loss weight

CUDA_VISIBLE_DEVICES=${GPU} python -u train_aug_final.py -s=voc2007 -b=resnet50 -o=${OUTPUTS_DIR} \
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

```

- Eval

```
CUDA_VISIBLE_DEVICES=0 python eval.py -s=voc2007 -b=resnet50 \
./${OUTPUTS_DIR}/ckpt-s1se_${LAYER}_sd_roi_g${GAMMA_se}e2_MIX${MASK}-voc2007-resnet50/model-11250.pth
```






