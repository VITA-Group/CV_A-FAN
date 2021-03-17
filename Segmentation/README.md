## Deeplabv3+ for Segmentation (w/w.o. A-FAN)

### Requirements

Environment reference: https://github.com/VainF/DeepLabV3Plus-Pytorch

- Install

`cd datasets/data`

`ln -s /path/to/cityscapes`

`pip install torch torchvision`

`pip install visdom`

`pip install advertorch`

`pip install tqdm`

`pip install tensorboard`

`pip install sklearn`

`pip install matplotlib`

### How to Run?

- Resnet50 backbone on VOC 2012 dataset

`bash ./sh/voc/2012/clean50/01_final50.sh 0`

- Resnet50 backbone on Cityscapes dataset

`bash ./sh/city/clean50/091_city_final01.sh`

### The Detailed Commands

- Train


```
GPU=0
EXP=EXP01
SELAYER=3     # perturbation in layer 3
SDLAYER=aspp  # perturbation in encoder layer aspp
GAMMASD=0.4   # perturbation strength in decoder
AdvWeight=0.3 # adv loss weight
GAMMASE=0.01  # perturbation strength in backbone
MIX=11        # mix feature

python -u main_aug_final.py --year 2012 --crop_val --batch_size 4 \
--model deeplabv3plus_resnet50 \
--pertub_idx_sd ${SDLAYER} \
--pertub_idx_se ${SELAYER} \
--adv_loss_weight_sd ${AdvWeight} \
--gamma_se ${GAMMASE} \
--gamma_sd ${GAMMASD} \
--gpu_id ${GPU} \
--mix_layer ${MIX} \
${EXP}
```