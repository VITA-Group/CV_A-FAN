GPU=$1
EVAL=FinalModels/00_final_tlc/voc2007_resnet50_final_7538.pth

CUDA_VISIBLE_DEVICES=${GPU} python infer_all.py -s=voc2007 -b=resnet50 -c=${EVAL} ./data/VOCdevkit/VOC2007/JPEGImages ./02_Our_method
# python infer.py -s=voc2007 -b=resnet101 -c=${EVAL} ./data/VOCdevkit/VOC2007/JPEGImages/000240.jpg ./vis_out/000240.jpg
# python infer.py -s=voc2007 -b=resnet101 -c=${EVAL} ./data/VOCdevkit/VOC2007/JPEGImages/001370.jpg ./vis_out/001370.jpg
# python infer.py -s=voc2007 -b=resnet101 -c=${EVAL} ./data/VOCdevkit/VOC2007/JPEGImages/002628.jpg ./vis_out/002628.jpg
# python infer.py -s=voc2007 -b=resnet101 -c=${EVAL} ./data/VOCdevkit/VOC2007/JPEGImages/006199.jpg ./vis_out/006199.jpg