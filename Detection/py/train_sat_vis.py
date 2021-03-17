import argparse
import os
import time
import uuid
from collections import deque
from typing import Optional

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader

from backbone.base import Base as BackboneBase
from config.train_config import TrainConfig as Config
from dataset.base import Base as DatasetBase
from extension.lr_scheduler import WarmUpMultiStepLR
from logger import Logger as Log
from model import Model
# from model_advlayer4 import Model
from roi.pooler import Pooler
from attack_algo import *
import attack_algo
import numpy as np
import random

import pdb

def _train(dataset_name: str, backbone_name: str, path_to_data_dir: str, path_to_checkpoints_dir: str, path_to_resuming_checkpoint: Optional[str], args):

    # Setup random seed
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    
    print("DATASET:[{}] DIR:[{}]".format(dataset_name, path_to_data_dir))
    dataset = DatasetBase.from_name(dataset_name)(path_to_data_dir, DatasetBase.Mode.TRAIN, Config.IMAGE_MIN_SIDE, Config.IMAGE_MAX_SIDE)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE,
                            sampler=DatasetBase.NearestRatioRandomSampler(dataset.image_ratios, num_neighbors=Config.BATCH_SIZE),
                            num_workers=8, collate_fn=DatasetBase.padding_collate_fn, pin_memory=True)

    Log.i('Found {:d} samples'.format(len(dataset)))

    backbone = BackboneBase.from_name(backbone_name)(pretrained=True)
    model = nn.DataParallel(
        Model(
            backbone, dataset.num_classes(), pooler_mode=Config.POOLER_MODE,
            anchor_ratios=Config.ANCHOR_RATIOS, anchor_sizes=Config.ANCHOR_SIZES,
            rpn_pre_nms_top_n=Config.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=Config.RPN_POST_NMS_TOP_N,
            anchor_smooth_l1_loss_beta=Config.ANCHOR_SMOOTH_L1_LOSS_BETA, proposal_smooth_l1_loss_beta=Config.PROPOSAL_SMOOTH_L1_LOSS_BETA
        ).cuda()
    )
    
    optimizer = optim.SGD(model.parameters(), lr=Config.LEARNING_RATE,
                          momentum=Config.MOMENTUM, weight_decay=Config.WEIGHT_DECAY)
    scheduler = WarmUpMultiStepLR(optimizer, milestones=Config.STEP_LR_SIZES, gamma=Config.STEP_LR_GAMMA,
                                  factor=Config.WARM_UP_FACTOR, num_iters=Config.WARM_UP_NUM_ITERS)
    step = 0
    time_checkpoint = time.time()
    losses = deque(maxlen=100)
    summary_writer = SummaryWriter(os.path.join(path_to_checkpoints_dir, 'summaries'))
    should_stop = False

    num_steps_to_display = Config.NUM_STEPS_TO_DISPLAY
    num_steps_to_snapshot = Config.NUM_STEPS_TO_SNAPSHOT
    num_steps_to_finish = Config.NUM_STEPS_TO_FINISH

    if path_to_resuming_checkpoint is not None:
        step = model.module.load(path_to_resuming_checkpoint, optimizer, scheduler)
        Log.i(f'Model has been restored from file: {path_to_resuming_checkpoint}')

    device_count = torch.cuda.device_count()
    assert Config.BATCH_SIZE % device_count == 0, 'The batch size is not divisible by the device count'
    Log.i('Start training with {:d} GPUs ({:d} batches per GPU)'.format(torch.cuda.device_count(),
                                                                        Config.BATCH_SIZE // torch.cuda.device_count()))

    
    for iters, (_, image_batch, _, bboxes_batch, labels_batch) in enumerate(dataloader):

        batch_size = image_batch.shape[0]
        image_batch = image_batch.cuda()
        bboxes_batch = bboxes_batch.cuda()
        labels_batch = labels_batch.cuda()
        
        inputs_all1 = {"x": image_batch, "adv": None, "out_idx": 1, "flag": 'head'}
        inputs_all2 = {"x": image_batch, "adv": None, "out_idx": 2, "flag": 'head'}
        inputs_all3 = {"x": image_batch, "adv": None, "out_idx": 3, "flag": 'head'}

        feature_map1 = model.train().forward(inputs_all1, bboxes_batch, labels_batch)
        feature_map2 = model.train().forward(inputs_all2, bboxes_batch, labels_batch)
        feature_map3 = model.train().forward(inputs_all3, bboxes_batch, labels_batch)

        gamma = 50
        feature_adv1 = attack_algo.PGD(feature_map1, image_batch, 
            y = {'bb':bboxes_batch, 'lb': labels_batch},
            model= model,
            steps = args.steps,
            eps = (args.eps / 255), 
            gamma = (gamma / 255),
            idx = 1,
            randinit = args.randinit,
            clip = args.clip)
        
        feature_adv2 = attack_algo.PGD(feature_map2, image_batch, 
            y = {'bb':bboxes_batch, 'lb': labels_batch},
            model= model,
            steps = args.steps,
            eps = (args.eps / 255), 
            gamma = (gamma / 255),
            idx = 2,
            randinit = args.randinit,
            clip = args.clip)

        feature_adv3 = attack_algo.PGD(feature_map3, image_batch, 
            y = {'bb':bboxes_batch, 'lb': labels_batch},
            model= model,
            steps = args.steps,
            eps = (args.eps / 255), 
            gamma = (gamma / 255),
            idx = 3,
            randinit = args.randinit,
            clip = args.clip)
        
    
        adv_list_layer1 = attack_algo.get_sample_points(feature_map1, feature_adv1, 5)
        adv_list_layer2 = attack_algo.get_sample_points(feature_map2, feature_adv2, 5)
        adv_list_layer3 = attack_algo.get_sample_points(feature_map3, feature_adv3, 5)

        mix_adv_list_layer1 = []
        mix_adv_list_layer2 = []
        mix_adv_list_layer3 = []

        for i in range(4):

            adv_feature_map1 = attack_algo.mix_feature(feature_map1, adv_list_layer1[i + 1])
            mix_adv_list_layer1.append(adv_feature_map1)
            adv_feature_map2 = attack_algo.mix_feature(feature_map2, adv_list_layer2[i + 1])
            mix_adv_list_layer2.append(adv_feature_map2)
            adv_feature_map3 = attack_algo.mix_feature(feature_map3, adv_list_layer3[i + 1])
            mix_adv_list_layer3.append(adv_feature_map3)
        
        
        adv_list_layer1 = [i.squeeze()[0].cpu().detach().numpy() for i in adv_list_layer1]
        adv_list_layer2 = [i.squeeze()[0].cpu().detach().numpy() for i in adv_list_layer2]
        adv_list_layer3 = [i.squeeze()[0].cpu().detach().numpy() for i in adv_list_layer3]

        mix_adv_list_layer1 = [i.squeeze()[0].cpu().detach().numpy() for i in mix_adv_list_layer1]
        mix_adv_list_layer2 = [i.squeeze()[0].cpu().detach().numpy() for i in mix_adv_list_layer2]
        mix_adv_list_layer3 = [i.squeeze()[0].cpu().detach().numpy() for i in mix_adv_list_layer3]


        print(iters)

        attack_algo.img_save(image_batch[0], save_name='img2/img{}.png'.format(iters))

        # for i, j in enumerate(adv_list_layer1):
        #     attack_algo.feature_map_save(j,  save_name='img3/img{}_layer1_sat{}.png'.format(iters, i))
            
        print("finish 1")
        for i, j in enumerate(mix_adv_list_layer1):
            attack_algo.feature_map_save(j, save_name='img2/img{}_layer1_sat{}_mix.png'.format(iters, i + 1))

        print("finish 1_mix")
        # for i, j in enumerate(adv_list_layer2):
        #     attack_algo.feature_map_save(j, save_name='img3/img{}_layer2_sat{}.png'.format(iters, i))

        print("finish 2")
        for i, j in enumerate(mix_adv_list_layer2):
            attack_algo.feature_map_save(j, save_name='img2/img{}_layer2_sat{}_mix.png'.format(iters, i + 1))

        print("finish 2_mix")
        # for i, j in enumerate(adv_list_layer3):
        #     attack_algo.feature_map_save(j, save_name='img3/img{}_layer3_sat{}.png'.format(iters, i))

        print("finish 3")
        for i, j in enumerate(mix_adv_list_layer3):
            attack_algo.feature_map_save(j, save_name='img2/img{}_layer3_sat{}_mix.png'.format(iters, i + 1))

        print("finish 3_mix")

        if iters == 5:
            return
        
if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        #PGD Setting 

        parser.add_argument("--random_seed", type=int, default=1, help="random seed (default: 1)")
        parser.add_argument('--steps', default=1, type=int, help='PGD-steps')
        parser.add_argument('--pertub_idx', help='index of perturb layers', default=3, type=int)
        parser.add_argument('--gamma', help='index of PGD gamma', default=0.5, type=float)
        parser.add_argument('--eps', default=2, type=float)
        parser.add_argument('--randinit', action="store_true", help="whether using apex")
        parser.add_argument('--clip', action="store_true", help="whether using apex")
        parser.add_argument('-s', '--dataset', type=str, choices=DatasetBase.OPTIONS, required=True, help='name of dataset')
        parser.add_argument('-b', '--backbone', type=str, choices=BackboneBase.OPTIONS, required=True, help='name of backbone model')
        parser.add_argument('-d', '--data_dir', type=str, default='./data', help='path to data directory')
        parser.add_argument('-o', '--outputs_dir', type=str, default='./outputs', help='path to outputs directory')
        parser.add_argument('-r', '--resume_checkpoint', type=str, help='path to resuming checkpoint')
        parser.add_argument('--image_min_side', type=float, help='default: {:g}'.format(Config.IMAGE_MIN_SIDE))
        parser.add_argument('--image_max_side', type=float, help='default: {:g}'.format(Config.IMAGE_MAX_SIDE))
        parser.add_argument('--anchor_ratios', type=str, help='default: "{!s}"'.format(Config.ANCHOR_RATIOS))
        parser.add_argument('--anchor_sizes', type=str, help='default: "{!s}"'.format(Config.ANCHOR_SIZES))
        parser.add_argument('--pooler_mode', type=str, choices=Pooler.OPTIONS, help='default: {.value:s}'.format(Config.POOLER_MODE))
        parser.add_argument('--rpn_pre_nms_top_n', type=int, help='default: {:d}'.format(Config.RPN_PRE_NMS_TOP_N))
        parser.add_argument('--rpn_post_nms_top_n', type=int, help='default: {:d}'.format(Config.RPN_POST_NMS_TOP_N))
        parser.add_argument('--anchor_smooth_l1_loss_beta', type=float, help='default: {:g}'.format(Config.ANCHOR_SMOOTH_L1_LOSS_BETA))
        parser.add_argument('--proposal_smooth_l1_loss_beta', type=float, help='default: {:g}'.format(Config.PROPOSAL_SMOOTH_L1_LOSS_BETA))
        parser.add_argument('--batch_size', type=int, help='default: {:g}'.format(Config.BATCH_SIZE))
        parser.add_argument('--learning_rate', type=float, help='default: {:g}'.format(Config.LEARNING_RATE))
        parser.add_argument('--momentum', type=float, help='default: {:g}'.format(Config.MOMENTUM))
        parser.add_argument('--weight_decay', type=float, help='default: {:g}'.format(Config.WEIGHT_DECAY))
        parser.add_argument('--step_lr_sizes', type=str, help='default: {!s}'.format(Config.STEP_LR_SIZES))
        parser.add_argument('--step_lr_gamma', type=float, help='default: {:g}'.format(Config.STEP_LR_GAMMA))
        parser.add_argument('--warm_up_factor', type=float, help='default: {:g}'.format(Config.WARM_UP_FACTOR))
        parser.add_argument('--warm_up_num_iters', type=int, help='default: {:d}'.format(Config.WARM_UP_NUM_ITERS))
        parser.add_argument('--num_steps_to_display', type=int, help='default: {:d}'.format(Config.NUM_STEPS_TO_DISPLAY))
        parser.add_argument('--num_steps_to_snapshot', type=int, help='default: {:d}'.format(Config.NUM_STEPS_TO_SNAPSHOT))
        parser.add_argument('--num_steps_to_finish', type=int, help='default: {:d}'.format(Config.NUM_STEPS_TO_FINISH))
        args = parser.parse_args()
        print_args(args, 100)
        print("INFO: PID   : [{}]".format(os.getpid()))

        dataset_name = args.dataset
        backbone_name = args.backbone

        exp_name = "s" + str(args.steps) + "p" + str(args.pertub_idx) + "g" + str(args.gamma) + "e" + str(args.eps)
        if args.randinit: exp_name += "_rand"
        if args.clip: exp_name += "_clip"

        path_to_data_dir = args.data_dir
        path_to_outputs_dir = args.outputs_dir
        path_to_resuming_checkpoint = args.resume_checkpoint

        path_to_checkpoints_dir = os.path.join(path_to_outputs_dir, 'ckpt-{:s}-{:s}-{:s}'
            .format(exp_name, dataset_name, backbone_name))
        if not os.path.exists(path_to_checkpoints_dir):
            print("create dir:[{}]".format(path_to_checkpoints_dir))
            os.makedirs(path_to_checkpoints_dir)

        Config.setup(image_min_side=args.image_min_side, image_max_side=args.image_max_side,
                     anchor_ratios=args.anchor_ratios, anchor_sizes=args.anchor_sizes, pooler_mode=args.pooler_mode,
                     rpn_pre_nms_top_n=args.rpn_pre_nms_top_n, rpn_post_nms_top_n=args.rpn_post_nms_top_n,
                     anchor_smooth_l1_loss_beta=args.anchor_smooth_l1_loss_beta, proposal_smooth_l1_loss_beta=args.proposal_smooth_l1_loss_beta,
                     batch_size=args.batch_size, learning_rate=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay,
                     step_lr_sizes=args.step_lr_sizes, step_lr_gamma=args.step_lr_gamma,
                     warm_up_factor=args.warm_up_factor, warm_up_num_iters=args.warm_up_num_iters,
                     num_steps_to_display=args.num_steps_to_display, num_steps_to_snapshot=args.num_steps_to_snapshot, num_steps_to_finish=args.num_steps_to_finish)

        Log.initialize(os.path.join(path_to_checkpoints_dir, 'train.log'))
        Log.i('Arguments:')
        for k, v in vars(args).items():
            Log.i(f'\t{k} = {v}')
        Log.i(Config.describe())

        print("-" * 100)
        print("INFO: PID   : [{}]".format(os.getpid()))
        print("EXP: Dataset:[{}] ADV Step:[{}] Layer:[{}] Gamma:[{}] Eps:[{}] Randinit:[{}] Clip:[{}]"
            .format(args.dataset, args.steps, args.pertub_idx, args.gamma, args.eps, args.randinit, args.clip))
        print("-" * 100)

        _train(dataset_name, backbone_name, path_to_data_dir, path_to_checkpoints_dir, path_to_resuming_checkpoint, args)

    main()
