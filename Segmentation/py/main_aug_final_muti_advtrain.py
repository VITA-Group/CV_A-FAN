import network
import utils
import os
import random
import args
import numpy as np
import time
from torch.utils import data
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.visualizer import Visualizer
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import attack_algo
import pdb

def main():

    opts = args.get_argparser().parse_args()
    args.print_args(opts)

    f0, f1 = opts.mix_layer
    f0, f1 = int(f0), int(f1)
    
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19

    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset=='voc' and not opts.crop_val:
        opts.val_batch_size = 1
    
    train_dst, val_dst = args.get_dataset(opts)
    
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }

    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)
    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1*opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

    if opts.lr_policy=='poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy=='step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        
    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)
    opts.exp = opts.dataset.lower() + "_" + opts.exp + "_selayer_" + str(opts.pertub_idx_se) + "_sdlayer_" + str(opts.pertub_idx_sd) + "_gamma_se" + str(opts.gamma_se) + "_gamma_sd" + str(opts.gamma_sd) + "_advweight" + str(opts.adv_loss_weight_sd) + "MIX" + str(opts.mix_layer) 
    # opts.exp = opts.dataset.lower() + "_" + opts.exp + "_layer_"+ str(opts.pertub_idx_sd) + "_gamma" + str(opts.gamma_sd)
    print("INFO: Save dir:[{}]".format(opts.exp))
    writer = SummaryWriter(log_dir='runs/' + opts.exp)
    utils.mkdir('checkpoints/' + opts.exp)
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):

        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    #==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        model.eval()
        val_score, ret_samples = args.validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        return

    interval_loss = 0
    total_time = 0
    while True: #cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels) in train_loader:

            t0 = time.time()
            cur_itrs += 1
            
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            adv_images = attack_algo.adv_input(
                x = images, 
                criterion = criterion,
                y = labels,
                model= model,
                steps = opts.steps_pgd,
                eps = (opts.eps_pgd / 255), 
                gamma = (opts.gamma_pgd / 255),
                randinit = opts.randinit_pgd,
                clip = opts.randinit_pgd)

            inputs_all_se = {"x": images, "adv": None, "out_idx": opts.pertub_idx_se, "flag": 'head'}
            inputs_all_sd = {"x": images, "adv": None, "out_idx": opts.pertub_idx_sd + "_head", "flag": 'clean'}
            inputs_all_m1 = {"x": images, "adv": None, "out_idx": 1,                  "flag": 'head'}
            inputs_all_m3 = {"x": images, "adv": None, "out_idx": 3,                  "flag": 'head'}
            inputs_all_m4 = {"x": images, "adv": None, "out_idx": 4,                  "flag": 'head'}

            optimizer.zero_grad()

            output_dict_se = model(inputs_all_se)
            output_dict_m1 = model(inputs_all_m1)
            output_dict_m3 = model(inputs_all_m3)
            output_dict_m4 = model(inputs_all_m4)
            decoder_feature_map_dict = model(inputs_all_sd)

            feature_map_sd = decoder_feature_map_dict['adv'].detach()

            low_level_feat = output_dict_se["low_level"]
            feature_map_se = output_dict_se['out'].detach()
            feature_map_m1 = output_dict_m1['out'].detach()
            feature_map_m3 = output_dict_m3['out'].detach()
            feature_map_m4 = output_dict_m4['out'].detach()

            feature_adv_se = attack_algo.PGD(
                x = feature_map_se, 
                image_batch=images, 
                low_level_feat = low_level_feat,
                criterion=criterion,
                y = labels,
                model= model,
                steps = opts.steps,
                eps = (opts.eps / 255), 
                gamma = (opts.gamma_se / 255),
                idx = opts.pertub_idx_se,
                randinit = opts.randinit,
                clip = opts.clip)

            feature_adv_m1 = attack_algo.PGD(
                x = feature_map_m1, 
                image_batch=images, 
                low_level_feat = low_level_feat,
                criterion=criterion,
                y = labels,
                model= model,
                steps = 1,
                eps = (1.0 / 255), 
                gamma = (0.0001 / 255),
                idx = 1)
            
            feature_adv_m3 = attack_algo.PGD(
                x = feature_map_m3, 
                image_batch=images, 
                low_level_feat = low_level_feat,
                criterion=criterion,
                y = labels,
                model= model,
                steps = 1,
                eps = (1.0 / 255), 
                gamma = (0.0001 / 255),
                idx = 3)
            
            feature_adv_m4 = attack_algo.PGD(
                x = feature_map_m4, 
                image_batch=images, 
                low_level_feat = low_level_feat,
                criterion=criterion,
                y = labels,
                model= model,
                steps = 1,
                eps = (1.0 / 255), 
                gamma = (0.0001 / 255),
                idx = 4)

            feature_adv_sd_dict = attack_algo.decoder_PGD(
                input_dict=decoder_feature_map_dict, 
                image_batch=images, 
                criterion=criterion,
                y = labels,
                model= model,
                steps = opts.steps,
                eps = (opts.eps / 255), 
                gamma = (opts.gamma_sd / 255),
                idx = opts.pertub_idx_sd,
                randinit = opts.randinit,
                clip = opts.clip)

            adv_feature_map_sd = feature_adv_sd_dict['adv'].detach()
            if opts.mix_sd:
                adv_feature_map_sd = attack_algo.mix_feature(feature_map_sd, adv_feature_map_sd)
            if opts.noise_sd != 0:
                adv_feature_map_sd += (2.0 * torch.rand(adv_feature_map_sd.shape).cuda() - 1.0) * opts.gamma_sd * opts.noise_sd
            feature_adv_sd_dict['adv'] = adv_feature_map_sd

            adv_list_se = attack_algo.get_sample_points(feature_map_se, feature_adv_se, 3)
            if f0:
                adv_list_se[1] = attack_algo.mix_feature(feature_map_se, adv_list_se[1])
            if f1:
                adv_list_se[2] = attack_algo.mix_feature(feature_map_se, adv_list_se[2])
            
            clean_input_dict   = {'x': images, "adv": None,                'out_idx': 0,                  'flag':'clean'}
            adv_input_se_dict1 = {'x': images, 'adv': adv_list_se[1],      'out_idx': opts.pertub_idx_se, 'flag':'tail', 'low_level_feat': low_level_feat}
            adv_input_se_dict2 = {'x': images, 'adv': adv_list_se[2],      'out_idx': opts.pertub_idx_se, 'flag':'tail', 'low_level_feat': low_level_feat}
            adv_input_sd_dict  = {'x': images, 'adv': feature_adv_sd_dict, 'out_idx': opts.pertub_idx_sd + "_tail", 'flag':'clean'}

            adv_input_dict     = {'x': adv_images, "adv": None,            'out_idx': 0,                  'flag':'clean'}
            adv_input_m1_dict  = {'x': images, 'adv': feature_adv_m1,      'out_idx': 1,                  'flag':'tail', 'low_level_feat': low_level_feat}
            adv_input_m3_dict  = {'x': images, 'adv': feature_adv_m3,      'out_idx': 3,                  'flag':'tail', 'low_level_feat': low_level_feat}
            adv_input_m4_dict  = {'x': images, 'adv': feature_adv_m4,      'out_idx': 4,                  'flag':'tail', 'low_level_feat': low_level_feat}


            output0 = model(clean_input_dict)
            output1 = model(adv_input_se_dict1)
            output2 = model(adv_input_se_dict2)
            output3 = model(adv_input_sd_dict)
            output4 = model(adv_input_dict)

            output5 = model(adv_input_m1_dict)
            output6 = model(adv_input_m3_dict)
            output7 = model(adv_input_m4_dict)

            loss0 = criterion(output0, labels)
            loss1 = criterion(output1, labels)
            loss2 = criterion(output2, labels)
            loss3 = criterion(output3, labels)

            loss4 = criterion(output4, labels)

            loss5 = criterion(output5, labels)
            loss6 = criterion(output6, labels)
            loss7 = criterion(output7, labels)

            # loss = 0.25 * (loss0 + loss1 + loss2 + loss3)
            clean_loss = 0.25 * (loss0 + loss1 + loss2) * 2 * (1 - opts.adv_loss_weight_sd) + 0.25 * loss3 * 2 * opts.adv_loss_weight_sd
            muti_loss = 0.3333 * (loss5 + loss6 + loss7)
            advt_loss = loss4
            loss = 0.3333 * (clean_loss + muti_loss + advt_loss)

            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            if vis is not None:
                vis.vis_scalar('Loss', cur_itrs, np_loss)

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' + 
                      "Epoch:[{}], Itrs:[{}/{}], Loss:[{:.4f}], Time:[{:.4f} min], Best IOU:[{:.4f}]"
                      .format(cur_epochs, cur_itrs, int(opts.total_itrs), interval_loss, total_time / 60, best_score))
                writer.add_scalar('Loss/train', interval_loss, cur_itrs)
                interval_loss = 0.0
                total_time = 0.0

            if (cur_itrs) % opts.val_interval == 0 and cur_itrs >= opts.total_itrs / 2:
                save_ckpt('checkpoints/' + opts.exp + '/latest_%s_%s_os%d.pth' %
                          (opts.model, opts.dataset, opts.output_stride))
                print("validation...")
                model.eval()
                val_score, ret_samples = args.validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))

                writer.add_scalar('mIOU/test', val_score['Mean IoU'], cur_itrs)

                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt('checkpoints/' + opts.exp +'/best_%s_%s_os%d.pth' %
                              (opts.model, opts.dataset, opts.output_stride))

                if vis is not None:  # visualize validation score and samples
                    vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                    vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                    vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                    for k, (img, target, lbl) in enumerate(ret_samples):
                        img = (denorm(img) * 255).astype(np.uint8)
                        target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                        concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                        vis.vis_image('Sample %d' % k, concat_img)
                model.train()

            scheduler.step() 
            t1 = time.time() 
            total_time += t1 - t0

            if cur_itrs >=  opts.total_itrs:
                print("syd: --------------------[SD]--------------------")
                print("syd: Model dir:[{}]".format(opts.exp))
                print("syd: Setting: Layer:[{}] Gamma:[{}] Best IOU:[{}]"
                    .format(opts.pertub_idx_sd, opts.gamma_sd, best_score))
                print("syd: --------------------[SD]--------------------")
                writer.close()
                return

        
if __name__ == '__main__':
    main()
