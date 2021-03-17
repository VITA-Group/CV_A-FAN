from typing import Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.base import Base as DatasetBase
from model import Model
# import attack_algo
import pdb
import numpy as np 
from alp_utils_zzy import *

class Evaluator(object):
    def __init__(self, dataset: DatasetBase, path_to_data_dir: str, path_to_results_dir: str):
        super().__init__()
        self._dataset = dataset
        self._dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
        self._path_to_data_dir = path_to_data_dir
        self._path_to_results_dir = path_to_results_dir

    def evaluate(self, model: Model) -> Tuple[float, str]:
        all_image_ids, all_detection_bboxes, all_detection_classes, all_detection_probs = [], [], [], []

        with torch.no_grad():
            for _, (image_id_batch, image_batch, scale_batch, _, _) in enumerate(tqdm(self._dataloader)):
                image_batch = image_batch.cuda()
                assert image_batch.shape[0] == 1, 'do not use batch size more than 1 on evaluation'

                inputs_all = {"x": image_batch, "adv": None, "out_idx": None, "flag": 'clean'}
                detection_bboxes, detection_classes, detection_probs, detection_batch_indices = \
                    model.eval().forward(inputs_all)

                scale_batch = scale_batch[detection_batch_indices].unsqueeze(dim=-1).expand_as(detection_bboxes).to(device=detection_bboxes.device)
                detection_bboxes = detection_bboxes / scale_batch

                kept_indices = (detection_probs > 0.05).nonzero().view(-1)
                detection_bboxes = detection_bboxes[kept_indices]
                detection_classes = detection_classes[kept_indices]
                detection_probs = detection_probs[kept_indices]
                detection_batch_indices = detection_batch_indices[kept_indices]

                all_detection_bboxes.extend(detection_bboxes.tolist())
                all_detection_classes.extend(detection_classes.tolist())
                all_detection_probs.extend(detection_probs.tolist())
                all_image_ids.extend([image_id_batch[i] for i in detection_batch_indices])

        mean_ap, detail = self._dataset.evaluate(self._path_to_results_dir, all_image_ids, all_detection_bboxes, all_detection_classes, all_detection_probs)
        return mean_ap, detail

    def rob_evaluate(self, model: Model, args) -> Tuple[float, str]:
        all_image_ids, all_detection_bboxes, all_detection_classes, all_detection_probs = [], [], [], []

        for _, (image_id_batch, image_batch, scale_batch, bboxes_batch, labels_batch) in enumerate(tqdm(self._dataloader)):
            image_batch = image_batch.cuda()
            bboxes_batch = bboxes_batch.cuda()
            labels_batch = labels_batch.cuda()
            assert image_batch.shape[0] == 1, 'do not use batch size more than 1 on evaluation'
            
            adv_image_batch = attack_algo.untarget_PGD(
                x = image_batch,  
                y = {'bb': bboxes_batch, 'lb': labels_batch},
                model= model,
                steps = args.steps,
                eps = (args.eps / 255), 
                gamma = (args.gamma / 255),
                randinit = args.randinit,
                clip = args.clip)

            inputs_all = {"x": adv_image_batch, "adv": None, "out_idx": None, "flag": 'clean'}
            detection_bboxes, detection_classes, detection_probs, detection_batch_indices = \
                model.eval().forward(inputs_all)
            
            scale_batch = scale_batch[detection_batch_indices].unsqueeze(dim=-1).expand_as(detection_bboxes).to(device=detection_bboxes.device)
            detection_bboxes = detection_bboxes / scale_batch

            kept_indices = (detection_probs > 0.05).nonzero().view(-1)
            detection_bboxes = detection_bboxes[kept_indices]
            detection_classes = detection_classes[kept_indices]
            detection_probs = detection_probs[kept_indices]
            detection_batch_indices = detection_batch_indices[kept_indices]

            all_detection_bboxes.extend(detection_bboxes.tolist())
            all_detection_classes.extend(detection_classes.tolist())
            all_detection_probs.extend(detection_probs.tolist())
            all_image_ids.extend([image_id_batch[i] for i in detection_batch_indices])

        mean_ap, detail = self._dataset.evaluate(self._path_to_results_dir, all_image_ids, all_detection_bboxes, all_detection_classes, all_detection_probs)
        return mean_ap, detail

    def ori_rob_evaluate(self, model: Model, args) -> Tuple[float, str]:
        all_image_ids, all_detection_bboxes, all_detection_classes, all_detection_probs = [], [], [], []

        for _, (image_id_batch, image_batch, scale_batch, bboxes_batch, labels_batch) in enumerate(tqdm(self._dataloader)):
            image_batch = image_batch.cuda()
            bboxes_batch = bboxes_batch.cuda()
            labels_batch = labels_batch.cuda()
            assert image_batch.shape[0] == 1, 'do not use batch size more than 1 on evaluation'
            
            adv_image_batch = attack_algo.eval_PGD(
                x = image_batch,  
                y = {'bb': bboxes_batch, 'lb': labels_batch},
                model= model,
                steps = args.steps,
                eps = (args.eps / 255), 
                gamma = (args.gamma / 255),
                randinit = args.randinit,
                clip = args.clip)

            inputs_all = {"x": adv_image_batch, "adv": None, "out_idx": None, "flag": 'clean'}
            detection_bboxes, detection_classes, detection_probs, detection_batch_indices = \
                model.eval().forward(inputs_all)
            
            scale_batch = scale_batch[detection_batch_indices].unsqueeze(dim=-1).expand_as(detection_bboxes).to(device=detection_bboxes.device)
            detection_bboxes = detection_bboxes / scale_batch

            kept_indices = (detection_probs > 0.05).nonzero().view(-1)
            detection_bboxes = detection_bboxes[kept_indices]
            detection_classes = detection_classes[kept_indices]
            detection_probs = detection_probs[kept_indices]
            detection_batch_indices = detection_batch_indices[kept_indices]

            all_detection_bboxes.extend(detection_bboxes.tolist())
            all_detection_classes.extend(detection_classes.tolist())
            all_detection_probs.extend(detection_probs.tolist())
            all_image_ids.extend([image_id_batch[i] for i in detection_batch_indices])

        mean_ap, detail = self._dataset.evaluate(self._path_to_results_dir, all_image_ids, all_detection_bboxes, all_detection_classes, all_detection_probs)
        return mean_ap, detail
    
    def ALP_evaluate(self, model: Model, args) -> Tuple[float, str]:

        np.random.seed(2)
        idx_image_list = list(np.random.randint(0,4000,20))

        xyz={}

        all_image_ids, all_detection_bboxes, all_detection_classes, all_detection_probs = [], [], [], []

        for idx, (image_id_batch, image_batch, scale_batch, bboxes_batch, labels_batch) in enumerate(tqdm(self._dataloader)):
            
            if not idx in idx_image_list:
                continue

            print(idx)
            
            image_batch = image_batch.cuda()
            bboxes_batch = bboxes_batch.cuda()
            labels_batch = labels_batch.cuda()
            assert image_batch.shape[0] == 1, 'do not use batch size more than 1 on evaluation'
            
            r1 = gradient_generate(image_batch, {'bb': bboxes_batch, 'lb': labels_batch}, model)
            r2 = rademacher(image_batch.size())

            r1 = r1.cuda()
            r2 = r2.cuda()

            X = np.arange(-0.1, 0.1, 0.005)
            Y = np.arange(-0.1, 0.1, 0.005)
            X, Y = np.meshgrid(X, Y)
            Z = np.zeros((40,40))

            for x_idx in range(40):
                for y_idx in range(40):
                    x_value = X[x_idx,y_idx]
                    y_value = Y[x_idx,y_idx]

                    new_input = x_value*r1+y_value*r2 
                    scale_input = (new_input-new_input.min())/(new_input.max()-new_input.min()).cuda()
                    
                    with torch.no_grad():
                        
                        inputs = {'x': scale_input, 'adv': None, 'out_idx': -1, 'flag':'clean'}
                        anchor_objectness_losses, anchor_transformer_losses, proposal_class_losses, proposal_transformer_losses = \
                            model.train().forward(inputs, bboxes_batch, labels_batch)
                        anchor_objectness_loss = anchor_objectness_losses.mean()
                        anchor_transformer_loss = anchor_transformer_losses.mean()
                        proposal_class_loss = proposal_class_losses.mean()
                        proposal_transformer_loss = proposal_transformer_losses.mean()
                        loss = anchor_objectness_loss + anchor_transformer_loss + proposal_class_loss + proposal_transformer_loss
                        Z[x_idx, y_idx] = loss.float().cpu().item()

            xyz[idx] = Z
            torch.save(xyz, 'alp_adv.pt')
        
    def sat_layer_evaluate(self, model: Model, args) -> Tuple[float, str]:
        all_image_ids, all_detection_bboxes, all_detection_classes, all_detection_probs = [], [], [], []

        for _, (image_id_batch, image_batch, scale_batch, bboxes_batch, labels_batch) in enumerate(tqdm(self._dataloader)):
            image_batch = image_batch.cuda()
            bboxes_batch = bboxes_batch.cuda()
            labels_batch = labels_batch.cuda()
            assert image_batch.shape[0] == 1, 'do not use batch size more than 1 on evaluation'
            
            inputs_all = {"x": image_batch, "adv": None, "out_idx": args.pertub_idx, "flag": 'head'}
            feature_map = model.train().forward(inputs_all, bboxes_batch, labels_batch)
            feature_map = feature_map.detach()
            
            feature_adv = attack_algo.PGD(feature_map, inputs_all["x"], 
                y = {'bb':bboxes_batch, 'lb': labels_batch},
                model= model,
                steps = args.steps,
                eps = (args.eps / 255), 
                gamma = (args.gamma / 255),
                idx = args.pertub_idx,
                randinit = args.randinit,
                clip = args.clip)
            
            adv_list = attack_algo.get_sample_points(feature_map, feature_adv, 5)
            input_adv_feature = adv_list[args.sat_layer]

            if args.mix:
                # input_adv_feature = attack_algo.mix_feature(feature_map, input_adv_feature)
                input_adv_feature = attack_algo.mix_feature(input_adv_feature, feature_map)

            eval_dict = {'x': image_batch, 'adv': input_adv_feature, 'out_idx': args.pertub_idx, 'flag':'tail'}
            detection_bboxes, detection_classes, detection_probs, detection_batch_indices = \
                model.eval().forward(eval_dict)

            scale_batch = scale_batch[detection_batch_indices].unsqueeze(dim=-1).expand_as(detection_bboxes).to(device=detection_bboxes.device)
            detection_bboxes = detection_bboxes / scale_batch

            kept_indices = (detection_probs > 0.05).nonzero().view(-1)
            detection_bboxes = detection_bboxes[kept_indices]
            detection_classes = detection_classes[kept_indices]
            detection_probs = detection_probs[kept_indices]
            detection_batch_indices = detection_batch_indices[kept_indices]

            all_detection_bboxes.extend(detection_bboxes.tolist())
            all_detection_classes.extend(detection_classes.tolist())
            all_detection_probs.extend(detection_probs.tolist())
            all_image_ids.extend([image_id_batch[i] for i in detection_batch_indices])

        mean_ap, detail = self._dataset.evaluate(self._path_to_results_dir, all_image_ids, all_detection_bboxes, all_detection_classes, all_detection_probs)
        return mean_ap, detail
