from typing import Tuple, List, Optional, Union

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from bbox import BBox
from extension.functional import beta_smooth_l1_loss
from support.layer.nms import nms
import pdb

class RegionProposalNetwork(nn.Module):

    def __init__(self, num_features_out: int, anchor_ratios: List[Tuple[int, int]], anchor_sizes: List[int],
                 pre_nms_top_n: int, post_nms_top_n: int, anchor_smooth_l1_loss_beta: float):
        super().__init__()

        self._features = nn.Sequential(
            nn.Conv2d(in_channels=num_features_out, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self._anchor_ratios = anchor_ratios # [(1, 2), (1, 1), (2, 1)]
        self._anchor_sizes = anchor_sizes   # [128, 256, 512]

        num_anchor_ratios = len(self._anchor_ratios)  # 3
        num_anchor_sizes = len(self._anchor_sizes)    # 3
        num_anchors = num_anchor_ratios * num_anchor_sizes # 3 * 3 = 9

        self._pre_nms_top_n = pre_nms_top_n   # 12000
        self._post_nms_top_n = post_nms_top_n # 2000
        self._anchor_smooth_l1_loss_beta = anchor_smooth_l1_loss_beta

        self._anchor_objectness = nn.Conv2d(in_channels=512, out_channels=num_anchors * 2, kernel_size=1)
        self._anchor_transformer = nn.Conv2d(in_channels=512, out_channels=num_anchors * 4, kernel_size=1)

    def forward(self, features,
                anchor_bboxes: Optional[Tensor] = None, gt_bboxes_batch: Optional[Tensor] = None,
                image_width: Optional[int]=None, image_height: Optional[int]=None, return_type='clean') -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]:
        '''
            features,       : from CNN backbone,    torch.Size([1, 1024, 38, 57])
            anchor_bboxes,  : all anchors,          torch.Size([1, 19494, 4])
            gt_bboxes_batch,: (x,y,w,h)             torch.Size([1, N, 4])
            image_width,    : 901
            image_height    : 600
        '''
        
        if return_type == 'clean':

            batch_size = features.shape[0]
            features = self._features(features)                      # torch.Size([1, 512, 38, 57])
            anchor_objectnesses = self._anchor_objectness(features)  # torch.Size([1, 18, 38, 57])
            anchor_transformers = self._anchor_transformer(features) # torch.Size([1, 36, 38, 57])

            anchor_objectnesses = anchor_objectnesses.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2) # torch.Size([1, 19494, 2])
            anchor_transformers = anchor_transformers.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4) # torch.Size([1, 19494, 4])

            if not self.training:
                return anchor_objectnesses, anchor_transformers
            else:
                # remove cross-boundary
                # NOTE: The length of `inside_indices` is guaranteed to be a multiple of `anchor_bboxes.shape[0]` as each batch in `anchor_bboxes` is the same
                inside_indices = BBox.inside(anchor_bboxes, left=0, top=0, right=image_width, bottom=image_height).nonzero().unbind(dim=1) # (batch index, satisified index)
                inside_anchor_bboxes = anchor_bboxes[inside_indices].view(batch_size, -1, anchor_bboxes.shape[2]) # 20000 -> 6000
                inside_anchor_objectnesses = anchor_objectnesses[inside_indices].view(batch_size, -1, anchor_objectnesses.shape[2]) # torch.Size([1, 7512, 2])
                inside_anchor_transformers = anchor_transformers[inside_indices].view(batch_size, -1, anchor_transformers.shape[2]) # torch.Size([1, 7512, 4])

                # find labels for each `anchor_bboxes`
                labels = torch.full((batch_size, inside_anchor_bboxes.shape[1]), -1, dtype=torch.long, device=inside_anchor_bboxes.device) # torch.Size([1, 7512]) all is -1
                ious = BBox.iou(inside_anchor_bboxes, gt_bboxes_batch) # torch.Size([2, 6252, 12]) iou for each anchor_bboxes, gt_bboxes_batch: torch.Size([2, 12, 4])
                
                # max element in each raw
                '''
                anchor_assignments: max iou gt bbox id for each anchor
                '''
                anchor_max_ious, anchor_assignments = ious.max(dim=2) # max gt iou for each row (each anchor), then we can assign anchor for this gt
                gt_max_ious, gt_assignments = ious.max(dim=1) # max element in each colomn
                anchor_additions = ((ious > 0) & (ious == gt_max_ious.unsqueeze(dim=1))).nonzero()[:, :2].unbind(dim=1)
                labels[anchor_max_ious < 0.3] = 0
                labels[anchor_additions] = 1
                labels[anchor_max_ious >= 0.7] = 1
                
                # select 256 x `batch_size` samples
                fg_indices = (labels == 1).nonzero() # torch.Size([6, 2])
                bg_indices = (labels == 0).nonzero() # torch.Size([7407, 2])
                fg_indices = fg_indices[torch.randperm(len(fg_indices))[:min(len(fg_indices), 128 * batch_size)]] # torch.Size([6, 2])
                bg_indices = bg_indices[torch.randperm(len(bg_indices))[:256 * batch_size - len(fg_indices)]]     # torch.Size([250, 2])
                selected_indices = torch.cat([fg_indices, bg_indices], dim=0)
                selected_indices = selected_indices[torch.randperm(len(selected_indices))].unbind(dim=1)  # shuffle selected index 256

                inside_anchor_bboxes = inside_anchor_bboxes[selected_indices]  # select pos and neg anchor
                gt_bboxes = gt_bboxes_batch[selected_indices[0], anchor_assignments[selected_indices]] # select indices corresponding gt bboxes
                gt_anchor_objectnesses = labels[selected_indices] # labels for all choose pos and neg anchor
                gt_anchor_transformers = BBox.calc_transformer(inside_anchor_bboxes, gt_bboxes) # compute cha ju
                batch_indices = selected_indices[0]

                anchor_objectness_losses, anchor_transformer_losses = self.loss(inside_anchor_objectnesses[selected_indices], # pred cls [256, 2]
                                                                                inside_anchor_transformers[selected_indices], # pred reg [256, 4]
                                                                                gt_anchor_objectnesses, # gt cls [256] :  classification labels
                                                                                gt_anchor_transformers, # gt reg [256,4]: regression  labels
                                                                                batch_size, batch_indices)

                return anchor_objectnesses, anchor_transformers, anchor_objectness_losses, anchor_transformer_losses

        elif return_type == 'head':

            return_dict = {}
            batch_size = features.shape[0]
            features = self._features(features)                      # torch.Size([1, 512, 38, 57])
            return_dict['batch_size'] = torch.tensor([[batch_size]]).cuda()
            return_dict['rpn_feature'] = features
            return return_dict

        elif return_type == 'tail':
            
            batch_size = features['batch_size'][0].item()
            rpn_feature = features['rpn_feature']

            anchor_objectnesses = self._anchor_objectness(rpn_feature)  # torch.Size([1, 18, 38, 57])
            anchor_transformers = self._anchor_transformer(rpn_feature) # torch.Size([1, 36, 38, 57])
            anchor_objectnesses = anchor_objectnesses.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2) # torch.Size([1, 19494, 2])
            anchor_transformers = anchor_transformers.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4) # torch.Size([1, 19494, 4])

            if not self.training:
                return anchor_objectnesses, anchor_transformers
            else:
                # remove cross-boundary
                # NOTE: The length of `inside_indices` is guaranteed to be a multiple of `anchor_bboxes.shape[0]` as each batch in `anchor_bboxes` is the same
                inside_indices = BBox.inside(anchor_bboxes, left=0, top=0, right=image_width, bottom=image_height).nonzero().unbind(dim=1) # (batch index, satisified index)
                inside_anchor_bboxes = anchor_bboxes[inside_indices].view(batch_size, -1, anchor_bboxes.shape[2]) # 20000 -> 6000
                inside_anchor_objectnesses = anchor_objectnesses[inside_indices].view(batch_size, -1, anchor_objectnesses.shape[2]) # torch.Size([1, 7512, 2])
                inside_anchor_transformers = anchor_transformers[inside_indices].view(batch_size, -1, anchor_transformers.shape[2]) # torch.Size([1, 7512, 4])

                # find labels for each `anchor_bboxes`
                labels = torch.full((batch_size, inside_anchor_bboxes.shape[1]), -1, dtype=torch.long, device=inside_anchor_bboxes.device) # torch.Size([1, 7512]) all is -1
                ious = BBox.iou(inside_anchor_bboxes, gt_bboxes_batch) # torch.Size([2, 6252, 12]) iou for each anchor_bboxes, gt_bboxes_batch: torch.Size([2, 12, 4])
                
                # max element in each raw
                '''
                anchor_assignments: max iou gt bbox id for each anchor
                '''
                anchor_max_ious, anchor_assignments = ious.max(dim=2) # max gt iou for each row (each anchor), then we can assign anchor for this gt
                gt_max_ious, gt_assignments = ious.max(dim=1) # max element in each colomn
                anchor_additions = ((ious > 0) & (ious == gt_max_ious.unsqueeze(dim=1))).nonzero()[:, :2].unbind(dim=1)
                labels[anchor_max_ious < 0.3] = 0
                labels[anchor_additions] = 1
                labels[anchor_max_ious >= 0.7] = 1
                
                # select 256 x `batch_size` samples
                fg_indices = (labels == 1).nonzero() # torch.Size([6, 2])
                bg_indices = (labels == 0).nonzero() # torch.Size([7407, 2])
                fg_indices = fg_indices[torch.randperm(len(fg_indices))[:min(len(fg_indices), 128 * batch_size)]] # torch.Size([6, 2])
                bg_indices = bg_indices[torch.randperm(len(bg_indices))[:256 * batch_size - len(fg_indices)]]     # torch.Size([250, 2])
                selected_indices = torch.cat([fg_indices, bg_indices], dim=0)
                selected_indices = selected_indices[torch.randperm(len(selected_indices))].unbind(dim=1)  # shuffle selected index 256

                inside_anchor_bboxes = inside_anchor_bboxes[selected_indices]  # select pos and neg anchor
                gt_bboxes = gt_bboxes_batch[selected_indices[0], anchor_assignments[selected_indices]] # select indices corresponding gt bboxes
                gt_anchor_objectnesses = labels[selected_indices] # labels for all choose pos and neg anchor
                gt_anchor_transformers = BBox.calc_transformer(inside_anchor_bboxes, gt_bboxes) # compute cha ju
                batch_indices = selected_indices[0]

                anchor_objectness_losses, anchor_transformer_losses = self.loss(inside_anchor_objectnesses[selected_indices], # pred cls [256, 2]
                                                                                inside_anchor_transformers[selected_indices], # pred reg [256, 4]
                                                                                gt_anchor_objectnesses, # gt cls [256] :  classification labels
                                                                                gt_anchor_transformers, # gt reg [256,4]: regression  labels
                                                                                batch_size, batch_indices)

                return anchor_objectnesses, anchor_transformers, anchor_objectness_losses, anchor_transformer_losses

        else: assert False


    def loss(self, anchor_objectnesses: Tensor, anchor_transformers: Tensor,
             gt_anchor_objectnesses: Tensor, gt_anchor_transformers: Tensor,
             batch_size: int, batch_indices: Tensor) -> Tuple[Tensor, Tensor]:
        
        cross_entropies = torch.empty(batch_size, dtype=torch.float, device=anchor_objectnesses.device)
        smooth_l1_losses = torch.empty(batch_size, dtype=torch.float, device=anchor_transformers.device)

        for batch_index in range(batch_size):
            selected_indices = (batch_indices == batch_index).nonzero().view(-1)

            cross_entropy = F.cross_entropy(input=anchor_objectnesses[selected_indices],
                                            target=gt_anchor_objectnesses[selected_indices])

            fg_indices = gt_anchor_objectnesses[selected_indices].nonzero().view(-1)
            smooth_l1_loss = beta_smooth_l1_loss(input=anchor_transformers[selected_indices][fg_indices],
                                                 target=gt_anchor_transformers[selected_indices][fg_indices],
                                                 beta=self._anchor_smooth_l1_loss_beta)

            cross_entropies[batch_index] = cross_entropy
            smooth_l1_losses[batch_index] = smooth_l1_loss

        return cross_entropies, smooth_l1_losses

    def generate_anchors(self, image_width: int, image_height: int, num_x_anchors: int, num_y_anchors: int) -> Tensor:
        '''
            image_width:    original image width   600
            image_height:   original image height  800
            num_x_anchors:  anchors in x           38
            num_y_anchors:  anchors in y           50
        '''
        
        center_ys = np.linspace(start=0, stop=image_height, num=num_y_anchors + 2)[1:-1]  # 50 points
        center_xs = np.linspace(start=0, stop=image_width, num=num_x_anchors + 2)[1:-1]   # 38 points
        ratios = np.array(self._anchor_ratios) # [(1, 2), (1, 1), (2, 1)]
        ratios = ratios[:, 0] / ratios[:, 1]   # array([0.5, 1. , 2. ])
        sizes = np.array(self._anchor_sizes)   # array([128, 256, 512])

        # NOTE: it's important to let `center_ys` be the major index (i.e., move horizontally and then vertically) for consistency with 2D convolution
        # giving the string 'ij' returns a meshgrid with matrix indexing, i.e., with shape (#center_ys, #center_xs, #ratios)
        center_ys, center_xs, ratios, sizes = np.meshgrid(center_ys, center_xs, ratios, sizes, indexing='ij')

        center_ys = center_ys.reshape(-1)
        center_xs = center_xs.reshape(-1)
        ratios = ratios.reshape(-1)
        sizes = sizes.reshape(-1)

        widths = sizes * np.sqrt(1 / ratios)
        heights = sizes * np.sqrt(ratios)

        center_based_anchor_bboxes = np.stack((center_xs, center_ys, widths, heights), axis=1)
        center_based_anchor_bboxes = torch.from_numpy(center_based_anchor_bboxes).float()
        anchor_bboxes = BBox.from_center_base(center_based_anchor_bboxes) # move to left top
        
        return anchor_bboxes

    def generate_proposals(self, anchor_bboxes: Tensor, objectnesses: Tensor, transformers: Tensor, image_width: int, image_height: int) -> Tensor:
        
        '''
        This process like inference
        input:
        anchor_bboxes,       : all original anchor_bbox              torch.Size([batch_size, 19494, 4])
        objectnesses,        : all anchor, classification output     torch.Size([batch_size, 19494, 2]), output is bbox or not
        transformers,        : all anchor, regression network output torch.Size([batch_size, 19494, 4]), output some regression small change
        image_width,         : 901
        image_height         : 600
        '''
        
        batch_size = anchor_bboxes.shape[0]

        proposal_bboxes = BBox.apply_transformer(anchor_bboxes, transformers)   # recover all the anchor from net work output
        proposal_bboxes = BBox.clip(proposal_bboxes, left=0, top=0, right=image_width, bottom=image_height) # clip proposal out bounary
        proposal_probs = F.softmax(objectnesses[:, :, 1], dim=-1)   # softmax to choose the top best anchor, dim=1 means final dimision

        _, sorted_indices = torch.sort(proposal_probs, dim=-1, descending=True)
        nms_proposal_bboxes_batch = []

        for batch_index in range(batch_size):
            sorted_bboxes = proposal_bboxes[batch_index][sorted_indices[batch_index]][:self._pre_nms_top_n] # choose top 12000 bboxes
            sorted_probs = proposal_probs[batch_index][sorted_indices[batch_index]][:self._pre_nms_top_n]   # choose top 12000 probs
            threshold = 0.7
            kept_indices = nms(sorted_bboxes, sorted_probs, threshold)      # nms get rid of to 2247
            nms_bboxes = sorted_bboxes[kept_indices][:self._post_nms_top_n] # top 2000
            nms_proposal_bboxes_batch.append(nms_bboxes)

        max_nms_proposal_bboxes_length = max([len(it) for it in nms_proposal_bboxes_batch])
        padded_proposal_bboxes = []
        
        for nms_proposal_bboxes in nms_proposal_bboxes_batch:
            padded_proposal_bboxes.append(
                torch.cat([
                    nms_proposal_bboxes,
                    torch.zeros(max_nms_proposal_bboxes_length - len(nms_proposal_bboxes), 4).to(nms_proposal_bboxes)
                ])
            )

        padded_proposal_bboxes = torch.stack(padded_proposal_bboxes, dim=0)
        return padded_proposal_bboxes
