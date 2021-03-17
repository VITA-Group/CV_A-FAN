import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pdb
import numpy as np
import matplotlib.pyplot as plt

def tensor_clamp(t, min, max, in_place=True):
    if not in_place:
        res = t.clone()
    else:
        res = t
    idx = res.data < min
    res.data[idx] = min[idx]
    idx = res.data > max
    res.data[idx] = max[idx]

    return res

def compute_loss(loss1, loss2, loss3, loss4):
    loss1 = loss1.mean()
    loss2 = loss2.mean()
    loss3 = loss3.mean()
    loss4 = loss4.mean()
    loss = loss1 + loss2 + loss3 + loss4
    return loss


def l2ball_proj(center, radius, t, in_place=True):
    if not in_place:
        res = t.clone()
    else:
        res = t

    direction = t - center
    dist = direction.view(direction.size(0), -1).norm(p=2, dim=1, keepdim=True)
    direction.view(direction.size(0), -1).div_(dist)
    dist[dist > radius] = radius
    direction.view(direction.size(0), -1).mul_(dist)
    res.data.copy_(center + direction)
    return res

def linfball_proj(center, radius, t, in_place=True):
    return tensor_clamp(t, min=center - radius, max=center + radius, in_place=in_place)


def PGD(x, image_batch, y=None, model=None, steps=3, eps=None, gamma=None, idx=1, randinit=False, clip=False):
    
    # Compute loss
    x_adv = x.clone()
    if randinit:
        x_adv += (2.0 * torch.rand(x_adv.shape).cuda() - 1.0) * eps
    x_adv = Variable(x_adv, requires_grad=True)
    # x = x.cuda()
    for t in range(steps):

        inputs = {'x': image_batch, 'adv': x_adv, 'out_idx':idx, 'flag':'tail'}
        anchor_objectness_losses, anchor_transformer_losses, proposal_class_losses, proposal_transformer_losses = \
             model.train().forward(inputs, y['bb'], y['lb'])
            
        anchor_objectness_loss = anchor_objectness_losses.mean()
        anchor_transformer_loss = anchor_transformer_losses.mean()
        proposal_class_loss = proposal_class_losses.mean()
        proposal_transformer_loss = proposal_transformer_losses.mean()
        loss = anchor_objectness_loss + anchor_transformer_loss + proposal_class_loss + proposal_transformer_loss
        
        grad0 = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
        x_adv.data.add_(gamma * torch.sign(grad0.data))
        
        if clip:
            linfball_proj(x, eps, x_adv, in_place=True)

    return x_adv


def rpn_roi_PGD(layer='roi' , rpn_roi_output_dict=None,
            y=None, model=None, steps=1, eps=None, gamma=None, randinit=False, clip=False, only_roi_loss=True):
    
    if layer == 'roi':

        roi_output_dict = rpn_roi_output_dict
        roi_feature = roi_output_dict['roi_output_dict']['roi_feature_map'].detach()
        x_adv = roi_feature.clone()

        if randinit:
            x_adv += (2.0 * torch.rand(x_adv.shape).cuda() - 1.0) * eps

        x_adv = Variable(x_adv, requires_grad=True)
        roi_output_dict['roi_output_dict']['roi_feature_map'] = x_adv
        
        for t in range(steps):
            
            inputs = {'adv': roi_output_dict, 'out_idx': 'roi_tail', 'flag':'clean'}
            anchor_objectness_losses, anchor_transformer_losses, proposal_class_losses, proposal_transformer_losses = \
                model.train().forward(inputs, y['bb'], y['lb'])
                
            anchor_objectness_loss = anchor_objectness_losses.mean()
            anchor_transformer_loss = anchor_transformer_losses.mean()
            proposal_class_loss = proposal_class_losses.mean()
            proposal_transformer_loss = proposal_transformer_losses.mean()
            # loss = anchor_objectness_loss + anchor_transformer_loss + proposal_class_loss + proposal_transformer_loss
            if only_roi_loss:
                loss = proposal_class_loss + proposal_transformer_loss
            else:
                loss = anchor_objectness_loss + anchor_transformer_loss + proposal_class_loss + proposal_transformer_loss
            grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
            x_adv.data.add_(gamma * torch.sign(grad.data))

            if clip:
                linfball_proj(rpn_feature1, eps, x_adv, in_place=True)

        roi_output_dict['roi_output_dict']['roi_feature_map'] = x_adv
        return roi_output_dict

    elif layer == 'rpn':

        rpn_output_dict = rpn_roi_output_dict
        
        rpn_feature = rpn_output_dict['rpn_feature_map_dict']['rpn_feature'].detach()
        x_adv = rpn_feature.clone()
        if randinit:
            x_adv += (2.0 * torch.rand(x_adv.shape).cuda() - 1.0) * eps
        x_adv = Variable(x_adv, requires_grad=True)
        rpn_output_dict['rpn_feature_map_dict']['rpn_feature'] = x_adv

        for t in range(steps):
            
            inputs = {'adv': rpn_output_dict, 'out_idx': 'rpn_tail', 'flag':'clean'}
            anchor_objectness_losses, anchor_transformer_losses, proposal_class_losses, proposal_transformer_losses = \
                model.train().forward(inputs, y['bb'], y['lb'])
                
            # anchor_objectness_loss = anchor_objectness_losses.mean()
            # anchor_transformer_loss = anchor_transformer_losses.mean()
            # proposal_class_loss = proposal_class_losses.mean()
            # proposal_transformer_loss = proposal_transformer_losses.mean()
            # loss = anchor_objectness_loss + anchor_transformer_loss + proposal_class_loss + proposal_transformer_loss
            # grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
            # x_adv.data.add_(gamma * torch.sign(grad.data))
            # x_adv = x_adv.detach()
        #     if clip:
        #         linfball_proj(rpn_feature, eps, x_adv, in_place=True)
        # num = x_adv1.shape[0] * x_adv1.shape[1] * x_adv1.shape[2] * x_adv1.shape[3]
        # no_eqal = num - (x_adv1==rpn_feature1).sum().item()
        # print("no eqal:[{}/{}]".format(no_eqal, num))
        rpn_output_dict['rpn_feature_map_dict']['rpn_feature'] = x_adv
        return rpn_output_dict

    else:
        assert False


def adv_input(x=None, y=None, model=None, steps=3, eps=None, gamma=None, randinit=False, clip=False):
    
    x_adv = x.clone()
    if randinit:
        x_adv += (2.0 * torch.rand(x_adv.shape).cuda() - 1.0) * eps
    x_adv = Variable(x_adv, requires_grad=True)
    
    for t in range(steps):

        inputs = {'x': x_adv, 'adv': None, 'out_idx': -1, 'flag':'clean'}
        anchor_objectness_losses, anchor_transformer_losses, proposal_class_losses, proposal_transformer_losses = \
             model.train().forward(inputs, y['bb'], y['lb'])
        anchor_objectness_loss = anchor_objectness_losses.mean()
        anchor_transformer_loss = anchor_transformer_losses.mean()
        proposal_class_loss = proposal_class_losses.mean()
        proposal_transformer_loss = proposal_transformer_losses.mean()
        loss = anchor_objectness_loss + anchor_transformer_loss + proposal_class_loss + proposal_transformer_loss
        
        grad0 = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
        x_adv.data.add_(gamma * torch.sign(grad0.data))
        
        if clip:
            linfball_proj(x, eps, x_adv, in_place=True)

    x_adv = torch.clamp(x_adv, 0, 1.0)
    return x_adv


# def untarget_PGD(x, y=None, model=None, steps=3, eps=None, gamma=None, randinit=False, clip=False):
#     # Compute loss
#     x_adv = x.clone()
#     if randinit:
#         x_adv += (2.0 * torch.rand(x_adv.shape).cuda() - 1.0) * eps
#     x_adv = Variable(x_adv, requires_grad=True)
#     # x = x.cuda()
#     for t in range(steps):
        
#         inputs = {'x': x_adv, 'adv': None, 'out_idx':0, 'flag':'clean'}
#         anchor_objectness_losses, _, proposal_class_losses, __ = \
#              model.train().forward(inputs, y['bb'], y['lb'])

#         anchor_objectness_loss = anchor_objectness_losses.mean()
#         proposal_class_loss = proposal_class_losses.mean()
#         cls_loss = anchor_objectness_loss + proposal_class_loss
#         grad0 = torch.autograd.grad(cls_loss, x_adv, only_inputs=True)[0]
        
#         x_adv.data.add_(gamma * grad0.data / (torch.norm(grad0, p=float('inf'))))
        
#         if clip:
#             linfball_proj(x, eps, x_adv, in_place=True)

#     return x_adv


def eval_PGD(x, y=None, model=None, steps=3, eps=None, gamma=None, idx=1, randinit=False, clip=False):
    
    # Compute loss
    x_adv = x.clone()
    if randinit:
        x_adv += (2.0 * torch.rand(x_adv.shape).cuda() - 1.0) * eps
    x_adv = Variable(x_adv, requires_grad=True)
    # x = x.cuda()
    for t in range(steps):
        
        inputs = {'x': x_adv, 'adv': None, 'out_idx':0, 'flag':'clean'}
        anchor_objectness_losses, anchor_transformer_losses, proposal_class_losses, proposal_transformer_losses = \
             model.train().forward(inputs, y['bb'], y['lb'])
        
        anchor_objectness_loss = anchor_objectness_losses.mean()
        anchor_transformer_loss = anchor_transformer_losses.mean()
        proposal_class_loss = proposal_class_losses.mean()
        proposal_transformer_loss = proposal_transformer_losses.mean()
        loss = anchor_objectness_loss + proposal_class_loss + anchor_transformer_loss + proposal_transformer_loss
        # loss = anchor_objectness_loss + proposal_class_loss
        grad0 = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
        x_adv.data.add_(gamma * torch.sign(grad0.data))
        
        if clip:
            linfball_proj(x, eps, x_adv, in_place=True)

    return x_adv


def get_sample_points(pointx, pointy, number):
    
    percent = 1.0 / (number - 1)
    per_list = [i * percent for i in range(1, number - 1)]
    final_list = [pointx]
    for i in per_list:
        point_new = torch.lerp(pointx, pointy, i)
        final_list.append(point_new)
    final_list.append(pointy)
    return final_list

    
def print_args(args, str_num=80):
    for arg, val in args.__dict__.items():
        print(arg + '.' * (str_num - len(arg) - len(str(val))) + str(val))
    print()


def mix_feature(clean_feature, adv_feature):

    eps=1e-5
    mean_cl = clean_feature.mean(dim=1, keepdim=True)
    std_cl = (clean_feature.var(dim=1, keepdim=True) + eps).sqrt()
    mean_adv = adv_feature.mean(dim=1, keepdim=True)
    std_adv = (adv_feature.var(dim=1, keepdim=True) + eps).sqrt()
    
    clean_feature = (clean_feature - mean_cl) / std_cl
    mix_feature = (clean_feature * (std_adv)) + (mean_adv)

    return mix_feature


def imsave(clean_img, adv_img,  save_name):
    
    clean = clean_img.cpu().detach().numpy()
    adv = adv_img.cpu().detach().numpy()
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1), plt.title('clean')
    plt.imshow(np.transpose(clean, (1, 2, 0)))
    plt.subplot(1,2,2), plt.title('adv')
    plt.imshow(np.transpose(adv, (1, 2, 0)))
    plt.savefig(save_name)

def feature_map_save(img, save_name):
    
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    plt.savefig(save_name, dpi=1000)

def img_save(img, save_name):

    img = img.cpu().detach().numpy()
    plt.xticks([])
    plt.yticks([])
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.savefig(save_name, dpi=1000)



def perturb_weight(model_dict):
    
    perturb_dict={}
    for key in model_dict.keys():
        if 'normal' in key:
            continue
        # if 'bn' in key:
        #     continue
        size = model_dict[key].size()
        perturb_dict[key] = torch.rand(size).cuda()
        # print(key)
    
    #normalize
    all_weight = []
    for key in perturb_dict.keys():
        all_weight.append(perturb_dict[key].view(-1))
        
    all_weight = torch.cat(all_weight, dim=0)
    # print(all_weight.shape)
    norm = torch.norm(all_weight, p=2)

    for key in perturb_dict.keys():
        perturb_dict[key] = perturb_dict[key]/norm
    return perturb_dict

# def feature_map_save(ori, img1, img2, img3, save_name):
    
#     ori = ori.cpu().detach().numpy()
#     img1 = img1.cpu().detach().numpy()
#     img2 = img2.cpu().detach().numpy()
#     img3 = img3.cpu().detach().numpy()
#     plt.figure(figsize=(10,30))
#     plt.subplot(1,4,1), plt.title('original')
#     plt.imshow(np.transpose(ori, (1, 2, 0)))
#     plt.subplot(1,4,2), plt.title('layer1')
#     plt.imshow(img1)
#     plt.subplot(1,4,3), plt.title('layer2')
#     plt.imshow(img2)
#     plt.subplot(1,4,4), plt.title('layer3')
#     plt.imshow(img3)
#     plt.savefig(save_name)


# def sat_feature_map_save(ori, img1_list, img2_list, img3_list, save_name):
    
#     ori = ori.cpu().detach().numpy()
#     img1_list = [i.squeeze()[0].cpu().detach().numpy() for i in img1_list]
#     img2_list = [i.squeeze()[0].cpu().detach().numpy() for i in img2_list]
#     img3_list = [i.squeeze()[0].cpu().detach().numpy() for i in img3_list]
#     # plt.figure(figsize=(40,30))
#     plt.subplot(5, 4, 1), plt.title('original')
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(np.transpose(ori, (1, 2, 0)))
#     id_list = [2, 6, 10, 14, 18]
#     for i, j in enumerate(id_list):

#         plt.subplot(5, 4, j)
#         if i == 0: plt.title('layer1')
#         plt.xticks([])
#         plt.yticks([])
#         plt.imshow(img1_list[i])
#         plt.subplot(5, 4, j + 1)
#         if i == 0: plt.title('layer2')
#         plt.xticks([])
#         plt.yticks([])
#         plt.imshow(img2_list[i])
#         plt.subplot(5, 4, j + 2)
#         if i == 0: plt.title('layer3')
#         plt.xticks([])
#         plt.yticks([])
#         plt.imshow(img3_list[i])

#     plt.savefig(save_name, dpi=1000)

