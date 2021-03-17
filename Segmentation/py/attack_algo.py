import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pdb
import numpy as np

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


def PGD(x, image_batch, low_level_feat, criterion, y=None, model=None, steps=3, eps=None, gamma=None, idx=1, randinit=False, clip=False):
    
    # Compute loss
    x_adv = x.clone()
    if randinit:
        x_adv += (2.0 * torch.rand(x_adv.shape).cuda() - 1.0) * eps
    x_adv = Variable(x_adv.cuda(), requires_grad=True)

    for t in range(steps):
        
        inputs = {'x': image_batch, 'adv': x_adv, 'out_idx':idx, 'flag':'tail', 'low_level_feat': low_level_feat}
        logits = model(inputs)
        loss = criterion(logits, y)
        grad0 = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
        x_adv.data.add_(gamma * torch.sign(grad0.data))

        if clip:
            linfball_proj(x, eps, x_adv, in_place=True)

    return x_adv

def decoder_PGD(input_dict, image_batch, criterion, y=None, model=None, steps=3, eps=None, gamma=None, idx=1, randinit=False, clip=False):

    decoder_feature_map = input_dict['adv']
    decoder_feature_map = decoder_feature_map.detach()
    # Compute loss
    x_adv = decoder_feature_map.clone()
    if randinit:
        x_adv += (2.0 * torch.rand(x_adv.shape).cuda() - 1.0) * eps
    x_adv = Variable(x_adv.cuda(), requires_grad=True)

    input_dict['adv'] = x_adv
    for t in range(steps):
        
        inputs = {'x': image_batch, 'adv': input_dict, 'out_idx':idx + "_tail", 'flag':'clean'}
        logits = model(inputs)
        loss = criterion(logits, y)
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
        x_adv.data.add_(gamma * torch.sign(grad.data))

        if clip:
            linfball_proj(x, eps, x_adv, in_place=True)
        
    input_dict['adv'] = x_adv
    return input_dict

def adv_input(x=None, criterion=None, y=None, model=None, steps=3, eps=None, gamma=None, randinit=False, clip=False):
    
    # Compute loss
    x_adv = x.clone()
    if randinit:
        x_adv += (2.0 * torch.rand(x_adv.shape).cuda() - 1.0) * eps
    x_adv = Variable(x_adv.cuda(), requires_grad=True)

    for t in range(steps):

        inputs = {'x': x_adv, 'adv': None, 'out_idx': 0, 'flag':'clean', 'low_level_feat': None}
        logits = model(inputs)
        loss = criterion(logits, y)
        grad0 = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
        x_adv.data.add_(gamma * torch.sign(grad0.data))
        if clip:
            linfball_proj(x, eps, x_adv, in_place=True)

    x_adv = torch.clamp(x_adv, 0, 1.0)
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
    

def mix_feature(clean_feature, adv_feature):

    eps = 1e-5
    mean_cl = clean_feature.mean(dim=1, keepdim=True)
    std_cl = (clean_feature.var(dim=1, keepdim=True) + eps).sqrt()
    mean_adv = adv_feature.mean(dim=1, keepdim=True)
    std_adv = (adv_feature.var(dim=1, keepdim=True) + eps).sqrt()
    mix_feature = (clean_feature - mean_cl) / std_cl
    mix_feature = mix_feature * std_adv + mean_adv
    return mix_feature



# def PGDL2(x, image_batch, low_level_feat, criterion, y=None, model=None, steps=3, eps=None, gamma=None, idx=1, randinit=False, clip=False):
    
#     # Compute loss
#     x_adv = x.clone()
#     if randinit:
#         x_adv += (2.0 * torch.rand(x_adv.shape).cuda() - 1.0) * eps
#     x_adv = Variable(x_adv.cuda(), requires_grad=True)
        
#     for t in range(steps):
        
#         inputs = {'x': image_batch, 'adv': x_adv, 'out_idx':idx, 'flag':'tail', 'low_level_feat': low_level_feat}
#         logits = model(inputs)
#         loss = criterion(logits, y)
#         loss = torch.mean(loss)
#         grad0 = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]

#         x_adv.data.add_(gamma * torch.sign(grad0.data))

#         if clip:
#             linfball_proj(x, eps, x_adv, in_place=True)

#     return x_adv
