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

def PGD(x, loss_fn, y=None, model=None, steps=3, gamma=None, start_idx=1, layer_number=16, eps=(2/255), randinit=False, clip=False):
    
    # Compute loss
    x_adv = x.clone()
    if randinit:
        # adv noise (-eps, eps)
        x_adv += (2.0 * torch.rand(x_adv.shape).cuda() - 1.0) * eps
    x_adv = Variable(x_adv.cuda(), requires_grad=True)
    x = x.cuda()

    for t in range(steps):

        out = model(x_adv, end_point=layer_number, start_point=start_idx)
        loss_adv0 = loss_fn(out, y)
        grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
        x_adv.data.add_(gamma * torch.sign(grad0.data))

        if clip:
            linfball_proj(x, eps, x_adv, in_place=True)

    return x_adv
