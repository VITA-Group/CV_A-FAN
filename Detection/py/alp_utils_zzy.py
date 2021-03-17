import os
import torch
import random
import numpy as np  
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn 

def rademacher(size):
    data = torch.rand(size)
    neg_one = -1* torch.ones(size)
    one = torch.ones(size)
    rand_rademacher = torch.where(data>0.5, neg_one, one)

    return rand_rademacher

def gradient_generate(input, y, model):

    input.requires_grad = True
    inputs = {'x': input, 'adv': None, 'out_idx': -1, 'flag':'clean'}
    anchor_objectness_losses, anchor_transformer_losses, proposal_class_losses, proposal_transformer_losses = \
        model.train().forward(inputs, y['bb'], y['lb'])
    anchor_objectness_loss = anchor_objectness_losses.mean()
    anchor_transformer_loss = anchor_transformer_losses.mean()
    proposal_class_loss = proposal_class_losses.mean()
    proposal_transformer_loss = proposal_transformer_losses.mean()
    loss = anchor_objectness_loss + anchor_transformer_loss + proposal_class_loss + proposal_transformer_loss
    
    grad0 = torch.autograd.grad(loss, input, only_inputs=True)[0]
    r1 = torch.sign(grad0.data)

    return r1 
