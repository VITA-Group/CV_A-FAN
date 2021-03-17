import argparse
import os
import pdb
import pickle
import random
import numpy as np  
import PIL.Image as Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler

from resnet_s import resnet56
from dataset import cifar10_dataloaders

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')


################################# base setting ###############################################
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--pretrained', help='pretrained_model', default='res56s_cifar10_baseline', type=str)
parser.add_argument('--batch_size', type=int, default=128, help='batch size')


layer_number = 0

def main():
    global args, layer_number
    args = parser.parse_args()
    print(args)
    torch.cuda.set_device(int(args.gpu))


    model = resnet56()
    layer_number = 34
    model.cuda()
    _, _, test_loader = cifar10_dataloaders(train_batch_size= args.batch_size, test_batch_size=args.batch_size, data_dir =args.data)
    criterion = nn.CrossEntropyLoss()

    checkpoint = torch.load(args.pretrained, map_location = torch.device('cuda:'+str(args.gpu)))
    model.load_state_dict(checkpoint['state_dict'])

    test_tacc,_ = validate(test_loader, model, criterion)



def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
    
        # compute output
        with torch.no_grad():
            output = model(input, end_point=layer_number, start_point=0)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), loss=losses, top1=top1))

    print('valid_accuracy {top1.avg:.3f}'
        .format(top1=top1))

    return top1.avg, losses.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()


