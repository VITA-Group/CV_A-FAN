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
parser.add_argument('--seed', default=None, type=int, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--resume', action="store_true", help="resume from checkpoint")
parser.add_argument('--save_dir', help='The directory used to save the trained models', default='res56s_cifar10_baseline', type=str)

################################optimizer setting #############################################
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--decreasing_lr', default='50,150', help='decreasing strategy')

best_prec1 = 0
layer_number = 0

def main():
    global args, best_prec1, layer_number
    args = parser.parse_args()
    print(args)
    torch.cuda.set_device(int(args.gpu))

    if args.seed:
        setup_seed(args.seed)


    model = resnet56()
    layer_number = 34
    model.cuda()
    train_loader, val_loader, test_loader = cifar10_dataloaders(train_batch_size= args.batch_size, test_batch_size=args.batch_size, data_dir =args.data)

    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)


    start_epoch = 0
    if args.resume:
        print('resume from checkpoint')
        checkpoint = torch.load(os.path.join(args.save_dir, 'checkpoint.pt'), map_location = torch.device('cuda:'+str(args.gpu)))
        best_prec1 = checkpoint['best_prec1']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    all_result = {}
    train_acc=[]
    ta=[]
    test_ta=[]

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):

        print(optimizer.state_dict()['param_groups'][0]['lr'])
        acc,_ = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        tacc,_ = validate(val_loader, model, criterion)

        # evaluate on test set
        test_tacc,_ = validate(test_loader, model, criterion)

        scheduler.step()

        train_acc.append(acc)
        ta.append(tacc)
        test_ta.append(test_tacc)

        # remember best prec@1 and save checkpoint
        is_best = tacc  > best_prec1
        best_prec1 = max(tacc, best_prec1)

        if is_best:

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, is_best, filename=os.path.join(args.save_dir, 'best_model.pt'))
    
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.pt'))
    
        plt.plot(train_acc, label='train_acc')
        plt.plot(ta, label='TA')
        plt.plot(test_ta, label='test_TA')
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, 'net_train.png'))
        plt.close()
    
        all_result['train'] = train_acc
        all_result['test_ta'] = test_ta
        all_result['ta'] = ta

        pickle.dump(all_result, open(os.path.join(args.save_dir, 'result.pkl'),'wb'))


def train(train_loader, model, criterion, optimizer, epoch):
    
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    wp_steps = len(train_loader)

    for i, (input, target) in enumerate(train_loader):

        if epoch ==0:
            warmup_lr(i, optimizer, warm_up_steps = wp_steps, max_lr=args.lr)

        input = input.cuda()
        target = target.cuda()

        output_clean = model(input, end_point=layer_number, start_point=0)
        loss = criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_clean.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                    epoch, i, len(train_loader), loss=losses, top1=top1))

    print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg, losses.avg
    
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

def save_checkpoint(state, is_best, filename='weight.pt'):
    """
    Save the training model
    """
    torch.save(state, filename)

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

def warmup_lr(step, optimizer, warm_up_steps=200, max_lr=0.1):

    lr = step*max_lr/(warm_up_steps-1)
    lr = min(lr,max_lr)
    for p in optimizer.param_groups:
        p['lr']=lr

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

def setup_seed(seed): 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 


if __name__ == '__main__':
    main()


