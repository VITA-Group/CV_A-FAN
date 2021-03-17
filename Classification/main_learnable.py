
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
from copy import deepcopy

from resnet_s import resnet56
from attack_algo import PGD
from dataset import cifar10_dataloaders

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')

################################# base setting ###############################################
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
parser.add_argument('--seed', default=None, type=int, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--resume', action="store_true", help="resume from checkpoint")
parser.add_argument('--save_dir', help='The directory used to save the trained models', default='res56s_aug_learnable', type=str)

################################optimizer setting #############################################
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--decreasing_lr', default='50,150', help='decreasing strategy')


################################optimizer setting #############################################
parser.add_argument('--steps', default=3, type=int, help='PGD-steps')
parser.add_argument('--gamma', help='index of PGD gamma', default=1, type=float)
parser.add_argument('--eps', default=2, type=float)
parser.add_argument('--randinit', action="store_true", help="whether using apex")
parser.add_argument('--clip', action="store_true", help="whether using apex")

################################ETA setting #############################################
parser.add_argument('--w_lr', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--init_weight', default=(1/9), type=float, help='initial weight for ETA')
parser.add_argument('--l1_coef', default=1, type=float, help='initial learning rate')

best_prec1 = 0
perturb_idx_list = [4, 8, 11, 14, 18, 21, 24, 28, 31]
layer_number = 34
def main():
    global args, best_prec1, perturb_idx_list, layer_number
    args = parser.parse_args()
    print(args)

    os.makedirs(args.save_dir, exist_ok = True)

    torch.cuda.set_device(int(args.gpu))

    if args.seed:
        setup_seed(args.seed)
    
    model = resnet56(init_weight_eta = args.init_weight)
    model.cuda()

    train_loader, val_loader, test_loader = cifar10_dataloaders(train_batch_size=args.batch_size, test_batch_size=args.batch_size, data_dir=args.data)

    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.sequential_model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    params_list = [{'params': model.w, 'lr': args.w_lr,
                        'weight_decay': 0},]
    optimizer_w = torch.optim.SGD(params_list, args.w_lr,
                                momentum=args.momentum,
                                weight_decay=0)                            

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)


    start_epoch = 0
    if args.resume:
        print('resume from checkpoint')
        checkpoint = torch.load(os.path.join(args.save_dir, 'checkpoint.pt'), map_location = torch.device('cuda:'+str(args.gpu)))
        best_prec1 = checkpoint['best_prec1']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer_w.load_state_dict(checkpoint['optimizer_w'])
        scheduler.load_state_dict(checkpoint['scheduler'])


    all_result = {}
    train_acc=[]
    ta=[]
    test_ta=[]

    all_norm_result = {'l2':{},'linf':{}}

    for epoch in range(start_epoch, args.epochs):

        for num in range(9):
            print('weight'+str(num+1)+' = ', model.w[num].item())
        
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        print(optimizer_w.state_dict()['param_groups'][0]['lr'])
        acc,_, epoch_norml2, epoch_normlinf = train(train_loader, model, criterion, optimizer, epoch, optimizer_w)
        all_norm_result['l2'][epoch+1] = epoch_norml2
        all_norm_result['linf'][epoch+1] = epoch_normlinf

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
                'optimizer_w': optimizer_w.state_dict(),
                'scheduler': scheduler.state_dict()
            }, is_best, filename=os.path.join(args.save_dir, 'best_model.pt'))
    
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
            'optimizer_w': optimizer_w.state_dict(),
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
        pickle.dump(all_norm_result, open(os.path.join(args.save_dir, 'result_norm.pkl'),'wb'))


def train(train_loader, model, criterion, optimizer, epoch, optimizer_w):
    
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    wp_steps = len(train_loader)

    norm_save_l2 = []
    norm_save_linf = []

    for i, (input, target) in enumerate(train_loader):

        if epoch ==0:
            warmup_lr(i, optimizer, warm_up_steps = wp_steps, max_lr=args.lr)

        input = input.cuda()
        target = target.cuda()

        adv_feature_maps = []
        clean_feature_maps = []
        output_list = []
        batch_norm_l2 = []
        batch_norm_linf = []        
        batch_size = input.shape[0]

        for num in perturb_idx_list:
            fea_map = model(input, end_point=num, start_point=0).detach()
            clean_feature_maps.append(fea_map)
            fea_map_adv = PGD(fea_map, criterion, 
                y=target, 
                model=model, 
                steps=args.steps, 
                gamma=(args.gamma/255),
                start_idx=num, 
                layer_number= layer_number,
                eps=(args.eps/255), 
                randinit=args.randinit, 
                clip=args.clip)
            adv_feature_maps.append(fea_map_adv)
    
        
        for idx, num in enumerate(perturb_idx_list):

            current_norm_l2 = torch.norm((adv_feature_maps[idx]-clean_feature_maps[idx]).clone().detach().cpu().reshape(batch_size, -1), dim=1, p=2)
            current_norm_linf = torch.norm((adv_feature_maps[idx]-clean_feature_maps[idx]).clone().detach().cpu().reshape(batch_size, -1), dim=1, p=float('inf'))
            
            batch_norm_l2.append(current_norm_l2.reshape(1,-1))
            batch_norm_linf.append(current_norm_linf.reshape(1,-1))

            adv_feature_maps[idx] = clean_feature_maps[idx]+model.w[idx]*(adv_feature_maps[idx]-clean_feature_maps[idx])
            output_list.append(model(adv_feature_maps[idx], end_point=layer_number, start_point=num))
        output_clean = model(input, end_point=layer_number, start_point=0)

        batch_norm_l2 = torch.cat(batch_norm_l2, dim=0)
        batch_norm_linf = torch.cat(batch_norm_linf, dim=0)

        norm_save_l2.append(batch_norm_l2)
        norm_save_linf.append(batch_norm_linf)

        loss_adv = 0
        for idx in range(9):
            loss_adv += criterion(output_list[idx], target)
        loss_clean = criterion(output_clean, target)
        loss_l1_penalty = torch.norm(model.w, p=1)
        
        loss = (loss_clean+loss_adv/9)/2 + loss_l1_penalty*args.l1_coef

        optimizer.zero_grad()
        optimizer_w.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_w.step()


        with torch.no_grad():
            model.w.data = sum_project(model.w.data, K=9)
        

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

    norm_mean_l2 = torch.mean(torch.cat(norm_save_l2, dim=1), dim=1)
    norm_mean_linf = torch.mean(torch.cat(norm_save_linf, dim=1), dim=1)

    print('l2 mean = {}'.format(norm_mean_l2))
    print('linf mean = {}'.format(norm_mean_linf))

    print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg, losses.avg, norm_mean_l2.numpy(), norm_mean_linf.numpy()
    
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

def sum_project(b_ori, K=9):

    b = deepcopy(b_ori)

    eq = torch.sum(b, dim=0)
    mu = (eq-1)/K

    b-=mu

    return b


if __name__ == '__main__':
    main()


