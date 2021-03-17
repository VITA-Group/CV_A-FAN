
import os
import pickle
import argparse
import numpy as np 
import matplotlib.pyplot as plt 

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--save_dir', help='The directory used to save the trained models', default='adv', type=str)
parser.add_argument('--save', help='The directory used to save the trained models', default='adv', type=str)
parser.add_argument('--num', default=1, type=int)
args = parser.parse_args()

data = pickle.load(open(os.path.join(args.save_dir, 'result_norm.pkl'),'rb'))

result = {}
for i in range(args.num):
    result[i] = []

for key in data.keys():
    epoch_norm = data[key]
    for i in range(args.num):
        result[i].append(epoch_norm[i])

for i in range(args.num):
    plt.plot(result[i], label='l2_perturbation_'+str(i+1))

plt.legned()
plt.savefig(args.save)

