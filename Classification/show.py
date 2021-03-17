import os
import pickle
import argparse
import numpy as np 


parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--save_dir', help='The directory used to save the trained models', default='adv', type=str)
args = parser.parse_args()


data = pickle.load(open(os.path.join(args.save_dir, 'result.pkl'),'rb'))

test = np.array(data['test_ta'])
val = np.array(data['ta'])

idx = np.argmax(val)
print('epoch:', idx+1, 'TA:', test[idx])
idx = np.argmax(test)
print(' Test epoch:', idx+1, 'TA:', test[idx])











