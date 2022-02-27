import os
import warnings
warnings.simplefilter('ignore')

import numpy as np
import random
from functools import partial

import torch
import torch.nn as nn

from greedy_coverage import set_func, greedy
from continuous_coverage import optimize_coverage_multilinear, CoverageInstanceMultilinear, dgrad_coverage, hessian_coverage
from continuous_submodular import ContinuousOptimizer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--k',      type = int,   required = True)
args = parser.parse_args()
k = args.k

num_items = 100
num_targets = 500

num_random_iter = 30
num_instances = 100
num_features = 43

num_epochs = 5
batch_size = 20
learning_rate = 1e-3

num_layers = 2
activation = 'relu'
intermediate_size = 200

def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.uniform_(.0, .01)


### instances ###
path = 'instances/'
Ps = torch.from_numpy(np.load(path + 'Ps.npz')['Ps']).float()      
data = torch.from_numpy(np.load(path + 'data.npz')['data']).float()
trains, tests = [], []
for i in range(num_random_iter):
    trains.append(np.load(path + '{}.npz'.format(i))['train'])
    tests.append(np.load(path + '{}.npz'.format(i))['test'])
w = np.ones(num_targets, dtype=np.float32)


### main ###
train_scores = []
test_scores  = []
for idx in range(num_random_iter):
    print(idx)
    test  = tests[idx] 
    train = trains[idx]
    dataset = torch.utils.data.TensorDataset(data[train], Ps[train]) 
     
    def eval_rnd(instances):
        sol = random.sample(range(num_items),k)   
        return np.mean([set_func(sol, Ps[i], w) for i in instances])

    train_score = eval_rnd(train)
    test_score  = eval_rnd(test)

    print(train_score)
    print(test_score)

    train_scores.append(train_score)
    test_scores.append(test_score)

print(np.mean(train_scores))
print(np.mean(test_scores))


### save results ###
path = 'results/'
fname = 'k{}_rnd.npz'.format(k)
if os.path.exists(path + fname) == False:
    idxs = range(num_random_iter)
    train_scores = np.array(train_scores)
    test_scores  = np.array(test_scores)
    np.savez(path + fname, idxs = idxs, train_scores = train_scores, test_scores = test_scores)