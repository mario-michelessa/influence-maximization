import os
import warnings
warnings.simplefilter('ignore')

import numpy as np
import random
from functools import partial

import torch
import torch.nn as nn

from predictive_model import make_fc

from greedy_coverage import set_func, marginal_vec, greedy
from greedy_submodular import GreedyOptimizer 

# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--beta',   type = int,   default = 1)
# parser.add_argument('--sample', type = int,   default = 10)
# parser.add_argument('--eps',    type = float, default = .2)
# parser.add_argument('--k',      type = int, default = 5)

# args = parser.parse_args()
# beta = args.beta
# sample_size = args.sample
# eps = args.eps
# k = args.k

beta = 1
sample_size = 10
eps = 0.2
k = 5

num_items = 100
num_targets = 500

num_random_iter = 10
num_instances = 100
num_features = 43

num_epochs = 5
batch_size = 100
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
    
    net = make_fc(num_features, num_layers, activation, intermediate_size)
    net.apply(init_weights)

    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)

    for epoch in range(num_epochs):
        print('epoch{}'.format(epoch))
        data_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True)
        for X_batch, P_batch in data_loader:
            loss = 0 
            for X, P in zip(X_batch, P_batch):
                true_set_func = partial(set_func, P = P, w = w)
                marginal_vec_pred = partial(marginal_vec, w = w)
                pred = net(X).view_as(P)                
                fn = GreedyOptimizer(true_set_func, marginal_vec_pred, n = num_items, K = k, eps = eps, sample_size = sample_size, beta = beta)
                loss -= fn(pred)
            loss = loss / batch_size
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        

    def eval_grd(net, instances):
        return np.mean([set_func(greedy(k, net(data[i]).view_as(Ps[0]), w)[1], Ps[i], w) for i in instances])

    train_score = eval_grd(net, train)
    test_score  = eval_grd(net, test)

    print(train_score)
    print(test_score)

    train_scores.append(train_score)
    test_scores.append(test_score)

print(np.mean(train_scores))
print(np.mean(test_scores))


### save results ###
path = 'results/'
fname = 'k{}_grd_b{}_s{}.npz'.format(k, beta, sample_size)
if os.path.exists(path + fname) == False:
    idxs = range(num_random_iter)
    train_scores = np.array(train_scores)
    test_scores  = np.array(test_scores)
    np.savez(path + fname, idxs = idxs, train_scores = train_scores, test_scores = test_scores)
