import os
import warnings
warnings.simplefilter('ignore')

import numpy as np
import random
from functools import partial

import torch
import torch.nn as nn

from predictive_model import make_fc

from greedy_coverage import set_func
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
        
    net = make_fc(num_features, num_layers, activation, intermediate_size)
    net.apply(init_weights)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)

    for epoch in range(num_epochs):
        print('epoch{}'.format(epoch))
        data_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True)
        for X_batch, P_batch in data_loader:
            loss = 0 
            for X, P in zip(X_batch, P_batch):
                pred = net(X).view_as(P)
                loss += loss_fn(pred, Ps[i])
            loss = loss / batch_size
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    optfunc = partial(optimize_coverage_multilinear, w = w, k=k, c = 0.95)
    dgrad = partial(dgrad_coverage, w = w)
    hessian = partial(hessian_coverage, w = w)
    opt = ContinuousOptimizer(optfunc, dgrad, hessian, 0.95)
    opt.verbose = False

    def eval_two_stage(net, instances):
        return np.mean([set_func(np.argsort(- opt(net(data[i]).view_as(Ps[0])).detach().numpy())[:k], Ps[i], w) for i in instances])

    train_score = eval_two_stage(net, train)
    test_score  = eval_two_stage(net, test)

    print(train_score)
    print(test_score)

    train_scores.append(train_score)
    test_scores.append(test_score)

print(np.mean(train_scores))
print(np.mean(test_scores))


### save results ###
path = 'results/'
fname = 'k{}_two_stage.npz'.format(k)
if os.path.exists(path + fname) == False:
    idxs = range(num_random_iter)
    train_scores = np.array(train_scores)
    test_scores  = np.array(test_scores)
    np.savez(path + fname, idxs = idxs, train_scores = train_scores, test_scores = test_scores)