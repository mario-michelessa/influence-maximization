import os
import warnings
warnings.simplefilter('ignore')

import numpy as np
import random
from functools import partial

import torch
import torch.nn as nn

from predictive_model import make_fc
from greedy_coverage import greedy

import time

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--beta',   type = int,   default = 1)
parser.add_argument('--sample', type = int,   default = 10)
parser.add_argument('--eps',    type = float, default = .2)
parser.add_argument('--nk',     type = int,   required = True)


args = parser.parse_args()
beta = args.beta
sample_size = args.sample
eps = args.eps
nk = args.nk

num_items = 100
num_targets = 500
K = 10

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


### road instances ###
path = 'instances/'
Ps = torch.from_numpy(np.load(path + 'Ps.npz')['Ps']).float()      
data = torch.from_numpy(np.load(path + 'data.npz')['data']).float()
trains, tests = [], []
for i in range(num_random_iter):
    trains.append(np.load(path + '{}.npz'.format(i))['train'])
    tests.append(np.load(path + '{}.npz'.format(i))['test'])
w = np.ones(num_targets, dtype=np.float32)

### set_func and marginal_vec ###
def set_func(S, P, w):
    w = torch.Tensor(w)
    return float(torch.dot(w, 1 - torch.prod(1 - P[S], axis = 0)))

def partial_marginal_vec(S, U, P, w):
    #assume S \cap U to be empty
    w = torch.Tensor(w)
    return torch.mv(P[U], w * torch.prod(1 - P[S], axis = 0))
partial_marginal_vec_pred = partial(partial_marginal_vec, w = w)


### main ###
train_scores = []
test_scores  = []
sgrd_time    = []
grad_time    = []

for idx in range(num_random_iter):
    print(idx)
    test  = tests[idx] 
    train = trains[idx]
    dataset = torch.utils.data.TensorDataset(data[train], Ps[train]) 
    
    net = make_fc(num_features, num_layers, activation, intermediate_size)
    net.apply(init_weights)

    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)

    n = num_items
    softmax = torch.nn.Softmax(dim = -1)
    
    for epoch in range(num_epochs):
        print('epoch{}'.format(epoch))
        data_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True)
        for X_batch, P_batch in data_loader:
            loss = 0 
            for X, P in zip(X_batch, P_batch):
                true_set_func = partial(set_func, P = P, w = w)
                pred = net(X).view_as(P)

                ### iteration over N trials ###
                vals = torch.zeros(sample_size)
                sumlogps = torch.zeros_like(vals)
                pred.retain_grad()
                with torch.enable_grad():
                    for i in range(sample_size):
                        
                        ### smoothed stoachstic greedy ###                        
                        start = time.time()

                        S, sumlogp = [], 0
                        ps = torch.zeros(K)
                        U = np.arange(n)
                        for k in range(K):
                            sample_idxs = np.random.choice(np.arange(len(U)), min(nk, len(U)), replace = False)
                            Uk = U[sample_idxs]
                            gvec = partial_marginal_vec_pred(S, Uk, pred)
                            pvec = softmax(gvec / eps)
                            sk_idx = np.random.choice(sample_idxs, p = pvec.detach().numpy())
                            sk = U[sk_idx]
                            ps[k] = pvec[int(np.where(Uk == sk)[0])]
                            S += [sk]
                            U = np.delete(U, sk_idx)

                        sgrd_time.append(time.time() - start)                     

                        ### keep sum of ln(p_k) ###
                        sumlogps[i] = torch.sum(torch.log(ps))

                        ### computation of Q(S) value ###
                        vals[i] = true_set_func(S)        

                    ### baseline correction ###
                    baseline = vals.mean() if beta else beta
                    fsumlogp = torch.dot(vals - baseline, sumlogps) / sample_size

                loss -= fsumlogp
            loss = loss / batch_size
            print('objective on a mini-batch:', float(vals.mean()))

            ### differentiation ###
            start = time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            grad_time.append(time.time() - start)

        

    def eval_grd(net, instances):
        return np.mean([set_func(greedy(K, net(data[i]).view_as(Ps[0]), w)[1], Ps[i], w) for i in instances])

    train_score = eval_grd(net, train)
    test_score  = eval_grd(net, test)

    print(train_score)
    print(test_score)

    train_scores.append(train_score)
    test_scores.append(test_score)

print(np.mean(train_scores))
print(np.mean(test_scores))

print(np.mean(np.array(sgrd_time)))
print(np.mean(np.array(grad_time)))


### save results ###
path = 'results/'
fname = 'nk{}.npz'.format(nk)
if os.path.exists(path + fname) == False:
    idxs = range(num_random_iter)
    train_scores = np.array(train_scores)
    test_scores  = np.array(test_scores)
    sgrd_time = np.array(sgrd_time)
    grad_time = np.array(grad_time)
    np.savez(path + fname, idxs = idxs, train_scores = train_scores, test_scores = test_scores, sgrd_time = sgrd_time, grad_time = grad_time)
