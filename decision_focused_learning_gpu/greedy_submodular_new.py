
import torch
from torch.autograd.function import once_differentiable
import random

#import numpy as np
class GreedyOptimizer(torch.autograd.Function):

    @staticmethod
    
    def forward(ctx, P, set_func, marginal_vec, num_items, k, eps,sample_size, beta, device):

        vals = torch.zeros(sample_size).to(device)
        sumlogps = torch.zeros_like(vals).to(device)
        P.retain_grad()
        
        with torch.enable_grad():
            for i in range(sample_size):
                '''
                perturbed greedy with gradient computation for backward
                '''
                S, sumlogp = [], 0
                U = torch.tensor(range(num_items)).to(device)
                for _ in range(k):  
                    g =   marginal_vec(S, P)
                    if len(U) == 0: break
                    p = torch.nn.Softmax(dim=-1)(g[U]/  eps) # quand l'algo est régularisé avec l'entropie, la forme close de p est un softmax 
                    v = torch.multinomial(p, num_samples=1, replacement=True)
                    S += [int(U[v])]
                    U = torch.cat([U[:v],U[v+1:]])

                    sumlogp = sumlogp + torch.log(p[v])
                sumlogps[i] = sumlogp
                vals[i] =   set_func(S)
            beta = vals.mean() if   beta else   beta
            fsumlogp = torch.dot(vals - beta, sumlogps) / sample_size
            grad = torch.autograd.grad(fsumlogp, P, retain_graph=True)[0]
            ctx.save_for_backward(grad)
        return vals.mean()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_P = ctx.saved_tensors[0]
        return - grad_P, None, None, None, None, None,None, None, None

class StochasticGreedyOptimizer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, P, true_set_func, partial_marginal_vec, n,  K,  eps, sample_size, beta, nk, device):
        vals = torch.zeros(sample_size).to(device)
        sumlogps = torch.zeros_like(vals).to(device)
        P.retain_grad()
        with torch.enable_grad():
            for i in range(sample_size): #sample size controls the number of time we execute the diff greedy algorithm

                S, sumlogp = [], 0
                U = torch.tensor(range(n))
                for k in range(K):
                    
                    sample_idxs = random.sample(range(len(U)), min(nk, len(U)))
                    sample_idxs = torch.tensor(sample_idxs)
                    Uk = U[sample_idxs]
                    # print(sample_idxs, Uk)
                    gvec = partial_marginal_vec(S, Uk, P) 
                    # print(len(gvec))
                    if len(U) == 0: break
                    p = torch.nn.functional.softmax(gvec/eps) 
                    id_v = torch.multinomial(p, num_samples=1, replacement=True) #returns the position in p
                    v = sample_idxs[id_v] #converts it into position in U
                    S +=[int(U[v])] 
                    U = torch.cat([U[:v],U[v+1:]])
                    sumlogp = sumlogp + torch.log(p[id_v])

                sumlogps[i] = sumlogp
                vals[i] = true_set_func(S)

            beta = vals.mean() if beta else beta
            fsumlogp = torch.dot(vals - beta, sumlogps) / sample_size
            grad = torch.autograd.grad(fsumlogp, P, retain_graph=True)[0]
            ctx.save_for_backward(grad)
        return vals.mean()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad = ctx.saved_tensors[0]
        return - grad, None, None, None, None, None,None, None, None, None