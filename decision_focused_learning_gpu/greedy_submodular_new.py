
import torch
from torch.autograd.function import once_differentiable

#import numpy as np
class GreedyOptimizer(torch.autograd.Function):

    @staticmethod
    
    def forward(ctx, P, set_func, marginal_vec, num_items, k, eps,sample_size, beta):

        vals = torch.zeros(sample_size)
        sumlogps = torch.zeros_like(vals)
        P.retain_grad()
        
        with torch.enable_grad():
            for i in range(sample_size):
                '''
                perturbed greedy with gradient computation for backward
                '''
                S, sumlogp = [], 0
                U = torch.tensor(range(num_items))
                for _ in range(k):  
                    g =   marginal_vec(S, P)
                    if len(U) == 0: break
                    p = torch.nn.Softmax(dim=-1)(g[U]/  eps) # quand l'algo est régularisé avec l'entropie, la forme close de p est un softmax 
                    v = torch.multinomial(p, num_samples=1, replacement=True)
                    S += [int(U[v])]
                    U = torch.cat([U[:v],U[v+1:]])

                    # vidx = int(torch.nonzero(U == v)[0])
                    # U = torch.cat([U[0:vidx], U[vidx+1:]])
                    # S += [int(v)]
                    
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
        return - grad_P, None, None, None, None, None,None, None