
import torch
#import numpy as np

# class GreedyContext():

#     def __init__(self, set_func, marginal_vec, n, K, eps, sample_size, beta = 1):
        
#         self.set_func = set_func
#         self.marginal_vec = marginal_vec
#         self.n = n
#         self.K = K
#         self.eps = eps
#         self.sample_size = sample_size
#         self.beta = beta
#         self.softmax = torch.nn.Softmax(dim = -1)
class GreedyOptimizer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, P):
        vals = torch.zeros(ctx.sample_size)
        sumlogps = torch.zeros_like(vals)
        P.retain_grad()
        with torch.enable_grad():
            for i in range(ctx.sample_size):
                '''
                perturbed greedy with gradient computation for backward
                '''
                S, sumlogp = [], 0
                U = torch.tensor(range(ctx.n))
                for k in range(ctx.K):
                    g = ctx.marginal_vec(S, P)
                    if len(U) == 0: break
                    p = ctx.softmax(g[U]/ctx.eps) # quand l'algo est régularisé avec l'entropie, la forme close de p est un softmax 
                    v = torch.multinomial(p, num_samples=1, replacement=True)
                    S +=[int(U[v])]
                    U = torch.cat([U[:v],U[v+1:]])

                    # vidx = int(torch.nonzero(U == v)[0])
                    # U = torch.cat([U[0:vidx], U[vidx+1:]])
                    # S += [int(v)]
                    
                    sumlogp = sumlogp + torch.log(p[v])
                sumlogps[i] = sumlogp
                vals[i] = ctx.set_func(S)
            beta = vals.mean() if ctx.beta else ctx.beta
            fsumlogp = torch.dot(vals - beta, sumlogps) / ctx.sample_size
            grad = torch.autograd.grad(fsumlogp, P, retain_graph=True)[0]
            ctx.save_for_backward(grad)
        return vals.mean()

    @staticmethod
    def backward(ctx, grad_output):
        grad = ctx.saved_tensors[0]
        return - grad