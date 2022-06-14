
import torch
import random

class GreedyOptimizer(torch.autograd.Function):
    # @staticmethod
    # def forward(ctx, P ):
    #     vals = torch.zeros(ctx.sample_size)
    #     sumlogps = torch.zeros_like(vals)
    #     P.retain_grad()
    #     with torch.enable_grad():
    #         for i in range(ctx.sample_size):
    #             '''
    #             perturbed greedy with gradient computation for backward
    #             '''
    #             S, sumlogp = [], 0
    #             U = torch.tensor(range(ctx.n))
    #             for k in range(ctx.K):
    #                 g = ctx.marginal_vec(S, P)
    #                 if len(U) == 0: break
    #                 p = ctx.softmax(g[U]/ctx.eps) # quand l'algo est régularisé avec l'entropie, la forme close de p est un softmax 
    #                 v = torch.multinomial(p, num_samples=1, replacement=True)
    #                 S +=[int(U[v])]
    #                 U = torch.cat([U[:v],U[v+1:]])

    #                 # vidx = int(torch.nonzero(U == v)[0])
    #                 # U = torch.cat([U[0:vidx], U[vidx+1:]])
    #                 # S += [int(v)]
                    
    #                 sumlogp = sumlogp + torch.log(p[v])
    #             sumlogps[i] = sumlogp
    #             vals[i] = ctx.set_func(S)
    #         beta = vals.mean() if ctx.beta else ctx.beta
    #         fsumlogp = torch.dot(vals - beta, sumlogps) / ctx.sample_size
    #         grad = torch.autograd.grad(fsumlogp, P, retain_graph=True)[0]
    #         ctx.save_for_backward(grad)
    #     return vals.mean()


    @staticmethod
    def forward(ctx, P, true_set_func, marginal_vec_pred, n,  K,  eps, sample_size, beta):
        vals = torch.zeros(sample_size)
        sumlogps = torch.zeros_like(vals)
        P.retain_grad()
        with torch.enable_grad():
            for i in range(sample_size): #sample size controls the number of time we execute the diff greedy algorithm
                '''
                perturbed greedy with gradient computation for backward
                '''
                S, sumlogp = [], 0
                U = torch.tensor(range(n))
                for k in range(K):
                    g = marginal_vec_pred(S, P)
                    if len(U) == 0: break
                    p = torch.nn.functional.softmax(g[U]/eps) # quand l'algo est régularisé avec l'entropie, la forme close de p est un softmax 
                    v = torch.multinomial(p, num_samples=1, replacement=True)
                    S +=[int(U[v])]
                    U = torch.cat([U[:v],U[v+1:]])

                    # vidx = int(torch.nonzero(U == v)[0])
                    # U = torch.cat([U[0:vidx], U[vidx+1:]])
                    # S += [int(v)]
                    
                    sumlogp = sumlogp + torch.log(p[v])
                sumlogps[i] = sumlogp
                vals[i] = true_set_func(S)
            beta = vals.mean() if beta else beta
            fsumlogp = torch.dot(vals - beta, sumlogps) / sample_size
            grad = torch.autograd.grad(fsumlogp, P, retain_graph=True)[0]
            ctx.save_for_backward(grad)
        return vals.mean()

    @staticmethod
    def backward(ctx, grad_output):
        grad = ctx.saved_tensors[0]
        return - grad

class StochasticGreedyOptimizer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, P, true_set_func, partial_marginal_vec, n,  K,  eps, sample_size, beta, nk):
        vals = torch.zeros(sample_size)
        sumlogps = torch.zeros_like(vals)
        P.retain_grad()
        with torch.enable_grad():
            for i in range(sample_size): #sample size controls the number of time we execute the diff greedy algorithm
                '''
                perturbed greedy with gradient computation for backward
                '''
                S, sumlogp = [], 0
                U = torch.tensor(range(n))
                for k in range(K):
                    
                    sample_idxs = random.sample(range(len(U)), min(nk, len(U)))
                    sample_idxs = torch.tensor(sample_idxs)
                    Uk = U[sample_idxs]

                    g = partial_marginal_vec(S, Uk, P) #only contains marginal gains of Uk
                    if len(U) == 0: break
                    p = torch.nn.functional.softmax(g[U]/eps) 
                    id_v = torch.multinomial(p, num_samples=1, replacement=True) #returns the position in p
                    v = sample_idxs[id_v] #converts it into position in U
                    S +=[int(U[v])] 
                    U = torch.cat([U[:v],U[v+1:]])

                    # vidx = int(torch.nonzero(U == v)[0])
                    # U = torch.cat([U[0:vidx], U[vidx+1:]])
                    # S += [int(v)]
                    
                    sumlogp = sumlogp + torch.log(p[v])
                sumlogps[i] = sumlogp
                vals[i] = true_set_func(S)
            beta = vals.mean() if beta else beta
            fsumlogp = torch.dot(vals - beta, sumlogps) / sample_size
            grad = torch.autograd.grad(fsumlogp, P, retain_graph=True)[0]
            ctx.save_for_backward(grad)
        return vals.mean()

    @staticmethod
    def backward(ctx, grad_output):
        grad = ctx.saved_tensors[0]
        return - grad

