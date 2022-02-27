
import torch
import numpy as np

class GreedyOptimizer(torch.autograd.Function):

    def __init__(self, set_func, marginal_vec, n, K, eps, sample_size, beta = 1):
        super(GreedyOptimizer, self).__init__()
        self.set_func = set_func
        self.marginal_vec = marginal_vec
        self.n = n
        self.K = K
        self.eps = eps
        self.sample_size = sample_size
        self.beta = beta
        self.softmax = torch.nn.Softmax(dim = -1)


    def forward(self, P):
        vals = torch.zeros(self.sample_size)
        sumlogps = torch.zeros_like(vals)
        P.retain_grad()
        with torch.enable_grad():
            for i in range(self.sample_size):
                '''
                perturbed greedy with gradient computation for backward
                '''
                S, sumlogp = [], 0
                U = range(self.n)
                for k in range(self.K):
                    g = self.marginal_vec(S, P)
                    if len(U) == 0: break
                    p = self.softmax(g[U]/self.eps) # quand l'algo est régularisé avec l'entropie, la forme close de p est un softmax
                    v = np.random.choice(U, p = p.detach().numpy())
                    vidx = int(np.where(U == v)[0])
                    U = np.delete(U, vidx)
                    S += [v]
                    sumlogp = sumlogp + torch.log(p[vidx])
                sumlogps[i] = sumlogp
                vals[i] = self.set_func(S)
            beta = vals.mean() if self.beta else self.beta
            fsumlogp = torch.dot(vals - beta, sumlogps) / self.sample_size
            grad = torch.autograd.grad(fsumlogp, P, retain_graph=True)[0]
            self.save_for_backward(grad)
        return vals.mean()

    def backward(self, grad_output):
        grad = self.saved_tensors[0]
        return - grad