import torch
from numba import cuda

def set_func(S, P, w):
    '''
    outputs set function value.
    '''
    w = torch.Tensor(w)
    s = torch.zeros(P.shape[0])
    s[S] = 1
    return float(torch.dot(w, 1 - torch.prod(1 - (s*P.T).T, axis = 0)))

def marginal_vec(S, P, w):
    '''
    outputs marginal gains of all elements for given S.
    '''
    w = torch.Tensor(w)
    s = torch.zeros(P.shape[0])
    s[S] = torch.ones(len(S))
    sc = torch.ones_like(s) - s
    return sc * torch.mv(P, w * torch.prod(1 - (s*P.T).T, axis = 0)) #mv = matrix vector multiplication, normal *   
    
def greedy(K, P, w):
    '''
    standard greedy algorithm for test instances. 
    '''
    val = 0
    S = []
    for i in range(K):
        g = marginal_vec(S, P, w)
        s = int(torch.argmax(g))
        if s not in S: S += [s]
        val += g[s]
    return val, S
    