from greedy_coverage_gpu import greedy
import numpy as np
import torch

device = 'cpu'
instances = 'instances_weibo/07-04-150cas/'

N_INSTANCES, N_INFLUENCERS, N_TARGETS, N_FEATURES = 20, 500, 500, 24
w = np.ones(N_TARGETS, dtype=np.float32)

def transform_Y(Y) : 
    Yc = np.copy(Y)
    #return np.minimum(100 * Y, np.ones((N_INSTANCES, N_INFLUENCERS, N_TARGETS)))
    t1 = np.quantile(Y[Y>0], 0.2) # weak probability
    t2 = np.quantile(Y[Y>0], 0.5) # medium probability
    t3 = np.quantile(Y[Y>0], 0.8) # high probability
    Y[Yc>0] = 0.1
    Y[Yc>t1] = 0.2
    Y[Yc>t2] = 0.5
    Y[Yc>t3] = 1.
    return Y  

scores_greedy = []
for instance in range(N_INSTANCES):
    XY = np.load(instances + str(instance) +'.npz')['arr_0']
    Y = torch.Tensor(XY[:,:,-3]).to(device)
    Y = transform_Y(Y)
    score = greedy(20,Y,w, device)[0]
    print(score)
    scores_greedy.append(score)

print(f"Greedy score : {np.mean(scores_greedy)} ({np.std(scores_greedy)})")