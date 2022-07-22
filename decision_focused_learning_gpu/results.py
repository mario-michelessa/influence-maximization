"""
Tests a list of model from grd_main on instances
And saves results in .txt file
"""

import torch
import numpy as np
from greedy_coverage_gpu import greedy, set_func
import random 

device = 'cpu'

models_path = "results/4-150/"
log_path = models_path + "perf_sparse.txt"

model_df_name = "net_df_20_5_db"
model_2s_name = "net_2s_20_5_db"

instances_path = "instances_weibo/06-30-sparse/"

LABEL = 'DB'
N_INSTANCES, N_INFLUENCERS, N_TARGETS, N_FEATURES = 1, 1000, 1000, 19 #size of instances
n_iter = 15 # number of different models in the folder

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

def create_XY(instance_path):
    X = np.zeros((N_INSTANCES, N_INFLUENCERS, N_TARGETS, N_FEATURES))
    Y = np.zeros((N_INSTANCES, N_INFLUENCERS, N_TARGETS))
    Ydb = np.zeros((N_INSTANCES, N_INFLUENCERS, N_TARGETS))

    for instance in range(N_INSTANCES) :
        XY = np.load(instance_path + f"{instance}.npz")['arr_0']
        X[instance] = XY[:,:,:-3]
        if LABEL == 'DB' :
            Y[instance] = XY[:,:,-3]
        elif LABEL == 'INFECTOR' :
            Y[instance] = XY[:,:,-2]
        else :
            Y[instance] = XY[:,:,-1]
        
        Ydb[instance] = XY[:,:,-3]

    if LABEL == 'DB' : Y = transform_Y(Y)
    Ydb = transform_Y(Ydb)
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()
    Ydb = torch.from_numpy(Ydb).float()
    return X, Y, Ydb

def dni(seeds, Y) : 
    return (Y[seeds,:].sum(dim=0) > 0).sum().item() 

def highest_degrees(X, k) : 
    return list(X[:, 0, 5].argsort(descending=True).numpy()[:k])

def results_model(logs, net, name):
    """
    returns lists of influence in terms of DNI and expectation for different values of k given a model
    """
    
    exps = []
    dnis = []
    for k in Ks :
        exps.append( np.mean([   set_func(   greedy(k, net(X[i]).view_as(Y[0]), w, device)[1], Y[i], w, device) for i in range(X.shape[0])]) )
        dnis.append( np.mean([   dni(        greedy(k, net(X[i]).view_as(Y[0]), w, device)[1], Ydb[i]) for i in range(X.shape[0])]) )
    logs.write(f"Exp {name}, " + ",".join(map(str,exps)) + "\n")
    logs.write(f"DNI {name}, " + ",".join(map(str,dnis)) + "\n")

def results_rdn_grd_deg(logs) : 
    exps_rnd, dnis_rnd = [], []
    exps_grd, dnis_grd = [], []
    exps_deg, dnis_deg = [], []
    
    for k in Ks :
        exps_rnd.append(np.mean([set_func(random.sample(range(N_INFLUENCERS),k), Y[i], w, device) for i in range(N_INSTANCES)]))
        dnis_rnd.append(np.mean([dni(     random.sample(range(N_INFLUENCERS),k), Ydb[i]) for i in range(N_INSTANCES)]))

        exps_grd.append(np.mean([greedy(k, Y[i], w, device)[0].item() for i in range(N_INSTANCES)]))
        dnis_grd.append(np.mean([dni(   greedy(k, Ydb[i], w, device)[1], Ydb[i]) for i in range(N_INSTANCES)]))

        exps_deg.append(np.mean([set_func(highest_degrees(X[i], k), Y[i], w, device) for i in range(N_INSTANCES)]))
        dnis_deg.append(np.mean([dni(   highest_degrees(X[i], k), Ydb[i]) for i in range(N_INSTANCES)]))
    
    logs.write(f"Exp rnd, " + ",".join(map(str,exps_rnd)) + "\n")
    logs.write(f"DNI rnd, " + ",".join(map(str,dnis_rnd)) + "\n")
    logs.write(f"Exp grd, " + ",".join(map(str,exps_grd)) + "\n")
    logs.write(f"DNI grd, " + ",".join(map(str,dnis_grd)) + "\n")
    logs.write(f"Exp deg, " + ",".join(map(str,exps_deg)) + "\n")
    logs.write(f"DNI deg, " + ",".join(map(str,dnis_deg)) + "\n")
    
if __name__ == "__main__" : 

    w = torch.ones(N_INFLUENCERS)
    X, Y, Ydb = create_XY(instances_path)

    if X.shape[3] == 19 : X = torch.cat((X[:,:,:,:6], torch.zeros((N_INSTANCES, N_INFLUENCERS, N_TARGETS,1)), X[:,:,:,6:15], torch.zeros((N_INSTANCES, N_INFLUENCERS, N_TARGETS,1)),  X[:,:,:,15:]), dim=3)

    Ks = [int(k * N_INFLUENCERS) for k in [0.01, 0.02, 0.05, 0.1, 0.2]] 
    logs = open(log_path, "w")

    for n in range(n_iter) : 
        model_df = torch.load(models_path + f"{model_df_name}_{n}.pt").to('cpu')
        model_2s = torch.load(models_path + f"{model_2s_name}_{n}.pt").to('cpu')    

        print(f'Testing models {n}')
        results_model(logs, model_df, "df")
        results_model(logs, model_2s, "2s")
        results_rdn_grd_deg(logs)

    logs.close()
