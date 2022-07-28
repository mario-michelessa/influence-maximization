CUDA_LAUNCH_BLOCKING="1"

import os 
import warnings
warnings.simplefilter('ignore')

import numpy as np
import random
from functools import partial

import torch
import torch.nn as nn

from predictive_model_gpu import make_fc
from greedy_coverage_gpu import set_func, marginal_vec, greedy
from greedy_submodular_new import GreedyOptimizer, StochasticGreedyOptimizer

import argparse
import datetime
 
### To edit
instance_path = "data/instances_weibo/07-04-150Cas/"
N_INSTANCES = 20
N_INFLUENCERS = 500
N_TARGETS = 500
N_FEATURES = 24
N_TRAIN = int(0.8 * N_INSTANCES)


beta = 1                                        # Parameter of greedy optimizer, default 1
sample_size = 10                                # Parameter of greedy optimizer, default 10
eps = 0.15                                      # Parameter of greedy optimizer, default 0.1
k = 20                                          # Parameter of greedy optimizer, default 5
activation = 'swish'                            # Activation function of the model, 'swish' or 'relu' or 'sigmoid'
dropout = 0.15                                  # Dropout proportion of the model
phi = 0.5                                       # Parameter of the unbalanced loss of the 2-staged model
q1, q2, q3 = 0.2, 0.5, 0.8                      # quantiles for decision-based ground_truth probabilities
low_p, med_p, high_p = 0.2, 0.5, 1              # values for decision-based ground_truth probabilities
momentum = 0.8                                  # Momentum of Adam optimizer
num_layers = 2                                  # Number of layers of the model
hidden_sizes = [240]                            # Sizes of the hidden layers

# Instances definition
def transform_Y(Y) : 
    """ Maps the values of decision-based labels into higher values"""
    Yc = np.copy(Y)
    t1 = np.quantile(Y[Y>0], q1) # weak probability
    t2 = np.quantile(Y[Y>0], q2) # medium probability
    t3 = np.quantile(Y[Y>0], q3) # high probability
    Y[Yc>0] = 0.1
    Y[Yc>t1] = low_p
    Y[Yc>t2] = med_p
    Y[Yc>t3] = high_p
    return Y  

def create_train_test(train_id):
    """
    input : train_id = subset of range(N_INSTANCES)
    returns X_train, Y_train, X_test, Y_test
    """

    X_train = np.zeros((N_TRAIN, N_INFLUENCERS, N_TARGETS, N_FEATURES))
    Y_train = np.zeros((N_TRAIN, N_INFLUENCERS, N_TARGETS))
    Y2_train = np.zeros((N_TRAIN, N_INFLUENCERS, N_TARGETS))
    
    X_test = np.zeros((N_INSTANCES - N_TRAIN, N_INFLUENCERS, N_TARGETS, N_FEATURES))
    Y_test = np.zeros((N_INSTANCES - N_TRAIN, N_INFLUENCERS, N_TARGETS))
    Y2_test = np.zeros((N_INSTANCES - N_TRAIN, N_INFLUENCERS, N_TARGETS))
    
    c_train, c_test = 0, 0
    for instance in range(N_INSTANCES) :

        XY = np.load(instance_path + f"{instance}.npz")['arr_0']
        if instance in train_id :

            X_train[c_train] = XY[:,:,:-3]
            Y_train[c_train] = XY[:,:,-3]
            if labels == "infector" : Y2_train[c_train] = XY[:,:,-2]
            elif labels == "inf2vec" : Y2_train[c_train] = XY[:,:,-1]
            c_train += 1
        
        else : 
            X_test[c_test] = XY[:,:,:-3]
            Y_test[c_test] = XY[:,:,-3]
            if labels == "infector" : Y2_test[c_test] = XY[:,:,-2]
            elif labels == "inf2vec" : Y2_test[c_test] = XY[:,:,-1]
            c_test += 1
    
    Y_train = transform_Y(Y_train)
    Y_test = transform_Y(Y_test)

    X_train = torch.from_numpy(X_train).float()
    Y_train = torch.from_numpy(Y_train).float()
    Y2_train = torch.from_numpy(Y2_train).float()

    X_test = torch.from_numpy(X_test).float()
    Y_test = torch.from_numpy(Y_test).float()
    Y2_test = torch.from_numpy(Y2_test).float()

    return X_train, Y_train, X_test, Y_test, Y2_train, Y2_test

# Metrics function
def dni(seeds, Y) : 
    """
    estimates the number of Different Nodes Infected given a seed set and a DATA-BASED label. 
        If a data-based label of (u,v) is positive, it means that v appears in at least one cascade of u
    for each column(target) if one of the labels is >0 then it appears in a cascade provoqued by one of the seeds
    """
    return (Y[seeds,:].sum(dim=0) > 0).sum().item() 

def eval_dni(net, X, Y):
    """
    input : X[instances, nI, nT, nF], Y[instances, nI, nT]
    estimates number of different nodes infected based on the cascades
    """
    result = np.mean([  dni(    greedy(k, net(X[i,:,:,:].to(device)).view_as(Y_train[0]), w, device)[1]    ,Y[i].to(device)) for i in range(X.shape[0])])
    return float(result) #float is here to delete the gradient if there is a torch gradient

def eval_spread(net, X, Y, k=k):
    """  estimates expectation of dni in the bipartite graph: sum(1-prod(1-p_uv))  """
    result = np.mean([    set_func(   greedy(k, net(X[i,:,:,:].to(device)).view_as(Y_train[0]), w, device)[1], Y[i, :, :].to(device), w, device) for i in range(X.shape[0])])
    return float(result)

# Training functions 

def loss_unbalanced(pred, P) : 
    """ to counter the sparsity of the data, when the model predicts lower probabilities than the ground truth, the loss is higher """
    return torch.sum(torch.exp(phi * (P - pred)) - phi * (P-pred) - 1)

def reg_avg(P) : 
    """Regularization term"""
    return reg_coeff * (0.02 * 1 / torch.mean(P) + 1 / (1 - torch.mean(P)))
    # torch.pow(torch.tan((torch.mean(P) + 0.5) * np.pi) + 1, 2)

def init_weights_df(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

def init_weights_2s(m):
    if type(m) == nn.Linear:
        m.weight.data.uniform_(-.5, .5)

#Baseline functions 

def highest_degrees(X, k) : 
    """ returns the k influencers having the highest ingoing degree in the social graph"""
    return list(X[:, 0, 5].argsort(descending=True).to('cpu').numpy()[:k])

def baselines() :
    """ Writes the performances of the random, greedy and degree heuristic in baseline.txt"""
    log_rnd = open(output_dir + 'baseline.txt', 'a')
    log_rnd.write(f'{n} - {date} \n')
    start = datetime.datetime.now()

    rd_score_xtrain = [set_func(random.sample(range(N_INFLUENCERS),k), Y_train[i].to(device), w, device) for i in range(Y_train.shape[0])]
    rd_score_xtest = [set_func(random.sample(range(N_INFLUENCERS),k), Y_test[i].to(device), w, device) for i in range(Y_test.shape[0])]

    greedy_train = [greedy(k, Y_train[i].to(device), w, device)[0].item() for i in range(Y_train.shape[0])]
    greedy_test = [greedy(k, Y_test[i].to(device), w, device)[0].item() for i in range(Y_test.shape[0])]

    deg_train = [set_func(highest_degrees(X_train[i], k), Y_train[i], w, device) for i in range(Y_train.shape[0])]
    deg_test = [set_func(highest_degrees(X_test[i], k), Y_test[i], w, device) for i in range(Y_test.shape[0])]
    
    print(f"Average random score : {np.mean(rd_score_xtrain)}({np.std(rd_score_xtrain)}) / {np.mean(rd_score_xtest)}({np.std(rd_score_xtest)})")
    print(f"Oracle greedy score  : {np.mean(greedy_train)}({np.std(greedy_train)}) / {np.mean(greedy_test)}({np.std(greedy_test)})")
    print(f"Average degree score : {np.mean(deg_train)}({np.std(deg_train)}) / {np.mean(deg_test)}({np.std(deg_test)})")
    
    log_rnd.write(f"{rd_score_xtrain} / {rd_score_xtest} \n")
    log_rnd.write(f"{greedy_train} / {greedy_test} \n")
    log_rnd.write(f"{deg_train} / {deg_test} \n")
    log_rnd.write(f"Runtime : {datetime.datetime.now() - start} \n \n")
    log_rnd.close()

def results_model(logs, net, name):
    """ Writes the performances (Spread/DNI on training/testing data) of net in logs"""  
    exps_train, exps_test = [], []
    dnis_train, dnis_test = [], []
    for k in Ks :
        exps_train.append(eval_spread(net, X_train, Y_train, k))
        exps_test.append(eval_spread(net, X_test, Y_test, k))
        dnis_train.append(eval_dni(net, X_train, Y_train))
        dnis_test.append(eval_dni(net, X_test, Y_test))
    logs.write(f"Exp-train {name}, " + ",".join(map(str,exps_train)) + "\n")
    logs.write(f"Exp-test {name}, " + ",".join(map(str,exps_test)) + "\n")
    logs.write(f"DNI-train {name}, " + ",".join(map(str,dnis_train)) + "\n")
    logs.write(f"DNI-test {name}, " + ",".join(map(str,dnis_test)) + "\n")

def results_rdn_grd_deg(logs) : 
    """ Writes the performances (Spread/DNI on training/testing data) of the random, greedy and degree heuristic in logs"""
    exps_train_rnd, exps_test_rnd = [], []
    dnis_train_rnd, dnis_test_rnd = [], []
    exps_train_grd, exps_test_grd = [], []
    dnis_train_grd, dnis_test_grd = [], []
    exps_train_deg, exps_test_deg = [], []
    dnis_train_deg, dnis_test_deg = [], []
    
    for k in Ks :
        exps_train_rnd.append(np.mean([set_func(random.sample(range(N_INFLUENCERS),k), Y_train[i], w, device) for i in range(N_TRAIN)]))
        exps_test_rnd.append(np.mean([ set_func(random.sample(range(N_INFLUENCERS),k), Y_test[i], w, device) for i in range(N_INSTANCES - N_TRAIN)]))
        dnis_train_rnd.append(np.mean([dni(     random.sample(range(N_INFLUENCERS),k), Y_train[i]) for i in range(N_TRAIN)]))
        dnis_test_rnd.append(np.mean([ dni(     random.sample(range(N_INFLUENCERS),k), Y_test[i]) for i in range(N_INSTANCES - N_TRAIN)]))

        exps_train_grd.append(np.mean([greedy(k, Y_train[i], w, device)[0].item() for i in range(N_TRAIN)]))
        exps_test_grd.append(np.mean([ greedy(k, Y_test[i], w, device)[0].item() for i in range(N_INSTANCES - N_TRAIN)]))
        dnis_train_grd.append(np.mean([dni(   greedy(k, Y_train[i], w, device)[1], Y_train[i]) for i in range(N_TRAIN)]))
        dnis_test_grd.append(np.mean([ dni(   greedy(k, Y_test[i], w, device)[1], Y_test[i]) for i in range(N_INSTANCES - N_TRAIN)]))

        exps_train_deg.append(np.mean([set_func(highest_degrees(X_train[i], k), Y_train[i], w, device) for i in range(N_TRAIN)]))
        exps_test_deg.append(np.mean([ set_func(highest_degrees(X_test[i], k), Y_test[i], w, device) for i in range(N_INSTANCES - N_TRAIN)]))
        dnis_train_deg.append(np.mean([dni(   highest_degrees(X_train[i], k), Y_train[i]) for i in range(N_TRAIN)]))
        dnis_test_deg.append(np.mean([ dni(   highest_degrees(X_test[i], k), Y_test[i]) for i in range(N_INSTANCES - N_TRAIN)]))
    
    logs.write(f"Exp-train rnd, " + ",".join(map(str,exps_train_rnd)) + "\n")
    logs.write(f"Exp-test rnd, " + ",".join(map(str,exps_test_rnd)) + "\n")
    logs.write(f"DNI-train rnd, " + ",".join(map(str,dnis_train_rnd)) + "\n")
    logs.write(f"DNI-test rnd, " + ",".join(map(str,dnis_test_rnd)) + "\n")
    logs.write(f"Exp-train grd, " + ",".join(map(str,exps_train_grd)) + "\n")
    logs.write(f"Exp-test grd, " + ",".join(map(str,exps_test_grd)) + "\n")
    logs.write(f"DNI-train grd, " + ",".join(map(str,dnis_train_grd)) + "\n")
    logs.write(f"DNI-test grd, " + ",".join(map(str,dnis_test_grd)) + "\n")
    logs.write(f"Exp-train deg, " + ",".join(map(str,exps_train_deg)) + "\n")
    logs.write(f"Exp-test deg, " + ",".join(map(str,exps_test_deg)) + "\n")
    logs.write(f"DNI-train deg, " + ",".join(map(str,dnis_train_deg)) + "\n")
    logs.write(f"DNI-test deg, " + ",".join(map(str,dnis_test_deg)) + "\n")
    
# ### Decision focused
def train_df() : 

    logs_df = open(output_dir + 'df_training.txt', 'a')
    logs_df.write(f"{n} - {date} \n")
    start = datetime.datetime.now()

    # Model definition
    net_df = make_fc(N_FEATURES, num_layers, activation, hidden_sizes, dropout, device)
    net_df.apply(init_weights_df)
    net_df = net_df.to(device)
    # net_df.modules

    # Training
    print(f"### DECISION-FOCUSED MODEL -- {n} ###")
    print("epoch | loss | train_score | test_score | train_dni | test_dni | avg dp | lr")

    optimizer = torch.optim.Adam(net_df.parameters(), lr = learning_rate, betas = (momentum, 0.999))
    marginal_vec_pred = partial(marginal_vec, w = w, device=device)

    #LR scheduler -- regularization
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=num_epochs//10 + 1, T_mult=2, eta_min=0.0001, last_epoch=-1)

    for epoch in range(num_epochs):
        data_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size)
        for X_batch, P_batch in data_loader:
            loss = 0 
            for X, P in zip(X_batch, P_batch):
                Xgpu = X.to(device)
                Pgpu = P.to(device)
                true_set_func = partial(set_func, P = Pgpu, w = w, device=device)
                pred = net_df(Xgpu).view_as(P)
                fn = GreedyOptimizer.apply
                loss += -fn(pred, true_set_func, marginal_vec_pred, N_INFLUENCERS,  k,  eps, sample_size, beta, device) + reg_avg(pred)
                
            loss = loss / batch_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        lr_sched.step()

        with torch.no_grad():
            train_score = eval_spread(net_df, X_train, Y_train)
            test_score  = eval_spread(net_df, X_test, Y_test)
            
            if labels=="db" : 
                train_dni = eval_dni(net_df, X_train, Y_train)
                test_dni  = eval_dni(net_df, X_test, Y_test)
            else : 
                train_dni = eval_dni(net_df, X_train, Y2_train)
                test_dni  = eval_dni(net_df, X_test, Y2_test)
            
            avg_dp = torch.mean(net_df(X_train[0].to(device)))
            
        s = f"{epoch} | {loss.item()} | {train_score} | {test_score} | {train_dni} | {test_dni} | {avg_dp} | {optimizer.param_groups[0]['lr']}"
        print(s)
        logs_df.write(s + '\n')
    logs_df.write('Runtime : ' + str(datetime.datetime.now() - start) + '\n \n')
    logs_df.close()

    return net_df

# ### 2 Stage
# Model definiton
def train_2s() : 

    logs_2s = open(output_dir + '2s_training.txt', 'a')
    logs_2s.write(str(date) + '\n')
    start = datetime.datetime.now()
    net_2s = make_fc(N_FEATURES, num_layers, activation, hidden_sizes, dropout, device)
    net_2s.apply(init_weights_2s)
    net_2s = net_2s.to(device)
    print(net_2s.modules)

    loss_fn = nn.MSELoss() 
    
    optimizer = torch.optim.Adam(net_2s.parameters(), lr = learning_rate_2s, betas = (momentum, 0.999))
    
    # Training
    print(f"### 2 STAGE MODEL -- {n} ###")
    print("epoch | loss | train_score | test_score | train_dni | test_dni | avg dp | lr")

    for epoch in range(num_epochs):
        data_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, ) #shuffle=True 
        for X_batch, P_batch in data_loader:
            loss = 0 
            for X, P in zip(X_batch, P_batch):
                X, P = X.to(device), P.to(device)
                pred = net_2s(X).view_as(P)
                loss += loss_unbalanced(pred, P) + reg_avg(pred)
#                loss += loss_fn(pred, P)
            loss = loss / batch_size
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(net_2s.parameters(), 1.)
            optimizer.step()
        
        with torch.no_grad():
            train_score = eval_spread(net_2s, X_train, Y_train)
            test_score  = eval_spread(net_2s, X_test, Y_test)
            
            if labels=="db" : 
                train_dni = eval_dni(net_2s, X_train, Y_train)
                test_dni  = eval_dni(net_2s, X_test, Y_test)
            else : 
                train_dni = eval_dni(net_2s, X_train, Y2_train)
                test_dni  = eval_dni(net_2s, X_test, Y2_test)
            avg_dp = torch.mean(net_2s(X_train[0].to(device)))

        s = f"{epoch} | {loss} | {train_score} | {test_score} | {train_dni} | {test_dni} | {avg_dp} | {optimizer.param_groups[0]['lr']}" 
        print(s)   
        logs_2s.write(s + '\n')

    logs_2s.write(f"Runtime : {datetime.datetime.now() - start}\n \n")
    logs_2s.close()
    return net_2s

parser = argparse.ArgumentParser()

parser.add_argument('--num-epochs', type=int, default=20,           help='')
parser.add_argument('--batch-size', type=int, default=1,            help='')
parser.add_argument('--learning-rate', type=float, default=1e-3,    help='')
parser.add_argument('--learning-rate-2s', type=float, default=1e-2, help='')
parser.add_argument('--net-df-path', type=str, default="net_df",    help='')
parser.add_argument('--net-2s-path', type=str, default="net_2s",    help='')
parser.add_argument('--labels', type=str, default="db",             help='db or infector or inf2vec')
parser.add_argument('--n-iter', type=int, default=1,                help='number of created models')
parser.add_argument('--output-dir', type=str, default="results/",   help='path where models are saved')
parser.add_argument('--reg-coeff', type=float, default=0.01,        help='regularization coefficient')
parser.add_argument('--seed', type=int, default=0,                  help='seed')
parser.add_argument('--device', type=str, default='cuda:0',         help='device')

if __name__ == '__main__':

    args=parser.parse_args()

    #--- Parameters
    num_epochs = int(args.num_epochs)         
    batch_size = int(args.batch_size)
    learning_rate = float(args.learning_rate)
    learning_rate_2s = float(args.learning_rate_2s)
    net_df_path = args.net_df_path
    net_2s_path = args.net_2s_path
    labels = args.labels
    n_iter = int(args.n_iter)
    output_dir = args.output_dir
    reg_coeff = float(args.reg_coeff)
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    torch.cuda.empty_cache()
    if device.type == 'cuda': print(torch.cuda.get_device_name(0))

    if not os.path.exists(output_dir) : os.mkdir(output_dir)
    
    print(f"Dataset : {instance_path}")
    logs = open(output_dir + '/perfs_train_test.txt', 'a')
    logs.write(f"{datetime.datetime.now()} - Dataset : {instance_path} - labels : {labels} - numepochs : {num_epochs} - batchsize : {batch_size} - lr : {learning_rate} - regcoeff : {reg_coeff} \n")
    Ks = [int(a * N_INFLUENCERS) for a in [0.01,0.02,0.05,0.1,0.2]]
    logs.write(f"Ks : {Ks} \n")

    w = np.ones(N_TARGETS, dtype=np.float32)
    for n in range(n_iter) : 
    
        train_id = random.sample(list(range(N_INSTANCES)), N_TRAIN)
        if labels == "db" : X_train, Y_train, X_test, Y_test, _,_ = create_train_test(train_id)
        else : X_train, Y2_train, X_test, Y2_test, Y_train, Y_test = create_train_test(train_id)
        dataset = torch.utils.data.TensorDataset(X_train, Y_train) 
        unbalance = torch.sum(Y_train[Y_train>0]) / torch.sum(torch.ones_like(Y_train))
        print(f"Unbalance : {unbalance}")

        date = datetime.datetime.now()
        baselines()
        net_df = train_df()
        net_2s = train_2s()

        with torch.no_grad():
            X_train, Y_train, X_test, Y_test = X_train.to(device), Y_train.to(device), X_test.to(device), Y_test.to(device)
            results_model(logs, net_df, "df")
            results_model(logs, net_2s, "2s")
        
        results_rdn_grd_deg(logs)

        torch.save(net_df, output_dir + f"{net_df_path}_{num_epochs}_{batch_size}_{labels}_{n}.pt")
        torch.save(net_2s, output_dir + f"{net_2s_path}_{num_epochs}_{batch_size}_{labels}_{n}.pt")

    logs.close()