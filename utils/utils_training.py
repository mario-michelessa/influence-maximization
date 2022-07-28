import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import torch
import random
from greedy_coverage_gpu import greedy, set_func

def print_training(loss, train, test, train_dni, test_dni, title = "") : 

    fig, ax = plt.subplots(1,3, figsize = (15,5))
    plt.suptitle(title)
    ax[0].plot(loss)
    ax[0].set_title('Learning curve')
    ax[0].set_ylabel('loss')
    ax[0].set_xlabel('epoch')

    ax[1].plot(train, label="train_scores")
    ax[1].plot(test, label="test_scores")
    ax[1].set_title('Quality')
    ax[1].set_ylabel('expected #influenced')
    ax[1].set_xlabel('epoch')
    ax[1].legend()

    ax[2].plot(train_dni, label="train_scores")
    ax[2].plot(test_dni, label="test_scores")
    ax[2].set_title('data-based quality')
    ax[2].set_ylabel('different nodes influenced')
    ax[2].set_xlabel('epoch')
    ax[2].legend()
    plt.show()

def dni(seeds, Y) : 
    """
    estimates the number of Different Nodes Infected given a seed set and a DATA-BASED label. 
        If a data-based label of (u,v) is positive, it means that v appears in at least one cascade of u
    for each column(target) if one of the labels is >0 then it appears in a cascade provoqued by one of the seeds
    """
    return (Y[seeds,:].sum(dim=0) > 0).sum().item() 

def eval_dni(net, X, Y, k, device="cuda"):
    """
    input : X[instances, nI, nT, nF], Y[instances, nI, nT]
    estimates number of different nodes infected based on the cascades
    """
    Xgpu = X.to(device)
    Ygpu = Y.to(device)
    result = np.mean([dni(greedy(k, net(Xgpu[i,:,:,:]).view_as(Ygpu[i]), w = torch.Tensor.ones(Xgpu.shape[1]))[1]    ,Ygpu[i]) for i in range(X.shape[0])])
    del Xgpu, Ygpu
    return float(result) #float is here to delete the gradient if there is a torch gradient

def eval_grd(net, X, Y, k, device="cuda"):
    """
    estimates expectation of dni in the bipartite graph: sum(1-prod(1-p_uv))
    """
    Xgpu = X.to(device)
    Ygpu = Y.to(device)
    w = torch.Tensor.ones(Xgpu.shape[1])
    # print(np.mean([greedy(k, net(X[i,:,:,:]).view_as(Y_train[0]), w)[1].__len__() for i in range(X.shape[0])]))
    result = np.mean([set_func(greedy(k, net(Xgpu[i]).view_as(Ygpu[i]), w)[1], Ygpu[i], w) for i in range(X.shape[0])])
    del Xgpu, Ygpu
    return float(result)

def eval_loss(net, X, Y, device="cuda"):
    """
    validation loss
    """
    Xgpu = X.to(device)
    Ygpu = Y.to(device)
    result = np.mean([np.sum((net(Xgpu[i]).view_as(Y[0]) - Ygpu[i] ).cpu().detach().numpy() **2.) for i in range(X.shape[0])])
    del Xgpu, Ygpu
    return float(result)

def eval_rnd(X, Y, k, device="cuda"):
    """
    Randomly selects seeds and computes influence
    """
    nI = X.shape[1]
    sol = random.sample(range(nI),k)
    Ygpu = Y.to(device)
    result = np.mean([set_func(sol, Ygpu[i,:,:], w = torch.Tensor.ones(nI)) for i in range(X.shape[0])])
    del Ygpu
    return result


def print_hidden_layers(net, title, num_layers) : 
    fig,ax = plt.subplots(1, num_layers, figsize = (20,5))
    plt.suptitle(title)

    for i,ax in enumerate(ax.flat) :
        weights = net[3*i].weight.detach().cpu().numpy()
        sns.heatmap(weights, ax=ax)
        ax.set_title(f'Layer {i + 1} : {weights.shape[0]}x{weights.shape[1]}')

def print_output_test(net, title, X_test, Y_test, device="cuda") :
    fig,ax = plt.subplots(2, 2, figsize = (15,10),sharex=True, sharey=True)
    # cbar_ax = fig.add_axes([.91, .3, .01, .4])
    # netcpu = copy.deepcopy(net).to('cpu')
    plt.suptitle(title)
    for i,ax in enumerate(ax.flat) :
        if i < 2 : 
            sns.heatmap(net(X_test[i].to(device)).view_as(Y_test[0]).detach().cpu(), 
                        ax=ax, 
                        # vmax = 1, 
                        # vmin = 0, 
                        # cbar_ax = None if i else cbar_ax,
                        # cbar=(i==0)
            )
            ax.set_title(f'm(X_test[{i}])')
        else : 
            sns.heatmap(Y_test[i-2], 
                        ax=ax, 
                        # vmax = 1, 
                        # vmin = 0, 
                        # cbar_ax = None if i else cbar_ax,
                        # cbar=(i==0)
            )
            ax.set_title(f'Y_test[{i-2}]')


def compare_score_model(train_scores_df, test_scores_df, train_scores_2s, test_scores_2s, greedy_train, rd_score_xtrain, rd_score_xtest, greedy_test, title=""):
    fig,ax = plt.subplots(1,2, figsize = (15,7), sharey=True)
    n = min(len(test_scores_2s),len(test_scores_df))

    ax[0].plot(train_scores_df[:n], label='decision-based')# ax = ax[0])
    ax[0].plot(train_scores_2s[:n], label='2-stage')# ax = ax[0])
    ax[0].hlines(y=greedy_train, xmin=0, xmax=n, linestyle='--', color='green', label='oracle-greedy')
    ax[0].hlines(y=rd_score_xtrain, xmin=0, xmax=n, linestyle='--', color='red', label='average random')
    ax[0].set_title("Influence on train dataset")
    ax[0].set_ylabel('Expected number of targets influenced')
    ax[0].set_xlabel('epochs')
    ax[0].legend(loc='upper right')
    
    
    ax[1].plot(test_scores_df[:n],label='decision-based')
    ax[1].plot(test_scores_2s[:n], label='2-stage')
    ax[1].hlines(y=rd_score_xtest, xmin=0, xmax=n, linestyle='--', color='red', label='average random')
    ax[1].hlines(y=greedy_test, xmin=0, xmax=n, linestyle='--', color='green', label='oracle-greedy')
    ax[1].set_title("Influence on test dataset")
    ax[1].set_xlabel('epochs')
    ax[1].legend(loc='upper right')
    
    fig.suptitle(title)
    plt.show()


def degree_seeds(k, X,) : 
    """
    Returns the k influencers having the biggest d_out
    """
    return X[:,0,5].sort(descending=True).indices[:k]

def plot_kdependence(X_train, Y_train, device, net_df, net_2s, step = 1, title=""):
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    
    l_degrees = []
    l_df = []
    l_2s = []
    l_greedy = []
    l_random = []
    w = torch.Tensor.ones(X_train.shape[1])
    nI = X_train.shape[1]
    for k in range(0,nI,step) : 
        if k % 50 == 0 : print(f"{k}/{X_train.shape[1]}") 
        
        l_df.append(np.mean([set_func(greedy(k, net_df(X_train[i,:,:,:]).view_as(Y_train[0]), w)[1], Y_train[i, :, :], w) for i in range(X_train.shape[0])]))
        l_2s.append(np.mean([set_func(greedy(k, net_2s(X_train[i,:,:,:]).view_as(Y_train[0]), w)[1], Y_train[i, :, :], w) for i in range(X_train.shape[0])]))
        
        l_greedy.append(np.mean([set_func(greedy(k, Y_train[i, :, :], w)[1], Y_train[i, :, :], w) for i in range(Y_train.shape[0])]))
        l_random.append(np.mean([np.mean([set_func( random.sample(range(nI),k) , Y_train[i,:,:], w) for i in range(X_train.shape[0])]) for _ in range(2)]))
        l_degrees.append(np.mean([set_func(degree_seeds(k, X_train[i]), Y_train[i], w) for i in range(X_train.shape[0])]))

    fig,ax = plt.subplots(figsize=(12,8))
    axX = [k*5 for k in range(len(l_2s))]
    ax.plot(axX, l_df, label='decisison-based')# ax = ax[0])
    ax.plot(axX, l_2s, label='2-stage')# ax = ax[0])
    ax.plot(axX, l_random, linestyle='--', color='red', label='random')
    ax.plot(axX, l_greedy, linestyle='--', color='green', label='oracle-greedy')
    ax.plot(axX, l_degrees, linestyle='--', color='blue', label='degree-heuristic')

    ax.legend(loc='lower right')
    ax.set_title("Influence of k on expected number of targets influenced")
    ax.set_ylabel('Average number of targets influenced')
    ax.set_xlabel('k = number of seeds')
    plt.plot()
