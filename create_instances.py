import os
import numpy as np
import pandas as pd
from numba import jit

import os 
import warnings
warnings.simplefilter('ignore')

from collections import defaultdict

# features_influencers_path = "data/weibo_features/features_influencers2_-1_150_cas.pkl"
# features_targets_path = "data/weibo_features/features_targets2_-1_150_cas.pkl"
# labels_path = "data/weibo_preprocessed/labels2_-1_150.pkl"
# edges_path = "data/weibo_preprocessed/edges2_-1_150.pkl"

features_influencers_path = "data/weibo_features/features_influencers1_5K_10_cas.pkl"
features_targets_path = "data/weibo_features/features_targets1_5K_10_cas.pkl"
labels_path = "data/weibo_preprocessed/labels1_5K_10.pkl"
edges_path = "data/weibo_preprocessed/edges1_5K_10.pkl"

influencers_embeddings_path = "data/weibo_preprocessed/influencers_embeddings.pkl"
targets_embeddings_path = "data/weibo_preprocessed/target_embeddings.pkl"
influencers_inf2vec_path = "data/weibo_preprocessed/influencers_inf2vec.pkl"
targets_inf2vec_path = "data/weibo_preprocessed/target_inf2vec.pkl"

PROB_TYPE = 'JI'

path = 'decision_focused_learning_gpu/instances_weibo/07-17-sparse_cas/'
N_INSTANCES = 20
N_INFLUENCERS = 1000
N_TARGETS = 1000
PROP_I = 1.

# import of the data
features_influencers = pd.read_pickle(features_influencers_path)
features_targets = pd.read_pickle(features_targets_path)
print("features_influencers : \n" + str(features_influencers.dtypes))
print("shape : " + str(features_influencers.shape))
print("features_targets : \n" + str(features_targets.dtypes))
print("shape : " + str(features_targets.shape))

labels = pd.read_pickle(labels_path)
labels.index = pd.MultiIndex.from_tuples(zip(labels['u'],labels['v'])) #important to do .loc[(u,v)]
labels = labels.sort_index() # infos are retreived faster
labels = labels.drop_duplicates()
print("labels : \n" + labels.head(2).to_markdown())
print("shape : " + str(labels.shape))

influencers_infector = pd.read_pickle(influencers_embeddings_path)
targets_infector = pd.read_pickle(targets_embeddings_path)

influencers_inf2vec = pd.read_pickle(influencers_inf2vec_path)
targets_inf2vec = pd.read_pickle(targets_inf2vec_path)

# Preprocessing labels
# removing the labels where we do not have the embeddings
# and removing the labels where we do not have the features
d_influencers = defaultdict(lambda : 0)
for i in list(influencers_infector.index) : d_influencers[i] += 1
for i in list(influencers_inf2vec.index) : d_influencers[i] += 1
for i in list(features_influencers.index) : d_influencers[i] += 1

d_targets = defaultdict(lambda : 0)
for i in list(targets_infector.index) : d_targets[i] += 1
for i in list(targets_inf2vec.index) : d_targets[i] += 1
for i in list(features_targets.index) : d_targets[i] += 1

labels = labels.drop(labels[labels.u.apply(lambda x : d_influencers[x] < 3)].index)
labels = labels.drop(labels[labels.v.apply(lambda x : d_targets[x] < 3)].index)

influencers = labels.groupby('u').count().index
targets = labels.groupby('v').count().index

print("labels : \n" + labels.head(2).to_markdown())
print("shape : " + str(labels.shape))
print(f'influencers : {len(influencers)}')
print(f'targets : {len(targets)}')

# Create feature vector
edges = pd.read_pickle(edges_path)
d_edges = defaultdict(lambda : 0)
for (u,v) in zip(edges.u, edges.v) :
    d_edges[(u,v)] = 1
del(edges)

def feature_vector(u,v, fu=None, fv=None) : 
    """
    Creates vector with
    - Influencers features
    - Target features
    - Topology link
    """
    if fu is None or fv is None : 
        fu = features_influencers.loc[u]
        fv = features_targets.loc[v]
        
    return np.concatenate([fu, fv, d_edges[(u,v)]], axis = None)

# Create instance
N_FEATURES = features_influencers.shape[1] + features_targets.shape[1] + 1

d_labels = defaultdict(lambda : False)
for (u,v) in zip(labels.u, labels.v) :
    d_labels[(u,v)] = True

@jit
def softmax(x):
        return np.exp(x)/np.sum(np.exp(x))

def transform_Yemb(Y) : 
    Yemb = np.apply_along_axis(lambda x:x-abs(max(x)), 1, Y) 
    Yemb = np.apply_along_axis(softmax, 1, Yemb)
    Yemb = np.around(Yemb,3)
    Yemb = np.abs(Yemb)/np.max(Yemb)
    return Yemb

def create_XY(sampled_influencers, sampled_targets) :
    nI = len(sampled_influencers)
    nT = len(sampled_targets)

    X = np.zeros((nI, nT, N_FEATURES))
    Y = np.zeros((nI, nT))
    Y_infector = np.zeros((nI, nT))
    Y_inf2vec = np.zeros((nI, nT))

    #To not call loc for each (u,v)
    fI = np.array(features_influencers.loc[sampled_influencers])
    fT = np.array(features_targets.loc[sampled_targets])
    eI = np.array(influencers_infector.loc[sampled_influencers])
    eT = np.array(targets_infector.loc[sampled_targets])
    iI = np.array(influencers_inf2vec.loc[sampled_influencers])
    iT = np.array(targets_inf2vec.loc[sampled_targets])

    for i in range(nI):
        for j in range(nT):
            u,v = sampled_influencers[i], sampled_targets[j]
            X[i,j, :] = feature_vector(u, v, fI[i], fT[j])

            #X[i,j, :] = np.concatenate([features_influencers.loc[sampled_influencers[i]], 
                                        # features_targets.loc[sampled_targets[j]]], 
                                        # axis = None)
            Y[i,j] = labels.loc[(u,v)][PROB_TYPE] if d_labels[(u,v)] else 0 
            Y_infector[i,j] = np.dot(eI[i], eT[j])
            Y_inf2vec[i,j] = np.dot(iI[i], iT[j])
        
    Y = np.reshape(Y, (nI, nT,1))
    Y_infector = transform_Yemb(np.reshape(Y_infector, (nI, nT, 1)))
    Y_inf2vec = transform_Yemb(np.reshape(Y_inf2vec, (nI, nT, 1)))

    return np.concatenate((X, Y, Y_infector, Y_inf2vec), axis = 2)

# Only considering best P% influencers

influencers = labels.groupby('u').count().sort_values('v', ascending=False)
n = int(influencers.shape[0] * PROP_I)
influencers = influencers.iloc[:n].index

# Generate all instances
if not os.path.exists(path) :
    os.mkdir(path)

for instance in range(N_INSTANCES) : 

    # if instance % (N_INSTANCES // 10) == 0 : print(f"Saving instance {instance}/{N_INSTANCES}...")

    if os.path.exists(path + f'{instance}.npz') :
        print("Instance already created")
    else :
        sampled_influencers = np.random.choice(influencers, N_INFLUENCERS, p = None, replace=False)
        sampled_targets = np.random.choice(targets, N_TARGETS, p = None, replace=False)
        XY = create_XY(sampled_influencers, sampled_targets)
        np.savez(path + f'{instance}.npz', XY)    
        del(XY)
    
print("End")


