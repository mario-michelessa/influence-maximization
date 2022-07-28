
import networkx as nx
import numpy as np
import pandas as pd
from collections import defaultdict
import os 
import argparse

import warnings
warnings.filterwarnings('ignore')

# graph_twitter_full.csv : 
#     - 62293412 lines
#     - 12,15
#     - 12,17
#     - 12,18

# data/twitter/retweetsranked_full.csv
#     - 2332764 lines
#     - 3059561715;2009-08-01 00:25:33;21172954;21926959;3059313429
#     - 3096677654;2009-08-03 04:20:16;16103374;7516242;3090911707

def subsampling_1(n_cascades, n_influences_max):
    """
    Only extracts the n_cascades first cascades and only the n_influences_max first reposts of each cascade    
    """
    n_cascades_per_u = df_retweets.groupby('u').agg({'mid' : 'count'})
    # n_cascades_per_v = df_retweets.groupby('v').agg({'mid' : 'count'})

    influencers = list(n_cascades_per_u.sample(n_cascades).index)

    df_labels = df_retweets[df_retweets['u'].isin(influencers)][['u','v']]
    df_labels = df_labels.groupby('u').head(n_influences_max)[['u','v']]
    
    Au = df_labels.groupby('u').count().v
    Au.name = 'Au'
    Av = df_labels.groupby('v').count().u
    Av.name = 'Av'

    print(f"i = {len(Au.index)}, t = {len(Av.index)}")
    
    df_labels = df_labels.merge(Au, on='u')
    df_labels = df_labels.merge(Av, on='v')

    Au2v = df_labels.groupby(['u', 'v']).count().Au.reset_index()
    df_labels = df_labels.merge(Au2v, on=['u','v'])

    df_labels.columns = ['u', 'v', 'Au', 'Av', 'Au2v']
    df_labels = df_labels.drop_duplicates()
    df_labels['u'] = df_labels['u'].astype(np.int64)
    df_labels['v'] = df_labels['v'].astype(np.int64)
    print(f"Unbalance =  { df_labels.shape[0] / len(Au.index) / len(Av.index)}")
    return df_labels

def subsampling_2(n_min_cascade_per_u, n_min_cascade_per_v) :
    """
    selects the row of df_retweets where u has more than u_min_cascade_per_u cascades and same with v
    """
    n_cascades_per_u = df_retweets.groupby('u').agg({'mid' : 'count'})
    n_cascades_per_v = df_retweets.groupby('v').agg({'mid' : 'count'})

    influencers = list(n_cascades_per_u[n_cascades_per_u['mid'] > n_min_cascade_per_u].index)
    targets = list(n_cascades_per_v[n_cascades_per_v['mid'] > n_min_cascade_per_v].index)

    print(f"i = {len(influencers)}, t = {len(targets)}")

    df_labels = df_retweets[df_retweets['u'].isin(influencers)][['u','v']]
    df_labels = df_labels[df_labels['v'].isin(targets)][['u','v']]
    Au = df_labels.groupby('u').count().v
    Au.name = 'Au'
    Av = df_labels.groupby('v').count().u
    Av.name = 'Av'
    

    df_labels = df_labels.merge(Au, on='u')
    df_labels = df_labels.merge(Av, on='v')

    Au2v = df_labels.groupby(['u', 'v']).count().Au.reset_index()
    df_labels = df_labels.merge(Au2v, on=['u','v'])

    df_labels.columns = ['u', 'v', 'Au', 'Av', 'Au2v']
    df_labels = df_labels.drop_duplicates()
    df_labels['u'] = df_labels['u'].astype(np.int64)
    df_labels['v'] = df_labels['v'].astype(np.int64)
    print(f"Unbalance =  { df_labels.shape[0] / len(influencers) / len(targets)}")
    return df_labels

def estimate_probabilities(df_labels, save=False, file_name="") : 
    """
    input : edges dataframe 
    """
    df_labels['BT'] = df_labels['Au2v'] / df_labels['Au']
    df_labels['JI'] = df_labels['Au2v'] / (df_labels['Au'] + df_labels['Av'])
    df_labels['LP'] = df_labels['Au2v'] / df_labels['Av']

    df_labels = df_labels[['u', 'v', 'BT', 'JI', 'LP']]
    df_labels = df_labels.drop_duplicates()
    print("db labels : \n" + df_labels.head(5).to_markdown())
    print(f"shape : {df_labels.shape}\n")

    if save :
        print("Saved " + file_name)
        pd.to_pickle(df_labels, output_folder+file_name)
    
    return df_labels

def create_edges(users_topology, save=False, file_name = "") : 
    """
    Creates weibo/edges_NB_CASCADES.pkl ---> |id|u|v|

    2.4M edges for 1000 cascades
    Takes 4min to extract
    """
    #Speeds up operation (v in users)
    d_users = defaultdict(lambda : False)
    for user in users_topology:
        d_users[int(user)] = True
    
    edges = []                 

    with open("data/twitter/graph_twitter_full.csv", 'r') as file :
        
        u_previous = -1
        u_in_table = False

        n_lines = 0
        
        for line in file :
            
            n_lines += 1
            if n_lines % 17000000 == 0 :
                print(f"{n_lines//1000000}M lines processed : {len(edges)} edges added")

            line = line.split(',')
            if len(line) >= 2 :
                u,v = int(line[0]), int(line[1])
                if u != u_previous : 
                    u_previous = u
                    u_in_table = (d_users[u])

                if u_in_table: 
                    if d_users[v] : 
                        edges.append((u,v))
    
    df_edges = pd.DataFrame(edges, columns=['u','v'])

    print("df_edges : \n" + df_edges.head(5).to_markdown())
    print(f"shape : {df_edges.shape}\n")
    if save : 
        df_edges.to_pickle(output_folder + file_name)
        print("Saved " + file_name)
    return df_edges

def create_df_features(df_edges, save=False) : 

    deg_out = df_edges.groupby('u').count()
    d_deg_out = {u:0 for u in influencers} # some influencers will have 0 degrees
    for u in deg_out.index : d_deg_out[u] = deg_out.loc[u].v

    deg_in = df_edges.groupby('v').count()
    d_deg_in= {v : 0 for v in targets}
    for v in deg_in.index : d_deg_in[v] = deg_in.loc[v].u

    print("Calculating PageRank")
    g = nx.DiGraph()
    g.add_nodes_from(users)
    g.add_edges_from(zip(df_edges.u, df_edges.v))

    pagerank = nx.pagerank(g)
    pagerank = pd.DataFrame.from_dict(pagerank, orient='index', columns= ['pagerank'])

    #calculating #reposts and #cascades
    reposts = df_retweets.groupby('u').count().mid
    cascades = df_retweets.groupby('mid').first().groupby('u').count().v

    
    d_fu = {u:[d_deg_out[u], float(pagerank.loc[u]), 0, reposts.loc[u], cascades.loc[u]] for u in influencers}
    features_influencers = pd.DataFrame.from_dict(d_fu, orient='index', columns=['deg_out', 'pagerank', 'likes', 'reposts', 'cascades'])
    d_fv = {v:[d_deg_in[v], float(pagerank.loc[v])] for v in targets}
    features_targets = pd.DataFrame.from_dict(d_fv, orient='index', columns=['deg_in', 'pagerank'])

    features_influencers['deg_out'] = features_influencers['deg_out'].apply(lambda x : np.log(1 + x) / 8)
    features_influencers['pagerank'] = features_influencers['pagerank'] / 0.00011
    features_influencers['reposts'] = features_influencers['reposts'].apply(lambda x : np.log(1 + x) / 2)
    features_influencers['cascades'] = features_influencers['cascades'].apply(lambda x : np.log(1 + x) / 2)

    features_targets['deg_in'] = features_targets['deg_in'].apply(lambda x : np.log(1 + x) / 7)
    features_targets['pagerank'] = features_targets['pagerank'] / 0.00011  

    if save : 
        features_influencers.to_pickle(output_folder + f'features_influencers{subsampling}_{n1}_{n2}.pkl')
        features_targets.to_pickle(output_folder + f'features_targets{subsampling}_{n1}_{n2}.pkl')
    return features_influencers, features_targets

def fill_y(u,v) : 
    if d_labels[(u,v)] : 
        return df_labels.loc[(u,v)][PROB_TYPE]
    else : 
        return 0

def feature_twitter(u,v, fu=None, fv=None ) :
    """
    i:followers_count, 'i:friends_count', 'i:statuses_count', 'i:verified', 'i:gender', 
    'i:d_out', 'i:pagerank',
    'i: #likes', 'i:#reposts', 'i:#cascades', 
    'i:high_topic', 'i:med_topic', 'i:low_topic', 
    't:followers_count', 't:friends_count', 't:statuses_count', 't:verified', 't:gender', 
    't:d_out', 't:pagerank', 
    't:high_topic', 't:med_topic', 't:low_topic', 
    'edge'
    """
    if fu is not None : 
        return np.concatenate((np.zeros(5),fu,np.zeros(8), fv,np.zeros(3), d_edges[(u,v)]), axis=None)

    return np.concatenate([np.zeros(5), #user profile features 
                            features_influencers.loc[u],
                            np.zeros(8), #topic features u + user profile features v
                            features_targets.loc[v],
                            np.zeros(3), #topic features v 
                            d_edges[(u,v)]], axis = None)

def create_XY(sampled_influencers, sampled_targets) :
    """
    from 2 sets of influencers and targets, creates features and labels according to the paper format
    """
    nI = len(sampled_influencers)
    nT = len(sampled_targets)

    X = np.zeros((nI, nT, N_FEATURES))
    Y = np.zeros((nI, nT))

    #To not call loc for each (u,v)
    fI = np.array(features_influencers.loc[sampled_influencers])
    fT = np.array(features_targets.loc[sampled_targets])
    # eI = np.array(influencers_embeddings.loc[sampled_influencers])
    # eT = np.array(targets_embeddings.loc[sampled_targets])

    for i in range(nI):
        for j in range(nT):
            u,v = sampled_influencers[i], sampled_targets[j]
            X[i,j, :] = feature_twitter(u, v, fI[i], fT[j])

            #X[i,j, :] = np.concatenate([features_influencers.loc[sampled_influencers[i]], 
                                        # features_targets.loc[sampled_targets[j]]], 
                                        # axis = None)
            Y[i,j] = fill_y(u,v)
            
    Y = np.reshape(Y, (nI, nT,1))    

    return np.concatenate((X, Y), axis = 2)


output_folder = "data/twitter_preprocessed/"

parser = argparse.ArgumentParser()
parser.add_argument('--subsampling', type=int, default=1)
parser.add_argument('--n1', type=int, default=1)
parser.add_argument('--n2', type=int, default=1)
parser.add_argument('--save', type=bool, default=False)
parser.add_argument('--instances-path', type=str, default="data/twitter_preprocessed/")

if __name__ == "__main__":
    args = parser.parse_args()
    subsampling = int(args.subsampling)
    n1 = int(args.n1)
    n2 = int(args.n2)
    save = bool(args.save) 
    instances_path = args.instances_path

    df_retweets = pd.read_csv("data/twitter/retweetsranked_full.csv", sep = ';', header=None, usecols=[2,3,4], names=['v','u','mid'])
    df_retweets = df_retweets.dropna().astype({'u': np.int64, 'v': np.int64})
    print("df_retweets : \n" + df_retweets.head(5).to_markdown())
    print(f"shape : {df_retweets.shape}\n")

    if subsampling == 1 : df_labels = subsampling_1(n1,n2)
    else : df_labels = subsampling_2(n1,n2)

    df_labels = estimate_probabilities(df_labels, True, f"labels{subsampling}_{n1}_{n2}.pkl")
    df_labels.index = pd.MultiIndex.from_tuples(zip(df_labels['u'],df_labels['v'])) #important to do .loc[(u,v)]
    d_labels = defaultdict(lambda : False)
    for (u,v) in zip(df_labels.u, df_labels.v) : d_labels[(u,v)] = True

    influencers = list(df_labels.groupby('u').count().index)
    targets = list(df_labels.groupby('v').count().index)
    print(len(influencers), len(targets))

    users = list(set(influencers).union(targets))
    # df_edges =create_edges(users, True, f"edges{subsampling}_{n1}_{n2}.pkl")
    df_edges = pd.read_pickle(f"data/twitter_preprocessed/edges{subsampling}_{n1}_{n2}.pkl")
    d_edges = defaultdict(lambda : 0)
    for (u,v) in zip(df_edges.u, df_edges.v) : d_edges[(u,v)] = 1

    features_influencers, features_targets = create_df_features(df_edges, save = True)
    print(features_influencers.mean())
    print(features_targets.mean())
    if not os.path.exists(instances_path) : 
        os.makedirs(instances_path)

    N_FEATURES = 24
    PROB_TYPE = 'JI'
    N_INSTANCES = 20
    N_INFLUENCERS = 1000
    N_TARGETS = 1000

    for instance in range(N_INSTANCES) : 

        if os.path.exists(instances_path + f'{instance}.npz') :
            print("Instance already created")
        else :
            if instance % (N_INSTANCES // 5) == 0 : print(f"Creating instance {instance}/{N_INSTANCES}")
            sampled_influencers = np.random.choice(influencers, N_INFLUENCERS, p = None, replace=False)
            sampled_targets = np.random.choice(targets, N_TARGETS, p = None, replace=False)

            XY = create_XY(sampled_influencers, sampled_targets)
            np.savez(instances_path + f'{instance}.npz', XY)    
            del(XY)
        
    print("End")

# Cascades analysis : 
# 
#     - N_MIN_REAC | #cascades
#     - 1          | 1.947M
#     - 2          | 158 986
#     - 3          | 57 216
#     - 4          | 32 091
#     - 5          | 21 337
#     - 10         | 6 035
#     - 15         | 2 913
#     - 20         | 1 681
