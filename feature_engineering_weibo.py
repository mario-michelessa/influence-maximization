import numpy as np
import pandas as pd
import warnings
warnings.simplefilter('ignore')
from collections import defaultdict
import networkx as nx

input_folder = 'data/weibo_preprocessed/'
output_folder = 'data/weibo_features/'

#To edit
labels_path = input_folder + "labels1_5K_10.pkl"
edges_path = input_folder + "edges1_5K_10.pkl"
features_influencers_path = "features_influencers1_5K_10_cas.pkl"
features_targets_path = "features_targets1_5K_10_cas.pkl"

user_profile_path = input_folder + "userProfile.pkl"
infos_influencers_path = input_folder + "infos_influencers_3.pkl"
infos_targets_path = input_folder + "infos_targets_3.pkl"

# Objective of this file 
# 
# Input : 
# 
#     user_profile : contains all profile features
#     labels : contains ground_truth influences of subsampled cascades
#     edges : contains induced graph topology of subsampled cascades
#     infos_influencers : contains topic averages of all influencers
#     infos_targets : contains topic averages of all targets
#     influencers_emb_path : contains IMINFECTOR embeddings of 25K influencers
#     targets_emb_path : contains IMINFECTOR embeddings of 1.1M targets
# 
# Output : 
# 
#     features_influencers : contains all features of the subsampled influencers
#     features_targets : contains all features of the subsampled targets
# 

if __name__ == '__main__':

    ### User profile infos 
    user_profile = pd.read_pickle(user_profile_path)
    features_user_profile = ['followers_count', 'friends_count', 'statuses_count', 'verified', 'gender']

    user_profile.index = user_profile.uid
    user_profile = user_profile[features_user_profile]
    user_profile['followers_count'] = user_profile['followers_count'].apply(lambda x : np.log(max(x, 0) + 1)) / 10
    user_profile['friends_count']   = user_profile['friends_count'].apply(lambda x : np.log(max(x, 0) + 1)) / 8
    user_profile['statuses_count']  = user_profile['statuses_count'].apply(lambda x : np.log(max(x, 0) + 1)) / 10
    user_profile['verified']        = user_profile.verified.cat.codes
    user_profile['gender']          = user_profile.gender.cat.codes

    print("df_user : \n" + user_profile.head(4).to_markdown())
    print(f"shape : {user_profile.shape}\n" )

    # Topology features estimation
    edges = pd.read_pickle(edges_path)

    deg_out = edges.groupby('u').count()
    d_deg_out= defaultdict(lambda : 0)
    for u in deg_out.index :
        d_deg_out[u] = deg_out.loc[u].v

    deg_in = edges.groupby('v').count()
    d_deg_in= defaultdict(lambda : 0)
    for v in deg_in.index : 
        d_deg_in[v] = deg_in.loc[v].u

    g = nx.DiGraph()
    g.add_edges_from(zip(edges.u, edges.v))

    print("Calculating pagerank...")
    pagerank = nx.pagerank(g)
    pagerank = pd.DataFrame.from_dict(pagerank, orient='index', columns=['pagerank'])
    pagerank = pagerank / pagerank.max()
    print("Done")
    # d_2g = defaultdict(lambda : 0)
    # influencers = edges.groupby('u').count().index
    # for i in influencers : 
    #     set_2neighboors = set()
    #     for n in nx.DiGraph.neighbors(g,i) : 
    #         set_2neighboors = set_2neighboors.union(set([n2 for n2 in nx.DiGraph.neighbors(g,n)]))
    #     d_2g[i] = len(set_2neighboors)

    # df_2reachable = pd.DataFrame.from_dict(d_2g, orient='index', columns=['2reachable'])
    # df_2reachable = df_2reachable / 2500

    # Topic informations
    infos_influencers = pd.read_pickle(infos_influencers_path)
    infos_targets = pd.read_pickle(infos_targets_path)

    # infos_influencers = infos_influencers[[0,1,2]] # remove infos coming from cascades
    infos_influencers['total_likes'] = infos_influencers['total_likes'].apply(lambda x : np.log(max(x, 0) + 1)) / 17
    infos_influencers['total_reposts']   = infos_influencers['total_reposts'].apply(lambda x : np.log(max(x, 0) + 1)) / 13
    infos_influencers['n_cascades']  = infos_influencers['n_cascades'].apply(lambda x : np.log(max(x, 0) + 1)) / 8

    print("infos_influencers : \n" + infos_influencers.head(4).to_markdown())
    print(f"shape : {infos_influencers.shape}\n" )

    # Build Influencers/Targets features tables
    labels = pd.read_pickle(labels_path)
    influencers = list(labels.groupby('u').count().index)
    targets = list(labels.groupby('v').count().index)

    print(f"sizes from label : \nI = {len(influencers)}\nT = {len(targets)}")

    #add user profile features
    features_influencers = user_profile.loc[user_profile.index.intersection(influencers)]
    features_influencers = features_influencers.groupby(features_influencers.index).first()
    features_targets = user_profile.loc[user_profile.index.intersection(targets)]
    features_targets = features_targets.groupby(features_targets.index).first()

    #add topology features
    features_influencers['d_out'] = pd.Series(features_influencers.index, index=features_influencers.index).apply(lambda uid : np.log(d_deg_out[uid] + 1) / 8)
    features_influencers = features_influencers.merge(pagerank, left_on=features_influencers.index, right_on=pagerank.index).set_index('key_0')
    features_targets['d_in'] = pd.Series(features_targets.index, index=features_targets.index).apply(lambda uid : np.log(d_deg_in[uid] + 1) / 7)
    features_targets = features_targets.merge(pagerank, left_on=features_targets.index, right_on=pagerank.index).set_index('key_0')

    # features_influencers = features_influencers.merge(df_2reachable, left_on=features_influencers.index, right_on=df_2reachable.index).set_index('key_0')

    #add topic features
    features_influencers = features_influencers.merge(infos_influencers, left_on=features_influencers.index, right_on=infos_influencers.index).set_index('key_0')
    features_targets = features_targets.merge(infos_targets, left_on=features_targets.index, right_on=infos_targets.index).set_index('key_0')

    print(f"sizes after feature extraction :\nI = {features_influencers.shape[0]}\nT = {features_targets.shape[0]}")

    # Saving tables
    features_influencers.to_pickle(output_folder + features_influencers_path)
    features_targets.to_pickle(output_folder + features_targets_path)


