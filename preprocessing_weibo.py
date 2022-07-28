import os 
import numpy as np
import pandas as pd
from collections import defaultdict
import re 
import scipy.cluster.hierarchy as spc
import warnings
import argparse
warnings.filterwarnings('ignore')

output_folder = "data/weibo_preprocessed/"

u_to_userids = pd.read_csv("data/weibo/weibodata/diffusion/uidlist.txt",header=None, names=['userid'])
u_to_userids.index.name = 'uid'

cascade_to_mids = pd.read_csv("data/weibo/weibodata/diffusion/repost_idlist.txt", header=None, names=['mid'])
cascade_to_mids.index.name = 'cascade'
cascade_to_mids.mid = cascade_to_mids.mid.astype(np.int64)

def convert_userProfile(save) : 
    """
    creates the dataframe of user data and saves it in userProfile.pkl
    """
    
    df = [[] for _ in range(15)]
    with open("data/weibo/weibodata/userProfile/user_profile1.txt", mode='r', encoding='gbk') as file : 
        c = 0
        for line in file : 
            if c >= 15 : #not adding the first 15 lines : headers
                df[c%15].append(line.replace('\n',''))
            c += 1
    
    with open("data/weibo/weibodata/userProfile/user_profile2.txt", mode='r', encoding='gbk') as file : 
        c = 0
        for line in file : 
            if c >= 15 :
                df[c%15].append(line.replace('\n',''))
            c += 1
    d = {}
    for i in range(14) :
        d[i] = df[i]
    df = pd.DataFrame.from_dict(d, orient='columns')
    df.columns = ['uid', 'bi_followers_count', 'city', 
                 'verified', 'followers_count', 'location', 
                 'province', 'friends_count', 'name', 
                 'gender', 'created_at', 'verified_type', 
                 'statuses_count', 'description']
    df = df.astype({    'uid':np.int64, 
                        'bi_followers_count':np.int32, 
                        'city':'category', 
                        'verified':'category', 
                        'followers_count':np.int32, 
                        'province':'category', 
                        'friends_count':np.int32, 
                        'gender':'category', 
                        'verified_type':'category', 
                        'statuses_count': np.int32})
    print(f"\nuser_profile : \n{df.dtypes}")
    print(f"shape : {df.shape}\n")
    
    if save : 
        df.to_pickle(output_folder + "userProfile.pkl")
    return df
    
def extract_info_cascades(save = False) :
    """
    creates 2 files with information about all the cascades : 
        - infos_cascades : 
            i:n_cascades - mid(int64) - date(pd.DateTime) - u(int32) - n_likes(int32) - n_reposts(int32) - users2(list(int32))
        - user_cascades : 
            i:v(int32) - mids(list(int64)) - Av(int32)

    """

    d_user_cascade = defaultdict(lambda : [[]]) # d_user_cascade[userid] = list of cascades in which user appears
    infos_cascades = []

    with open("data/weibo/weibodata/total.txt", 'r') as file :
        count_line = 0 
        mid_cascade = None

        for line in file:
        
            if count_line % 60000 == 0 : print(f"{count_line//2000}K cascades processed")
        
            if count_line%2 == 0 : #mid - date - user1 - #likes
                mid_cascade = np.int64(line.replace('\n', '').split()[0])
                infos_cascades.append(line.replace('\n', '').split())
        
            else : # (uid timestamp )*
                curr_cascade = line.replace('\n', '').split()
                users2 = []
                for i in range(0, len(curr_cascade), 2) :
                    u = np.int32(curr_cascade[i])
                    d_user_cascade[u][0].append(mid_cascade)
                    users2.append(u)
                    
                infos_cascades[-1].append(len(curr_cascade)//2) #adds the number of reposts
                infos_cascades[-1].append(users2) 
                
            count_line+= 1
    print(f"{count_line//2000}K cascades processed")
    user_cascades = pd.DataFrame.from_dict(d_user_cascade, orient='index', columns = ['mids'])
    user_cascades['#cascades'] = user_cascades.mids.apply(len)
    
    print(f"\nuser_cascades : \n{user_cascades.dtypes}")
    print(f"shape : {user_cascades.shape}\n")
    
    infos_cascades = pd.DataFrame(infos_cascades, columns = ['mid', 'date', 'User1', '#likes', '#reposts', 'users2'],)
    infos_cascades = infos_cascades.astype(dtype={'mid' : np.int64, 'User1' : np.int32, '#likes' : np.int32, '#reposts' : np.int32})
    infos_cascades['date'] = pd.to_datetime(infos_cascades.date)

    print(f"\ninfos_cascades : \n{infos_cascades.dtypes}" )
    print(f"shape : {infos_cascades.shape}\n")

    if save:
        user_cascades.to_pickle(output_folder + "user_cascades.pkl")
        infos_cascades.to_pickle(output_folder + "infos_cascades.pkl")

    return user_cascades, infos_cascades

def subsampling_1(n_cascades, n_influences_max, infos_cascades):
    """
    Only extracts the n_cascades first cascades and only the n_influences_max first reposts of each cascade
    
    """
    edges = pd.DataFrame.copy(infos_cascades.iloc[:n_cascades][['User1','users2']])
    edges.users2 = edges.users2.apply(lambda x : x[:n_influences_max])
    edges = edges.explode('users2')
    edges.columns = ['u','v']

    Au = edges.groupby('u').count().v
    Au.name = 'Au'
    Av = edges.groupby('v').count().u
    Av.name = 'Av'
    edges = edges.merge(Au, on='u')
    edges = edges.merge(Av, on='v')

    Au2v = edges.groupby(['u', 'v']).count().Au.reset_index()
    edges = edges.merge(Au2v, on=['u','v'])
    
    edges.columns = ['u', 'v', 'Au', 'Av', 'Au2v']
    return edges

def subsampling_2(n_cascades_max, n_min_reposts): 
    """
    Only selects edges between influencers and targets having more than n_min_reposts globally
    """
    targets = list(user_cascades[user_cascades['#cascades'] > n_min_reposts].index)
    d_targets = defaultdict(lambda : False)
    for t in targets : 
        d_targets[t] = True

    edges = pd.DataFrame.copy(infos_cascades.iloc[:n_cascades_max][['User1','users2']])
    edges.users2 = edges.users2.apply(lambda vs : [v for v in vs if d_targets[v]])
    
    edges = edges.explode('users2')
    edges.columns = ['u','v']

    Au = edges.groupby('u').count().v
    Au.name = 'Au'
    Av = edges.groupby('v').count().u
    Av.name = 'Av'

    edges = edges.merge(Au, on='u')
    edges = edges.merge(Av, on='v')

    Au2v = edges.groupby(['u', 'v']).count().Au.reset_index()
    edges = edges.merge(Au2v, on=['u','v'])
    
    edges.columns = ['u', 'v', 'Au', 'Av', 'Au2v']
    edges = edges.drop_duplicates()

    return edges

def estimate_probabilities(edges, save=False, file_name="") : 
    """
    input : edges dataframe 
    """
    edges['BT'] = edges['Au2v'] / edges['Au']
    edges['JI'] = edges['Au2v'] / (edges['Au'] + edges['Av'])
    edges['LP'] = edges['Au2v'] / edges['Av']

    edges = edges.merge(u_to_userids, left_on='u', right_on = 'uid',)
    edges = edges.merge(u_to_userids, left_on='v', right_on = 'uid',)
    edges['u'] = edges['userid_x']
    edges['v'] = edges['userid_y']


    edges = edges[['u', 'v', 'BT', 'JI', 'LP']]
    edges = edges.drop_duplicates()
    print("edges_probabilities : \n" + edges.head(5).to_markdown())
    print(f"shape : {edges.shape}\n")

    if save :
        print("Saved " + file_name)
        pd.to_pickle(edges, output_folder+file_name)
    
    return edges

def create_topic_file(save=False):
    
    d = {'mid' : []}
    for i in range(100) :
        d[i] = []

    with open('data/weibo/weibodata/topic-100/doc', 'r') as file : 
        i = 0
        for line in file :
            line = line.split('\t')
            if len(line) < 2 : continue
            
            d['mid'].append(np.int64(line[1]))
            for t in range(100) :
                d[int(line[2*t+2])].append(float(line[2*t+3]))

            i += 1
        df_topic = pd.DataFrame.from_dict(d)
        df_topic.index = df_topic.mid
        df_topic = df_topic.drop(columns='mid')
        del(d)
    
    print("df_topics : \n")
    print(f"shape : {df_topic.shape}\n")
    
    if save : 
        df_topic.to_pickle(output_folder + "topic.pkl")       
    return df_topic

def convert_new_topic(df_topic, save=False) : 
    """
    input : df_topic 
    output : df with N_TOPICS columns 
    """
    df_topic_sample = df_topic.sample(n = 10000)
    corr = df_topic_sample.corr().values

    pdist = spc.distance.pdist(corr, metric='euclidean')
    linkage = spc.linkage(pdist, method='complete')
    idx = spc.fcluster(linkage, N_TOPICS, 'maxclust')

    df_new = pd.DataFrame(np.zeros((df_topic.shape[0], N_TOPICS)), columns=[str(k) for k in range(N_TOPICS)])
    df_new.index = df_topic.index
    for i in range(100) : 
        new_i = idx[i] -1
        df_new[str(new_i)] += df_topic[i]
    
    if save : 
        df_new.to_pickle(output_folder + f"topics_{N_TOPICS}.pkl")
    return df_new

def convert_new_topic2(df_topic, save=False):
    LOW = [0, 49, 26, 13, 47, 73, 97, 80, 66, 55, 17, 68, 58, 59, 8, 92, 65, 15, 70, 61]
    MID = [45, 54, 85, 24, 96, 22, 32, 40, 77, 4, 74, 67, 41, 79, 71, 60, 95, 28, 38, 33, 20, 81, 63, 46, 27, 52, 34, 94, 18, 16, 53, 9, 10, 50, 91, 89, 48, 25, 19, 43, 93, 82, 83, 31, 30, 56, 69, 29, 88, 1, 84, 51, 12, 11, 62, 57, 3, 37, 78, 76]
    HIGH = [86, 64, 98, 35, 23, 72, 44, 21, 75, 90, 5, 87, 6, 2, 39, 14, 7, 99, 42, 36]
    df_new = pd.DataFrame(np.zeros((df_topic.shape[0], 3)), columns=['LowT', 'MidT', 'HighT'])
    df_new.index = df_topic.index
    for i in range(100) : 
        if i in LOW : 
            df_new['LowT'] += df_topic[i]
        elif i in MID :
            df_new['MidT'] += df_topic[i]
        elif i in HIGH : 
            df_new['HighT'] += df_topic[i]
    if save : 
        df_new.to_pickle(output_folder + f"topics_infl.pkl")
    return df_new

def create_topic_vector(mids) :
    """
    given a list of mids, estimates the average topic representation 
    """

    n = len(mids)
    s = np.zeros((3,))
    for mid in mids :
        s += np.array(d_new_topic[mid])
    return 1/n * s

def create_topic_per_target(save=False):
    """
    creates a df linking the uids of targets to the average topic representation of the reposted posts
    input : user_cascades
    """
    infos_targets = pd.DataFrame.copy(user_cascades[['mids']])
    infos_targets = infos_targets.merge(u_to_userids, left_on=infos_targets.index, right_on=u_to_userids.index)[['mids','userid']]

    # topics_users = infos_targets.mids.apply(create_topic_vector)
    # topics_users = pd.DataFrame(topics_users.to_list(), index = topics_users.index,)
    # topics_users.to_pickle(output_folder + "backup_topics_users_3.pkl")
    topics_users = pd.read_pickle(output_folder + "backup_topics_users_3.pkl")
    
    infos_targets = pd.concat([infos_targets, topics_users], axis=1)
    infos_targets = infos_targets.drop(columns="mids")
    infos_targets = infos_targets.set_index('userid')

    print("infos_targets : \n" + infos_targets.head(2).to_markdown())
    print(f"shape : {infos_targets.shape}\n")

    if save :
        print('Saving...')
        infos_targets.to_pickle(output_folder + "infos_targets_3.pkl")

    return infos_targets

def create_topic_per_influencer(save=False):
    """
    creates a df linking the uids of influencers to the average topic representation of the created posts
    """
    infos_influencers = infos_cascades.groupby('User1').agg({'mid' : list, 
                                                            '#likes' : 'sum', 
                                                            '#reposts' : 'sum', 
                                                            'users2' : 'count'})
    infos_influencers = infos_influencers.merge(u_to_userids, left_on = infos_influencers.index, right_on=u_to_userids.index)
    infos_influencers = infos_influencers.drop(columns='key_0')
    infos_influencers = infos_influencers.set_index('userid')
    infos_influencers.columns = ['mids','total_likes', 'total_reposts', 'n_cascades']

    df2 = infos_influencers.mids.apply(create_topic_vector) 
    df2 = pd.DataFrame(df2.to_list(), index = df2.index,)
    
    infos_influencers = pd.concat([infos_influencers, df2], axis=1)
    infos_influencers = infos_influencers.drop(columns="mids")

    print("infos_influencers : \n" + infos_influencers.head(2).to_markdown())
    print(f"shape : {infos_influencers.shape}\n")

    if save :
        print("Saving...")
        infos_influencers.to_pickle(output_folder + "infos_influencers_3.pkl")

    return infos_influencers

def create_edges(users_topology) : 
    """
    Creates weibo/edges_NB_CASCADES.pkl ---> |id|u|v|
    2.4M edges for 1000 cascades
    Takes 4min to extract
    """
    
    #Speeds up operation (v in users)
    d_users = defaultdict(lambda : False)
    for user in users_topology:
        d_users[int(user)] = True

    edges_topology = []                 

    with open("data/weibo/weibodata/graph_170w_1month.txt", 'r') as file :
        
        u_previous = -1
        u_in_table = False

        n_lines = 0
        
        for line in file :
            
            n_lines += 1
            if n_lines % 17000000 == 0 :
                print(f"{n_lines//1000000}M lines processed : {len(edges_topology)} edges added")

            line = line.split(' ')
            if len(line) >= 2 :
                u,v = int(line[0]), int(line[1])

                if u != u_previous : 
                    u_previous = u
                    u_in_table = (d_users[u])

                if u_in_table: 
                    if d_users[v] : 
                        edges_topology.append((u,v))
             
    return edges_topology

def process_edges(edges_topology, save = False, file_name = ""):
    """
    replaces user id with the real ones 
    """
    df_edges = pd.DataFrame(edges_topology, columns=['u','v'])
    df_edges = df_edges.merge(u_to_userids, how='inner', left_on='u', right_on='uid')\
                        .merge(u_to_userids, how='inner', left_on='v', right_on='uid')\
                        .drop(columns = ['u','v'])
    df_edges = df_edges.rename(columns = { 'userid_y' : 'u', 'userid_x' : 'v'})
    
    if save :
        print(f"Saved {file_name}")
        df_edges.to_pickle(output_folder + file_name)
    
    return df_edges

def extract_embeddings(file_path) : 
    """
    creates dict target2emb and influencer2emb
    """

    match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?')
    d_emb = {}
    emb = ''
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file :
            emb += line
            if line.endswith(']]\n'):
                l_temp = re.findall(match_number, emb)
                assert len(l_temp)==51
                number = [float(x) for x in l_temp[1:]]
                d_emb[l_temp[0]] =  number
                emb = '' 

    print("Embeddings extracted.")
    return d_emb

def process_embeddings(d, save=False, filename = "") : 
    df = pd.DataFrame.from_dict(d, orient='index')
    
    df.index = df.index.astype(np.int64)
    df = df.merge(u_to_userids, left_on = df.index, right_on=u_to_userids.index)
    df.index = df.userid
    df = df.drop(columns=['key_0', 'userid'])

    print("shape : " + str(df.shape))
    if save : 
        df.to_pickle(output_folder + filename)
    return df

#############################################################################################################################

parser = argparse.ArgumentParser()

parser.add_argument('--subsampling', type=int, default=2,help='')
parser.add_argument('--n1', type=int, default=-1,help='')
parser.add_argument('--n2', type=int, default=200,help='')

if __name__ == '__main__':

    #--- Parameters
    args=parser.parse_args()
    subsampling = int(args.subsampling)         
    n1 = int(args.n1)
    n2 = int(args.n2)
    
        
    ### User profile infos ###
    user_profile = convert_userProfile(True)
    # user_profile = pd.read_pickle(output_folder + "userProfile.pkl")

    ### Cascades infos ###
    user_cascades, infos_cascades = extract_info_cascades(True)
    # user_cascades = pd.read_pickle(output_folder + "user_cascades.pkl")
    # infos_cascades = pd.read_pickle(output_folder + "infos_cascades.pkl")

    ### Subsampling and db label estimation ###
    if subsampling == 1 : 
        edges = subsampling_1(n1, n2) #n1 = #cascades, n2 = #reposts per cascades
    elif subsampling == 2 : 
        edges = subsampling_2(n1, n2)
    else : print("wrong subsampling")

    i = len(edges.groupby('u').count().index)
    t = len(edges.groupby('v').count().index)
    print(edges.shape, i, t)
    
    labels = estimate_probabilities(edges, True, f"labels{subsampling}_{n1}_{n2}.pkl")

    # ### Topic infos ###
    df_topic = create_topic_file(True)
    df_topic = pd.read_pickle(output_folder + "topic.pkl")
    N_TOPICS = 3
    df_new_topic = convert_new_topic2(df_topic, True)
    # df_new_topic = pd.read_pickle(output_folder + f"topics_3.pkl")    
    d_new_topic = defaultdict(lambda : 1/N_TOPICS * np.ones(N_TOPICS))
    for mid in df_new_topic.index : 
        d_new_topic[mid] = np.array(df_new_topic.loc[mid])
    infos_targets = create_topic_per_target(True)
    infos_influencers = create_topic_per_influencer(True)

    ### Topology infos ###
    influencers = list(edges.groupby('u').count().index) #with uid
    targets = list(edges.groupby('v').count().index)
    users_topology = sorted(list(set(influencers + targets)))

    print(f"I = {len(influencers)}, T = {len(targets)}")
    print(f"users_topology : must be <1.8M \n {users_topology[-10:]} ..." )
    print(f"length : {len(users_topology)}\n")

    edges_topology = create_edges(users_topology)
    df_edges = process_edges(edges_topology, True, f"edges{subsampling}_{n1}_{n2}.pkl")

    ### Embeddings infos ###
    # target2emb = extract_embeddings("papers\IMINFECTOR\data\Embeddings\inf2vec_target_embeddings_7.txt")
    # influencer2emb = extract_embeddings("papers\IMINFECTOR\data\Embeddings\inf2vec_source_embeddings_7.txt")
    # target2emb = process_embeddings(target2emb, True, "target_inf2vec.pkl")
    # influencer2emb = process_embeddings(influencer2emb, True, "influencers_inf2vec.pkl")

    # target2emb = extract_embeddings("data\weibo\weibo_embedding\mtl_n_target_embeddings_p.txt")
    # influencer2emb = extract_embeddings("data\weibo\weibo_embedding\mtl_n_source_embeddings_p.txt")
    # target2emb = process_embeddings(target2emb, True, "target_infector.pkl")
    # influencer2emb = process_embeddings(influencer2emb, True, "influencers_infector.pkl")
