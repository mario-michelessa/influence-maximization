import os
import torch
import numpy as np
import random
import pandas as pd

num_targets = 500
num_items = 100

num_instances = 100
test_pct = 0.2
num_random_iter = 30


'''
make movie info dataframe
'''

f = open('ml-100k/u.item', encoding = "ISO-8859-1")
data = f.readlines()
f.close()

lines = []
for line in data:
    line = list(map(int, line.split('|')[5:]))
    lines.append(line)

item_df = pd.DataFrame(lines, columns = ['g{}'.format(i) for i in range(1,20)])


'''
make user info dataframe
'''

f = open('ml-100k/u.user', encoding = "ISO-8859-1")
data = f.readlines()
f.close()

lines = []
for line in data:
    line = line.split('|')
    lines.append(line)

df = pd.DataFrame(lines, columns = ['id', 'age', 'sex', 'job', 'zipcode'])
sex_list = list(set(list(df['sex'])))
job_list = list(set(list(df['job'])))

def featvec(feat, clist):
    output = [0] * len(clist)
    output[clist.index(feat)] = 1
    return output

new_lines = []
for line in lines:
    new_line = [int(line[1])] + featvec(line[2], sex_list) + featvec(line[3], job_list)
    new_lines.append(new_line)

user_df = pd.DataFrame(new_lines, columns = ['age'] + sex_list + job_list)


'''
make link probability matrix
'''

f = open('ml-100k/u.data')
data = f.readlines()
f.close()

data = np.array([list(map(int, d.split()[:-1])) for d in data])
P_full = np.zeros([np.max(data[:,1]) , np.max(data[:,0])])
for d in data: 
    P_full[d[1]-1, d[0]-1] = .02 * d[2] 


'''
make features
'''
num_movies, num_movie_features = item_df.shape
num_users, num_user_features = user_df.shape
num_features = num_movie_features + num_user_features

features = np.zeros([num_movies, num_users, num_features])

for u in range(num_users):
    features[:, u, :] = np.c_[np.tile(user_df.iloc[u],(num_movies, 1)), np.array(item_df)]


'''
make random instances
'''

Ps   = np.zeros([num_instances, num_items, num_targets])
data = np.zeros([num_instances, num_items, num_targets, num_features])
for t in range(num_instances):
    m_idx = random.sample(range(np.shape(P_full)[1]), num_items)
    u_idx = random.sample(range(np.shape(P_full)[1]), num_targets)
    P_used  = P_full[m_idx,:]
    Ps[t] = P_used[:, u_idx]
    data[t] = features[m_idx, :, :][:, u_idx, :]


path = 'instances/'
if os.path.exists(path + 'Ps.npz') == False:
    np.savez(path + 'Ps.npz', Ps = Ps)
if os.path.exists(path + 'data.npz') == False:
    np.savez(path + 'data.npz', data = data)

for i in range(num_random_iter):
    test = random.sample(range(num_instances), int(test_pct*num_instances))
    train = [i for i in range(num_instances) if i not in test]
    if os.path.exists(path + '{}.npz'.format(i)) == False:
        np.savez(path + '{}'.format(i), train = train, test = test)
