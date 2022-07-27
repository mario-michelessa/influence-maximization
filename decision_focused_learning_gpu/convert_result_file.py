"""
Extracts mean / std tables from the results file returned by results.py and grd_main.py --> perfs_train_test.txt / perfs_sparse.txt
"""

import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n-iter', type=int, default=15)
parser.add_argument('--file-path', type=str, default='results/perfs_sparse.txt')
parser.add_argument('--sparse', type=bool, default=False)

args=parser.parse_args()
n_iter = int(args.n_iter)
file_path = args.file_path
sparse = bool(args.sparse)

names_sparse = ["Exp df", "Exp 2s", "Exp rnd", "Exp grd", "Exp deg", "DNI df", "DNI 2s", "DNI rnd", "DNI grd",  "DNI deg"]
names_train_test = ["Exp-train df",  "Exp-train 2s", "Exp-train rnd",  "Exp-train grd", "Exp-train deg", 
                    "DNI-train df",  "DNI-train 2s", "DNI-train rnd",  "DNI-train grd",  "DNI-train deg",
                    "Exp-test df", "Exp-test 2s", "Exp-test rnd", "Exp-test grd", "Exp-test deg", 
                    "DNI-test df", "DNI-test 2s", "DNI-test rnd", "DNI-test grd", "DNI-test deg" ]

with open(file_path, 'r') as file : 
    names = names_sparse if sparse else names_train_test
    d =  {}
    for cat in names : d[cat] = np.zeros((n_iter, 5))
    n = -1
    for line in file : 
        line = line.split(',')
        if line[0] == 'Exp df' or line[0] == 'Exp-train df' : 
            n += 1
        if line[0] in d.keys() : 
            d[line[0]][n,:] = [float(x) for x in line[1:]]
    
d_mean, d_std = {}, {}
for k in d.keys() : 
    d_mean[k] = np.mean(d[k], axis=0)
    d_std[k] = np.std(d[k], axis=0)

df_mean = pd.DataFrame.from_dict(d_mean, orient='index')
df_std = pd.DataFrame.from_dict(d_std, orient='index')
print(df_mean.to_markdown())
print(df_std.to_markdown())
df_mean.to_csv(file_path.replace('.txt', '_mean.csv'))
df_std.to_csv(file_path.replace('.txt', '_std.csv'))

