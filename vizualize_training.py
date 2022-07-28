
import numpy as np 
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', type=str, default='results/')
parser.add_argument('--title', type=str, default='')
args=parser.parse_args()
input_dir = args.input_dir
title = args.title

l = []
with open(input_dir + '2s_training.txt', 'r') as file : 
    
    model = -1
    for line in file : 
        line = line.replace('\n', '').split(' | ')
        if len(line) > 2 :
            if line[0] == '0' :
                model += 1 
            l.append([model] + line) 
df = pd.DataFrame(l).astype(float)

fig, ax = plt.subplots(2,3,figsize = (20,15))
titles = ['loss', 'spread-train', 'spread-test', 'DNI-train', 'DNI-test', 'avgDP']
for i,ax in enumerate(ax.reshape(-1)) :
    if i < 6 :
        sns.lineplot(data = df, x=1, y=i+2, hue=0, ax=ax)
        ax.get_legend().remove()
        ax.set_xlabel('epochs')
        ax.set_ylabel(titles[i])

fig.suptitle(title)
        

plt.show()