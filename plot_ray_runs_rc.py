import yaml
import os
import ray
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from glob import glob
import numpy as np
import re

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


fig, axis = plt.subplots(figsize=(8,7))

root_path = '/home/users/coelho/quantum-computing/QRL/logs/maxcut_weighted/dynamic/5nodes/complete_graph/skolik/grid_layers/'
paths, labels, types = [], [], []
# paths += [root_path + '2023-12-27--11-39-44_QRL_QPG'] #'2023-12-27--10-13-20_QRL_QPG']
# paths += [root_path + '2023-12-27--12-29-23_QRL_QPG'] #'2023-12-27--10-13-20_QRL_QPG']
paths += [root_path + '2024-03-04--09-41-29_QRL_QPG']

title = 'MaxCut Weighted Dynamic 5 Nodes (Complete Graphs Skolik) 10 Layers'
fig_name = f'{title}'
data_length = 500
#optimal_cost = -82.16
#max_cost = 222.1


labels += [['s-ppgl',
            'm-ppgl',
            'h-ppgl',
            's-ppgl-linear',
            'm-ppgl-linear'
        ]]


#labels += [[    'output_scaling_lr=0.2500',
#                'output_scaling_lr=0.1000',
#                'output_scaling_lr=0.0500'
#        ]]


#types += [{ '0': ['lr_output_scaling=0.2500'],
#            '1': ['lr_output_scaling=0.1000'],
#            '2': ['lr_output_scaling=0.0500']
#        }]

types += [{'0': ['=s-ppgl,num_layers=10'],
           '1': ['=m-ppgl,num_layers=10'],
           '2': ['=h-ppgl,num_layers=10'],
           '3': ['=s-ppgl-linear,num_layers=10'],
           '4': ['=m-ppgl-linear,num_layers=10']
        }]

for idx, path in enumerate(paths):
    curves = [[] for _ in range(len(labels[idx]))]
    curves_x = [[] for _ in range(len(labels[idx]))]


    results_file_name = "/result.json"
    result_files  = [f.path for f in os.scandir(path) if f.is_dir() ]
    results = [pd.read_json(f + results_file_name,lines=True) for f in result_files]
    result = pd.concat(results)
    
    for key, value in types[idx].items():
        for i, data_exp in enumerate(results):
            if all(x in result_files[i] for x in value):
                #approx_ratio = []
                #for j in range(len(data_exp["custom_metrics"])):
                #    approx_ratio.append(abs(np.mean(data_exp["custom_metrics"][j]["approximation_ratio"])))
                #curves[int(key)].append(approx_ratio[:data_length])
                curves_x[int(key)].append(data_exp['num_env_steps_sampled'].values[:data_length])
                curves[int(key)].append(data_exp['episode_reward_mean'].values[:data_length])


    for id, curve in enumerate(curves):
        data = np.vstack(curve)
        #for j in range(len(data)):
            #for i in range(len(data[j])):
                #data[j][i] =  data[j][i] / (max_cost - optimal_cost)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        upper = mean +  std
        lower = mean  -  std
        axis.plot(data_exp['num_env_steps_sampled'].values[:data_length], mean, label=labels[idx][id])
        axis.fill_between(data_exp['num_env_steps_sampled'].values[:data_length], lower, upper, alpha=0.5)



#axis.hlines(y=30.267, xmin= 0, xmax=50000, linestyles='dashdot', color='black')
axis.set_ylim(3.25, 3.85)
axis.set_xlabel("$environment \, steps$", fontsize=13)
axis.set_ylabel("$Return$", fontsize=15)
axis.set_title(title, fontsize=15)
axis.legend(fontsize=12, loc='lower left')
axis.minorticks_on()
axis.grid(which='both', alpha=0.4)
fig.tight_layout()
fig.savefig(f'{fig_name}.png', dpi=100)
