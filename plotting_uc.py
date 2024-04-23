import yaml
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import itertools
from utils.config.common import extract_hyperparameters


# Plot all runs in one plot
fig, (uc_5, uc_10, uc_15) = plt.subplots(1, 3, figsize=(14, 4))

path = 'logs/uc/paper_2/15units/joint'
colors = ['darkcyan', 'indianred', 'slategray', 'peru', 'darkcyan', 'mediumseagreen', 'cornflowerblue']

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

folder_path = os.path.basename(os.path.normpath(path))

with open(path + '/alg_config.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

key_names, hyperparameters, single_elements = extract_hyperparameters(config['algorithm_config'])

combinations = list(itertools.product(*hyperparameters))
trial_names, label_names = [], []
for comb in combinations:
    trials = []
    for (name, element) in zip(key_names, comb):
        if type(element) == int:
            trials.append(f'{name}={str(element)},')
        elif type(element) == float:
            trials.append(f'{name}={element:.4f},')
        else:
            trials.append(f'{name}={str(element)},')
    trial_names.append(trials)
    label_names.append([str(element) for element in comb])



label_names = [
    '$sge-sgv$',
    '$mge-sgv$',
    '$mge-mgv$',
    '$sge-sgv+hea$',
    # '$hppgl+hea$',
    # '$mppgl+hea$'
]

key_y_axis = 'episode_reward_mean'
types = {} 
labels = []
trial_names = np.stack(trial_names)[np.array([0,1,2,4])]
for i, name in enumerate(trial_names):
    types[str(i)] = name
    labels.append(label_names[i])

curves = [[] for _ in range(i+1)]
curves_x_axis = [[] for _ in range(i+1)]

results_file_name = "/result.json"
result_files  = [f.path for f in os.scandir(path) if f.is_dir() ]
results = [pd.read_json(f + results_file_name,lines=True) for f in result_files]

for key, value in types.items():
    for i, data_exp in enumerate(results):
        if all(x in result_files[i] for x in value):
            curves[int(key)].append(data_exp[key_y_axis].values)
            curves_x_axis[int(key)].append(data_exp['num_env_steps_sampled'].values)                    
    
for id, curve in enumerate(curves):
    if len(curve) >= 1:
        min_length = min([len(d) for d in curve])
        data = [d[:min_length] for d in curve]
        x_axis = curves_x_axis[id][0][:min_length]
        data = np.vstack(data)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        upper = mean +  std
        lower = mean  -  std
        uc_15.plot(x_axis, mean, label=labels[id], color=colors[id])
        uc_15.fill_between(x_axis, lower, upper, alpha=0.5, color=colors[id])
    else:
        print('One trial seems to be empty')




path = 'logs/uc/paper_2/5units/2024-04-09--09-01-13_QRL_QPG'

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

folder_path = os.path.basename(os.path.normpath(path))

with open(path + '/alg_config.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

key_names, hyperparameters, single_elements = extract_hyperparameters(config['algorithm_config'])

combinations = list(itertools.product(*hyperparameters))
trial_names, label_names = [], []
for comb in combinations:
    trials = []
    for (name, element) in zip(key_names, comb):
        if type(element) == int:
            trials.append(f'{name}={str(element)},')
        elif type(element) == float:
            trials.append(f'{name}={element:.4f},')
        else:
            trials.append(f'{name}={str(element)},')
    trial_names.append(trials)
    label_names.append([str(element) for element in comb])



label_names = [
    '$sge-sgv$',
    '$mge-sgv$',
    '$mge-mgv$',
    '$sge-sgv+hea$',
    # '$hppgl+hea$',
    # '$mppgl+hea$'
]

colors = ['darkcyan', 'indianred', 'slategray', 'peru', 'darkcyan', 'mediumseagreen', 'cornflowerblue']

key_y_axis = 'episode_reward_mean'
types = {} 
labels = []
trial_names = np.stack(trial_names)[np.array([0,1,2,4])]
for i, name in enumerate(trial_names):
    types[str(i)] = name
    labels.append(label_names[i])

curves = [[] for _ in range(i+1)]
curves_x_axis = [[] for _ in range(i+1)]

results_file_name = "/result.json"
result_files  = [f.path for f in os.scandir(path) if f.is_dir() ]
results = [pd.read_json(f + results_file_name,lines=True) for f in result_files]

for key, value in types.items():
    for i, data_exp in enumerate(results):
        if all(x in result_files[i] for x in value):
            curves[int(key)].append(data_exp[key_y_axis].values)
            curves_x_axis[int(key)].append(data_exp['num_env_steps_sampled'].values)                    
    
for id, curve in enumerate(curves):
    if len(curve) >= 1:
        min_length = min([len(d) for d in curve])
        data = [d[:min_length] for d in curve]
        x_axis = curves_x_axis[id][0][:min_length]
        data = np.vstack(data)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        upper = mean +  std
        lower = mean  -  std
        uc_5.plot(x_axis, mean, label=labels[id], color=colors[id])
        uc_5.fill_between(x_axis, lower, upper, alpha=0.5, color=colors[id])
    else:
        print('One trial seems to be empty')


path = 'logs/uc/paper_2/10units/2024-04-09--08-58-10_QRL_QPG'

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

folder_path = os.path.basename(os.path.normpath(path))

with open(path + '/alg_config.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

key_names, hyperparameters, single_elements = extract_hyperparameters(config['algorithm_config'])

combinations = list(itertools.product(*hyperparameters))
trial_names, label_names = [], []
for comb in combinations:
    trials = []
    for (name, element) in zip(key_names, comb):
        if type(element) == int:
            trials.append(f'{name}={str(element)},')
        elif type(element) == float:
            trials.append(f'{name}={element:.4f},')
        else:
            trials.append(f'{name}={str(element)},')
    trial_names.append(trials)
    label_names.append([str(element) for element in comb])




label_names = [
    '$sge-sgv$',
    '$mge-sgv$',
    '$mge-mgv$',
    '$sge-sgv+hea$',
    # '$hppgl+hea$',
    # '$mppgl+hea$'
]

key_y_axis = 'episode_reward_mean'
types = {} 
labels = []
trial_names = np.stack(trial_names)[np.array([0,1,2,4])]
for i, name in enumerate(trial_names):
    types[str(i)] = name
    labels.append(label_names[i])

curves = [[] for _ in range(i+1)]
curves_x_axis = [[] for _ in range(i+1)]

results_file_name = "/result.json"
result_files  = [f.path for f in os.scandir(path) if f.is_dir() ]
results = [pd.read_json(f + results_file_name,lines=True) for f in result_files]

for key, value in types.items():
    for i, data_exp in enumerate(results):
        if all(x in result_files[i] for x in value):
            curves[int(key)].append(data_exp[key_y_axis].values)
            curves_x_axis[int(key)].append(data_exp['num_env_steps_sampled'].values)                    
    
for id, curve in enumerate(curves):
    if len(curve) >= 1:
        min_length = min([len(d) for d in curve])
        data = [d[:min_length] for d in curve]
        x_axis = curves_x_axis[id][0][:min_length]
        data = np.vstack(data)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        upper = mean +  std
        lower = mean  -  std
        uc_10.plot(x_axis, mean, label=labels[id], color=colors[id])
        uc_10.fill_between(x_axis, lower, upper, alpha=0.5, color=colors[id])
    else:
        print('One trial seems to be empty')


# path = 'logs/uc/paper/10units/2024-03-27--15-37-38_QRL_QPG'

# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# folder_path = os.path.basename(os.path.normpath(path))

# with open(path + '/alg_config.yml') as f:
#     config = yaml.load(f, Loader=yaml.FullLoader)

# key_names, hyperparameters, single_elements = extract_hyperparameters(config['algorithm_config'])

# combinations = list(itertools.product(*hyperparameters))
# trial_names, label_names = [], []
# for comb in combinations:
#     trials = []
#     for (name, element) in zip(key_names, comb):
#         if type(element) == int:
#             trials.append(f'{name}={str(element)},')
#         elif type(element) == float:
#             trials.append(f'{name}={element:.4f},')
#         else:
#             trials.append(f'{name}={str(element)},')
#     trial_names.append(trials)
#     label_names.append([str(element) for element in comb])



# label_names = [
#     # '$sppgl$',
#     # '$hppgl$',
#     # '$mppgl$',
#     '$sppgl+hea$',
#     '$hppgl+hea$',
#     '$mppgl+hea$'
# ]

# key_y_axis = 'episode_reward_mean'
# types = {} 
# labels = []

# for i, name in enumerate(trial_names):
#     types[str(i)] = name
#     labels.append(label_names[i])

# curves = [[] for _ in range(i+1)]
# curves_x_axis = [[] for _ in range(i+1)]

# results_file_name = "/result.json"
# result_files  = [f.path for f in os.scandir(path) if f.is_dir() ]
# results = [pd.read_json(f + results_file_name,lines=True) for f in result_files]

# for key, value in types.items():
#     for i, data_exp in enumerate(results):
#         if all(x in result_files[i] for x in value):
#             curves[int(key)].append(data_exp[key_y_axis].values)
#             curves_x_axis[int(key)].append(data_exp['num_env_steps_sampled'].values)                    
    
# for id, curve in enumerate(curves):
#     if len(curve) >= 1:
#         min_length = min([len(d) for d in curve])
#         data = [d[:min_length] for d in curve]
#         x_axis = curves_x_axis[id][0][:min_length]
#         data = np.vstack(data)
#         mean = np.mean(data, axis=0)
#         std = np.std(data, axis=0)
#         upper = mean +  std
#         lower = mean  -  std
#         uc_10.plot(x_axis, mean, label=labels[id], color=colors[id+3])
#         uc_10.fill_between(x_axis, lower, upper, alpha=0.5, color=colors[id+3])
#     else:
#         print('One trial seems to be empty')




# ax_pg.hlines(y=mean_edge_init, xmin= 0, xmax=50000, linestyles='dashdot', color='black', label='$sppg-init$')
# axis.set_ylim(-30, 3)
uc_5.set_xlabel("$environment \, steps$", fontsize=15)
# ax_pg.set_ylabel('$approximation ratio$', fontsize=15)
# uc_5.set_ylim(0.75, 1.01)
uc_5.set_xticks([0, 50000, 100000, 150000, 200000])
uc_5.set_title('$5 \, \, Units$', fontsize=15)
uc_5.legend(fontsize=12, loc='lower right')
uc_5.minorticks_on()
uc_5.grid(which='major', alpha=0.4)

uc_10.set_xlabel("$environment \, steps$", fontsize=15)
# ax_pg.set_ylabel('$approximation ratio$', fontsize=15)
# uc_5.set_ylim(0.75, 1.01)
uc_10.set_xticks([0, 50000, 100000, 150000, 200000])
uc_10.set_title('$10 \, \, Units$', fontsize=15)
uc_10.legend(fontsize=12, loc='lower right')
uc_10.minorticks_on()
uc_10.grid(which='major', alpha=0.4)

uc_5.set_ylabel("$reward$", fontsize=15)
uc_15.set_xlabel("$environment \, steps$", fontsize=15)
# ax_pg.set_ylabel('$approximation ratio$', fontsize=15)
# uc_5.set_xlim(0, 150000)
# uc_10.set_xlim(0, 150000)
# uc_15.set_xlim(0, 150000)

uc_5.set_xticks([0, 50000, 100000, 150000])
uc_10.set_xticks([0, 50000, 100000, 150000])
uc_15.set_xticks([0, 50000, 100000, 150000])
uc_15.set_title('$15 \, \, Units$', fontsize=15)
uc_15.legend(fontsize=12, loc='lower right')
uc_15.minorticks_on()
uc_15.grid(which='major', alpha=0.4)


# ax_pg.set_title(title, fontsize=15)
# plt.text(1.01, 0.6, info_text, transform=ax_main.transAxes, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
fig.tight_layout()
plt.savefig('comparision5.pdf', dpi=600)
print('Done')
