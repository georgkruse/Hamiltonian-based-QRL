import yaml
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import itertools
from utils.config.common import extract_hyperparameters



path = 'logs/w-maxcut/dynamic5/nodes/qdqn/2024-03-27--09-43-46_QRL_QDQN'

results_file_name = "/result.json"
result_files  = [f.path for f in os.scandir(path) if f.is_dir() ]
results = [pd.read_json(f + results_file_name,lines=True) for f in result_files]

cust_ = []
for i, data_exp in enumerate(results):
    cust_.append(np.mean(data_exp['custom_metrics'].values[0]['approximation_ratio']))

mean_node_init = np.mean(cust_)

path = 'logs/w-maxcut/dynamic5/edges/qdqn/2024-03-26--17-31-39_QRL_QDQN'

results_file_name = "/result.json"
result_files  = [f.path for f in os.scandir(path) if f.is_dir() ]
results = [pd.read_json(f + results_file_name,lines=True) for f in result_files]

cust_ = []
for i, data_exp in enumerate(results):
    cust_.append(np.mean(data_exp['custom_metrics'].values[0]['approximation_ratio']))

mean_edge_init = np.mean(cust_)





path = 'logs/w-maxcut/dynamic5/edges/qpg/2024-03-26--21-19-20_QRL_QPG'

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


# Plot all runs in one plot
fig, (ax_node, ax_edge) = plt.subplots(1, 2, figsize=(8, 6))

info_text = []
if 'lr' not in key_names:
    info_text.append(f"lr: {config['algorithm_config']['lr']} \n")
if 'lr_output_scaling' not in key_names:
    info_text.append(f"lr_out: {config['algorithm_config']['lr_output_scaling']} \n")
if 'num_layers' not in key_names:
    info_text.append(f"num_layers: {config['algorithm_config']['num_layers']} \n")

info_text = ''.join(info_text)

fig_name = f"{config['env']}, {folder_path}"
title = f'{fig_name}'

types = {} 
labels = []
trial_names = [trial_names[0]]
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
            cust_ = []
            for idx in range(data_exp['custom_metrics'].shape[0]):
                cust_.append(np.mean(data_exp['custom_metrics'].values[idx]['approximation_ratio']))
            curves[int(key)].append(cust_)
            # curves[int(key)].append(data_exp[key_y_axis].values)
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
        ax_edge.plot(x_axis, mean, label='$QPG - sppgl$', color='darkcyan')
        ax_edge.fill_between(x_axis, lower, upper, alpha=0.5, color='darkcyan')
    else:
        print('One trial seems to be empty')


path = 'logs/w-maxcut/dynamic5/edges/qdqn/2024-03-26--17-33-05_QRL_QDQN'

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


info_text = []
if 'lr' not in key_names:
    info_text.append(f"lr: {config['algorithm_config']['lr']} \n")
if 'lr_output_scaling' not in key_names:
    info_text.append(f"lr_out: {config['algorithm_config']['lr_output_scaling']} \n")
if 'num_layers' not in key_names:
    info_text.append(f"num_layers: {config['algorithm_config']['num_layers']} \n")

info_text = ''.join(info_text)

fig_name = f"{config['env']}, {folder_path}"
title = f'{fig_name}'

types = {} 
labels = []
trial_names = [trial_names[0]]
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
            cust_ = []
            for idx in range(data_exp['custom_metrics'].shape[0]):
                cust_.append(np.mean(data_exp['custom_metrics'].values[idx]['approximation_ratio']))
            curves[int(key)].append(cust_)
            # curves[int(key)].append(data_exp[key_y_axis].values)
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
        ax_edge.plot(x_axis, mean, label='$QDQN - sppgl$', color='indianred')
        ax_edge.fill_between(x_axis, lower, upper, alpha=0.5, color='indianred')
    else:
        print('One trial seems to be empty')


path = 'logs/w-maxcut/dynamic5/nodes/qpg/2024-03-27--10-21-07_QRL_QPG'

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


types = {} 
labels = []
trial_names = [trial_names[0]]
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
            cust_ = []
            for idx in range(data_exp['custom_metrics'].shape[0]):
                cust_.append(np.mean(data_exp['custom_metrics'].values[idx]['approximation_ratio']))
            curves[int(key)].append(cust_)
            # curves[int(key)].append(data_exp[key_y_axis].values)
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
        ax_edge.plot(x_axis, mean, label='$QPG - sppgl+hea$', color='darkcyan', alpha=0.5, linestyle='dashed')
        ax_edge.fill_between(x_axis, lower, upper, alpha=0.25, color='darkcyan')
    else:
        print('One trial seems to be empty')

path = 'logs/w-maxcut/dynamic5/edges/qdqn/2024-03-27--13-30-07_QRL_QDQN'

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


types = {} 
labels = []
trial_names = [trial_names[0]]
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
            cust_ = []
            for idx in range(data_exp['custom_metrics'].shape[0]):
                cust_.append(np.mean(data_exp['custom_metrics'].values[idx]['approximation_ratio']))
            curves[int(key)].append(cust_)
            # curves[int(key)].append(data_exp[key_y_axis].values)
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
        ax_edge.plot(x_axis, mean, label='$QDQN - sppgl+hea$', color='indianred', alpha=0.5, linestyle='dashed')
        ax_edge.fill_between(x_axis, lower, upper, alpha=0.35, color='indianred')
    else:
        print('One trial seems to be empty')

ax_edge.hlines(y=mean_edge_init, xmin=0, xmax=50000, linestyles='dashdot', color='grey', label='$sppg-init$')
ax_edge.hlines(y=1.0, xmin= 0, xmax=50000, linestyles='-', color='grey')

# axis.set_ylim(-30, 3)
ax_edge.set_xlabel("$environment \, steps$", fontsize=15)
# ax_edge.set_ylabel('$approximation ratio$', fontsize=15)
ax_edge.set_ylim(0.75, 1.01)
ax_edge.set_title('$Edge \, Measurement$',fontsize=15)
ax_edge.legend(fontsize=12, loc='lower right')
ax_edge.minorticks_on()
ax_edge.grid(which='major', alpha=0.4)




##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################




path = 'logs/w-maxcut/dynamic5/nodes/qpg/2024-03-26--21-24-56_QRL_QPG'

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



info_text = []
if 'lr' not in key_names:
    info_text.append(f"lr: {config['algorithm_config']['lr']} \n")
if 'lr_output_scaling' not in key_names:
    info_text.append(f"lr_out: {config['algorithm_config']['lr_output_scaling']} \n")
if 'num_layers' not in key_names:
    info_text.append(f"num_layers: {config['algorithm_config']['num_layers']} \n")

info_text = ''.join(info_text)

fig_name = f"{config['env']}, {folder_path}"
title = f'{fig_name}'

types = {} 
labels = []
trial_names = [trial_names[0]]
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
            cust_ = []
            for idx in range(data_exp['custom_metrics'].shape[0]):
                cust_.append(np.mean(data_exp['custom_metrics'].values[idx]['approximation_ratio']))
            curves[int(key)].append(cust_)
            # curves[int(key)].append(data_exp[key_y_axis].values)
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
        ax_node.plot(x_axis, mean, label='$QPG - sppgl$', color='darkcyan',)
        ax_node.fill_between(x_axis, lower, upper, alpha=0.5, color='darkcyan',)
    else:
        print('One trial seems to be empty')


path = 'logs/w-maxcut/dynamic5/nodes/qdqn/2024-03-26--21-32-56_QRL_QDQN'

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


types = {} 
labels = []
trial_names = [trial_names[0]]
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
            cust_ = []
            for idx in range(data_exp['custom_metrics'].shape[0]):
                cust_.append(np.mean(data_exp['custom_metrics'].values[idx]['approximation_ratio']))
            curves[int(key)].append(cust_)
            # curves[int(key)].append(data_exp[key_y_axis].values)
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
        ax_node.plot(x_axis, mean, label='$QDQN - sppgl$', color='indianred',)
        ax_node.fill_between(x_axis, lower, upper, alpha=0.5, color='indianred',)
    else:
        print('One trial seems to be empty')


path = 'logs/w-maxcut/dynamic5/nodes/qpg/2024-03-27--10-20-09_QRL_QPG'

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


types = {} 
labels = []
trial_names = [trial_names[0]]
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
            cust_ = []
            for idx in range(data_exp['custom_metrics'].shape[0]):
                cust_.append(np.mean(data_exp['custom_metrics'].values[idx]['approximation_ratio']))
            curves[int(key)].append(cust_)
            # curves[int(key)].append(data_exp[key_y_axis].values)
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
        ax_node.plot(x_axis, mean, label='$QPG - sppgl+hea$', color='darkcyan', alpha=0.5, linestyle='dashed')
        ax_node.fill_between(x_axis, lower, upper, alpha=0.25, color='darkcyan')
    else:
        print('One trial seems to be empty')

path = 'logs/w-maxcut/dynamic5/nodes/qdqn/2024-03-27--13-31-32_QRL_QDQN'

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


types = {} 
labels = []
trial_names = [trial_names[0]]
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
            cust_ = []
            for idx in range(data_exp['custom_metrics'].shape[0]):
                cust_.append(np.mean(data_exp['custom_metrics'].values[idx]['approximation_ratio']))
            curves[int(key)].append(cust_)
            # curves[int(key)].append(data_exp[key_y_axis].values)
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
        ax_node.plot(x_axis, mean, label='$QDQN - sppgl+hea$', color='indianred', alpha=0.5, linestyle='dashed')
        ax_node.fill_between(x_axis, lower, upper, alpha=0.35, color='indianred')
    else:
        print('One trial seems to be empty')

ax_node.hlines(y=mean_node_init, xmin= 0, xmax=50000, linestyles='dashdot', color='grey', label='$sppg-init$')
ax_node.hlines(y=1.0, xmin= 0, xmax=50000, linestyles='-', color='grey')
ax_node.set_xlabel("$environment \, steps$", fontsize=15)
ax_node.set_ylabel('$approximation \, ratio$', fontsize=15)
ax_node.set_ylim(0.75, 1.01)
ax_node.set_title('$Node \, Measurement$', fontsize=15)
ax_node.legend(fontsize=12, loc='lower right')
ax_node.minorticks_on()
ax_node.grid(which='major', alpha=0.4)


# ax_edge.set_title(title, fontsize=15)
# plt.text(1.01, 0.6, info_text, transform=ax_main.transAxes, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
fig.tight_layout()
plt.savefig('comparision1.pdf', dpi=600)
print('Done')
