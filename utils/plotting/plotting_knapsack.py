import yaml
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import itertools
from utils.config.common import extract_hyperparameters


# Plot all runs in one plot
fig, (ax_5, ax_10, ax_val) = plt.subplots(1, 3, figsize=(14, 5))

qrl_results_path = "results_parallel.npy"
qrl_results = np.load(qrl_results_path,allow_pickle=True).item()

types = ["slack","unbalanced"]
problem_sizes = [4,5,6,7,8,9,10]
dataset_size = 100

# Plot the data as a vertical scatter plot
for i,size in enumerate(problem_sizes):
    ax_val.scatter([size - 0.3] * dataset_size, qrl_results["approximation_ratios"][f"{size}"]["=s-ppgl"]["test"], marker = "o",color='darkcyan', alpha=0.3, label="s-ppgl: training" if i == 0 else None)
    ax_val.scatter([size - 0.1] * dataset_size, qrl_results["approximation_ratios"][f"{size}"]["=s-ppgl"]["validation"], marker = "x",color='darkcyan', alpha=0.3, label="s-ppgl: validation" if i == 0 else None)
    ax_val.scatter([size + 0.1] * dataset_size, qrl_results["approximation_ratios"][f"{size}"]["=angular"]["test"], marker = "o",color='peru', alpha=0.3, label="angular: training" if i == 0 else None)
    ax_val.scatter([size + 0.3] * dataset_size, qrl_results["approximation_ratios"][f"{size}"]["=angular"]["validation"], marker = "x",color='peru', alpha=0.3, label="angular: validation" if i == 0 else None)

# Set custom x-axis ticks
custom_ticks = [size for size in problem_sizes]  # Assuming problem_sizes is a list of x-axis tick values
ax_val.set_xticks(custom_ticks)
ax_val.set_xlabel("$Instances \, Size$", fontsize=15)



# Manually create legend
l1 = ax_val.legend(["sge-sgv: training", "sge-sgv: validation","sge-sgv+hea: training","sge-sgv+hea: validation"])
ax_val.set_ylim(0.4, 1.03)
ax_val.grid(which='major', alpha=0.4)
ax_val.hlines(y=1.0, xmin=3.5, xmax=10.5, linestyles='-', color='grey')

for item in l1.legendHandles:
    item._alpha = 1.0 #("black")

path = 'logs/rodrigo/5/2024-03-25--16-25-05_QRL_QPG'

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

label_names = [
    '$sge-sgv$',
    '$mge-mgv$',
    '$mge-sgv$',
    # '$sppgl+hea$',
    # '$hppgl+hea$',
    # '$mppgl+hea$'
]

colors = ['darkcyan','indianred', 'slategray', 'darkcyan', 'darkcyan', 'mediumseagreen', 'cornflowerblue']

trial_names = trial_names[:3]
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
        ax_5.plot(x_axis, mean, label=label_names[id], color=colors[id])
        ax_5.fill_between(x_axis, lower, upper, alpha=0.5, color=colors[id])
    else:
        print('One trial seems to be empty')


path = 'logs/rodrigo/5_hwe/2024-03-26--20-54-12_QRL_QPG'

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

label_names = [
    # '$sppgl$',
    # '$mppgl$',
    # '$hppgl$',
    '$sge-sgv+hea$',
    # '$hppgl+hea$',
    # '$mppgl+hea$'
]

colors = ['peru', 'mediumseagreen', 'cornflowerblue']

# trial_names = trial_names[:3]
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
        ax_5.plot(x_axis, mean, label=label_names[id], color=colors[id])
        ax_5.fill_between(x_axis, lower, upper, alpha=0.5, color=colors[id])
    else:
        print('One trial seems to be empty')


ax_5.hlines(y=1.0, xmin= 0, xmax=200000, linestyles='-', color='grey')

# axis.set_ylim(-30, 3)
ax_5.set_xlabel("$environment \, steps$", fontsize=15)
# ax_edge.set_ylabel('$approximation ratio$', fontsize=15)
# ax_5.set_ylim(0.75, 1.01)
ax_5.set_title('$5 \, Items$',fontsize=15)
ax_5.legend(fontsize=12, loc='lower right')
ax_5.minorticks_on()
ax_5.grid(which='major', alpha=0.4)




##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################


path = 'logs/rodrigo/10/2024-03-26--09-54-51_QRL_QPG'

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

label_names = [
    '$sge-sgv$',
    '$mge-mgv$',
    '$mge-sgv$',
    # '$sppgl+hea$',
    # '$hppgl+hea$',
    # '$mppgl+hea$'
]

colors = ['darkcyan', 'indianred', 'slategray',  'darkcyan', 'mediumseagreen', 'cornflowerblue']

trial_names = trial_names[:3]
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
        ax_10.plot(x_axis, mean, label=label_names[id], color=colors[id])
        ax_10.fill_between(x_axis, lower, upper, alpha=0.5, color=colors[id])
    else:
        print('One trial seems to be empty')

path = 'logs/rodrigo/10_hwe/2024-03-26--13-24-54_QRL_QPG'

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

label_names = [
    # '$sppgl$',
    # '$mppgl$',
    # '$hppgl$',
    '$sge-sgv+hea$',
    # '$hppgl+hea$',
    # '$mppgl+hea$'
]

colors = ['peru', 'mediumseagreen', 'cornflowerblue']

# trial_names = trial_names[:3]
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
        ax_10.plot(x_axis, mean, label=label_names[id], color=colors[id])
        ax_10.fill_between(x_axis, lower, upper, alpha=0.5, color=colors[id])
    else:
        print('One trial seems to be empty')


ax_10.hlines(y=1.0, xmin= 0, xmax=200000, linestyles='-', color='grey')

# axis.set_ylim(-30, 3)
ax_10.set_xlabel("$environment \, steps$", fontsize=15)
ax_5.set_ylabel('$approximation \, ratio$', fontsize=15)
ax_10.set_ylim(0.55, 1.03)
ax_5.set_ylim(0.55, 1.03)
ax_5.set_xticks([0, 50000, 100000, 150000, 200000])
ax_10.set_xticks([0, 50000, 100000, 150000, 200000])
ax_10.set_title('$10 \, Items$',fontsize=15)
ax_10.legend(fontsize=12, loc='lower right')
ax_10.minorticks_on()
ax_10.grid(which='major', alpha=0.4)


fig.tight_layout()
plt.savefig('knapsack_plot_1.pdf', dpi=600)
print('Done')
