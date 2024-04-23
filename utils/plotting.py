import yaml
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import itertools
from utils.config.common import extract_hyperparameters

def plotting(path, config=None):
    
    if config.evaluation['plotting']['mode'] == 'custom':
        path = config.evaluation['plotting']['path']

    if 'y_axis' in config.evaluation['plotting'].keys():
        key_y_axis = config.evaluation['plotting']['y_axis']
        y_axis_label = f"${config.evaluation['plotting']['y_axis']}$"

    else:    
        key_y_axis = 'episode_reward_mean'
        y_axis_label = '$reward$'

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
    fig, ax_main = plt.subplots(1, figsize=(8, 6))

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
                # if key_y_axis in data_exp.keys():
                if key_y_axis == 'approximation_ratio':
                    cust_ = []
                    for idx in range(data_exp['custom_metrics'].shape[0]):
                        cust_.append(np.mean(data_exp['custom_metrics'].values[idx]['approximation_ratio']))
                    curves[int(key)].append(cust_)
                else:
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
            ax_main.plot(x_axis, mean, label=labels[id])
            ax_main.fill_between(x_axis, lower, upper, alpha=0.5)
        else:
            print('One trial seems to be empty')

    # ax_main.hlines(y=0, xmin= 0, xmax=100000, linestyles='dashdot', color='black')
    # axis.set_ylim(-30, 3)
    ax_main.set_xlabel("$environment \, steps$", fontsize=13)
    ax_main.set_ylabel(y_axis_label, fontsize=15)
    # ax_main.set_xlim(0, 100000)
    ax_main.set_title(title, fontsize=15)
    ax_main.legend(fontsize=12, loc='lower right')
    ax_main.minorticks_on()
    ax_main.grid(which='both', alpha=0.4)
    plt.text(1.01, 0.6, info_text, transform=ax_main.transAxes, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
    fig.tight_layout()
    plt.savefig(f'{path}/{fig_name}.png') #, dpi=1200)
    
    
    # key_combinations = list(itertools.combinations(key_names, 2)) 

    for j in range(len(single_elements)):
        tmp_trial_names, tmp_label_names = [], []

        if type(single_elements[j][1]) == int:
            main_element = f'{single_elements[j][0]}={single_elements[j][1]},'
        elif type(single_elements[j][1]) == float:
            main_element = f'{single_elements[j][0]}={single_elements[j][1]:.4f},'
        else:
            main_element = f'{single_elements[j][0]}={single_elements[j][1]},'


        for idx, name in enumerate(trial_names):
            if main_element in name:
                tmp_trial_names.append(name)
                tmp_label_names.append(label_names[idx])

        # Plot all runs in one plot
        fig, ax_main = plt.subplots(1, figsize=(8, 6))

        info_text = []
        if 'lr' not in key_names:
            info_text.append(f"lr: {config['algorithm_config']['lr']} \n")
        if 'lr_output_scaling' not in key_names:
            info_text.append(f"lr_out: {config['algorithm_config']['lr_output_scaling']} \n")
        if 'num_layers' not in key_names:
            info_text.append(f"num_layers: {config['algorithm_config']['num_layers']} \n")

        info_text = ''.join(info_text)

        fig_name = f"{config['env']}, {folder_path}"
        title = f'{fig_name} \n constant {single_elements[j]}'

        types = {} 
        labels = []
        for i, name in enumerate(tmp_trial_names):
            types[str(i)] = name
            labels.append(tmp_label_names[i])

        curves = [[] for _ in range(i+1)]
        curves_x_axis = [[] for _ in range(i+1)]

        results_file_name = "/result.json"
        result_files  = [f.path for f in os.scandir(path) if f.is_dir() ]
        results = [pd.read_json(f + results_file_name,lines=True) for f in result_files]

        for key, value in types.items():
            for i, data_exp in enumerate(results):
                if all(x in result_files[i] for x in value):
                    # if key_y_axis in data_exp.keys():
                    if key_y_axis == 'approximation_ratio':
                        cust_ = []
                        for idx in range(data_exp['custom_metrics'].shape[0]):
                            cust_.append(np.mean(data_exp['custom_metrics'].values[idx]['approximation_ratio']))
                        curves[int(key)].append(cust_)
                    else:
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
                ax_main.plot(x_axis, mean, label=labels[id])
                ax_main.fill_between(x_axis, lower, upper, alpha=0.5)

        # ax_main.hlines(y=0, xmin= 0, xmax=100000, linestyles='dashdot', color='black')
        # axis.set_ylim(-30, 3)
        ax_main.set_xlabel("$environment \, steps$", fontsize=13)
        ax_main.set_ylabel(y_axis_label, fontsize=15)
        # ax_main.set_xlim(0, 100000)
        ax_main.set_title(title, fontsize=15)
        ax_main.legend(fontsize=12, loc='lower right')
        ax_main.minorticks_on()
        ax_main.grid(which='both', alpha=0.4)
        plt.text(1.01, 0.6, info_text, transform=ax_main.transAxes, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
        fig.tight_layout()
        plt.savefig(f'{path}/{fig_name}_{single_elements[j]}.png') #, dpi=1200)

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default='logs/qppo/uc/02_07/2024-02-07--10-18-17_QRL_QPPO', 
                        metavar="FILE", help="path to config file", type=str)
    args = parser.parse_args()
    plotting(args.path)



