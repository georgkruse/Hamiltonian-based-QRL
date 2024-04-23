import numpy as np
import pennylane as qml
import time 
import gymnasium as gym 
import math
import argparse
import itertools
# import pandas as pd
import os, yaml
import numpy as np
# from PIL import Image
# import PIL.ImageDraw as ImageDraw
import matplotlib.pyplot as plt    
import ray
from ray import tune, air
import os
import pandas as pd
from collections import namedtuple
import shutil
from utils.config.create_config import add_hyperparameters, extract_hyperparameters
import datetime
import torch
import pennylane as qml
from pennylane import numpy as pnp
from circuits.quantum_circuits import vqc_switch
from utils.config.create_env import wrapper_switch

def variance_calculation(path):
    # Load config and specify some details
    with open(path) as f:
        base_config = yaml.load(f, Loader=yaml.FullLoader)

    algorithm_config = base_config['algorithm_config']
    algorithm_config['measurement_type_actor'] = base_config['evaluation']['variance_calculation']['measurement_type_actor']
    config = base_config['evaluation']['variance_calculation']
    qubits =  config['qubits']
    
    if base_config['evaluation']['set_seed']:
        np.random.seed(seed=base_config['evaluation']['seed'])
        
    def init_weights(size):
        return pnp.random.uniform(0, 2*np.pi, size)

    def calculate(algorithm_config):

        info = {'var_all_gradients': []}
        info['var_expecation'] = []
        for i in range(algorithm_config['num_layers']):
            info[f'var_weights_actor_{i}'] = []
            info[f'var_input_scaling_actor_{i}'] = []
        info[f'var_weights_0'] = []
        info[f'var_input_scaling_0'] = []
        info[f'var_weights_L-mid'] = []
        info[f'var_input_scaling_L-mid'] = []
        info[f'var_weights_L'] = []
        info[f'var_input_scaling_L'] = []

        for num_qubits in qubits:

            algorithm_config['num_qubits'] = num_qubits
            dev = qml.device('lightning.qubit', wires=num_qubits)
            qnode = qml.QNode(vqc_switch[algorithm_config['vqc_type'][0]], dev, diff_method='adjoint') #, argnum=0)

            if base_config['env'] == 'UC':
                base_config['env_config']['num_generators'] = num_qubits
            # env = wrapper_switch[base_config['env']](base_config['env_config'])

            if algorithm_config['encoding_type'] == 'graph_encoding':
                if algorithm_config['graph_encoding_type'] in ['s-ppgl', 's-ppgl-linear', 's-ppgl-quadratic']:
                    size_vqc = 1
                    size_input_scaling = 1
                elif algorithm_config['graph_encoding_type'] == 'm-ppgl':
                    size_vqc = num_qubits
                    size_input_scaling = sum(range(num_qubits+1))+num_qubits
                elif algorithm_config['graph_encoding_type'] == 'm-ppgl-linear':
                    size_vqc = num_qubits
                    size_input_scaling = num_qubits
                elif algorithm_config['graph_encoding_type'] == 'm-ppgl-quadratic':
                    size_vqc = num_qubits
                    size_input_scaling = sum(range(num_qubits+1))
                elif algorithm_config['graph_encoding_type'] == 'h-ppgl':
                    size_vqc = 1
                    size_input_scaling = sum(range(num_qubits+1))+num_qubits
                elif algorithm_config['graph_encoding_type'] == 'h-ppgl-linear':
                    size_vqc = 1
                    size_input_scaling = num_qubits + 1
                elif algorithm_config['graph_encoding_type'] == 'h-ppgl-quadratic':
                    size_vqc = 1
                    size_input_scaling = sum(range(num_qubits+1)) + 1
                elif algorithm_config['graph_encoding_type'] in ['angular', 'angular-hwe']:
                    size_vqc = num_qubits*algorithm_config['num_variational_params']
                    size_input_scaling = num_qubits*algorithm_config['num_scaling_params']
                elif algorithm_config['graph_encoding_type'] == 'hamiltonian-hwe':
                    size_vqc = num_qubits*algorithm_config['num_variational_params']
                    size_input_scaling = 0
                if algorithm_config['block_sequence'] in ['enc_var_ent', 'enc_var', 'enc_ent_var']:
                    size_vqc += num_qubits*algorithm_config['num_variational_params']
            else:
                size_vqc = num_qubits*algorithm_config['num_variational_params']
                size_input_scaling = num_qubits*algorithm_config['num_scaling_params']      

            # theta, _ = env.reset()
            # theta['linear_0'] = np.reshape(theta['linear_0'], (-1, *theta['linear_0'].shape))
            # theta['quadratic_0'] = np.reshape(theta['quadratic_0'], (-1, *theta['quadratic_0'].shape))
            
            theta = {}
            linear, quadratic = [], []
            for i in range(num_qubits):
                linear.append([i, pnp.random.uniform(0, 2*np.pi)])
                for j in range(num_qubits):
                    if i < j:
                        quadratic.append([i, j,  pnp.random.uniform(0, 2*np.pi)])

            theta['linear_0'] = np.reshape(np.stack(linear), (-1, num_qubits, 2))
            theta['quadratic_0'] = np.reshape(np.stack(quadratic), (-1, sum(range(num_qubits)), 3))

            tmp = {'var_all_gradients': np.empty((1,))}
            tmp['var_expecation'] = []
            for i in range(algorithm_config['num_layers']):
                tmp[f'var_weights_actor_{i}'] = []
                tmp[f'var_input_scaling_actor_{i}'] = []
                
            tmp[f'var_weights_0'] = []
            tmp[f'var_input_scaling_0'] = []
            tmp[f'var_weights_L-mid'] = []
            tmp[f'var_input_scaling_L-mid'] = []
            tmp[f'var_weights_L'] = []
            tmp[f'var_input_scaling_L'] = []

            for i in range(config['evaluations']):
                weights = {}
                weights[f'weights_actor'] = init_weights((algorithm_config['num_layers'], size_vqc))
                weights[f'input_scaling_actor'] = init_weights((algorithm_config['num_layers'], size_input_scaling))

                probs = qnode(theta=theta, weights=weights, config=algorithm_config, type='actor', activations=None, H=None)
                grad = qml.grad(qnode, argnum=0)
                gradient = grad(weights, theta=theta, config=algorithm_config, type='actor', activations=None, H=None)             

                tmp['var_expecation'].append(probs)
                for i in range(algorithm_config['num_layers']):
                    tmp[f'var_weights_actor_{i}'].append(gradient[f'weights_actor'][i, 0])
                    tmp[f'var_input_scaling_actor_{i}'].append(gradient[f'input_scaling_actor'][i, 0])
                tmp['var_all_gradients'] = gradient[f'weights_actor']
                tmp['var_all_gradients'] = gradient[f'input_scaling_actor']
                tmp[f'var_weights_0'].append(gradient[f'weights_actor'][0,0])
                tmp[f'var_input_scaling_0'].append(gradient[f'input_scaling_actor'][0,0])
                tmp[f'var_weights_L-mid'].append(gradient[f'weights_actor'][math.ceil(i/2),0])
                tmp[f'var_input_scaling_L-mid'].append(gradient[f'input_scaling_actor'][math.ceil(i/2),0])
                tmp[f'var_weights_L'].append(gradient[f'weights_actor'][-1,0])
                tmp[f'var_input_scaling_L'].append(gradient[f'input_scaling_actor'][-1,0])
            
            info['var_all_gradients'].append(np.var(tmp['var_all_gradients']))
            info['var_expecation'].append(np.var(tmp['var_expecation']))
            for i in range(algorithm_config['num_layers']):
                info[f'var_weights_actor_{i}'].append(np.var(tmp[f'var_weights_actor_{i}']))
                info[f'var_input_scaling_actor_{i}'].append(np.var(tmp[f'var_input_scaling_actor_{i}']))
            info[f'var_weights_0'].append(np.var(tmp[f'var_weights_0']))
            info[f'var_input_scaling_0'].append(np.var(tmp[f'var_input_scaling_0']))
            info[f'var_weights_L-mid'].append(np.var(tmp[f'var_weights_L-mid']))
            info[f'var_input_scaling_L-mid'].append(np.var(tmp[f'var_input_scaling_L-mid']))
            info[f'var_weights_L'].append(np.var(tmp[f'var_weights_L']))
            info[f'var_input_scaling_L'].append(np.var(tmp[f'var_input_scaling_L']))
            print(num_qubits)       

        return info
    

    if config['calculate_vars']:
        ray.init(local_mode=base_config['ray_local_mode'], _temp_dir=os.path.dirname(os.path.dirname(os.getcwd())) + '/' + 'ray_logs')

        param_space = add_hyperparameters(algorithm_config)
                
        # Create the name tags for the trials and log directories
        name = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S") + '_' + 'variances'
        ray_path = os.getcwd() + '/' + config['logging_path']
        path_ = ray_path + '/' + name

        # Copy the config files into the ray-run folder
        os.makedirs(os.path.dirname(path_ + '/'), exist_ok=True)
        shutil.copy(path, path_ + '/alg_config.yml')

        tuner = tune.Tuner(
            calculate,
            param_space=param_space,
            run_config=air.RunConfig(stop={"training_iteration": 1}, name=name, local_dir=ray_path), 
        )
        tuner.fit()
        path = path_

    else:
        path = config['path']

    
    if config['plot_vars']:

        with open(path + '/alg_config.yml') as f:
            base_config = yaml.load(f, Loader=yaml.FullLoader)  
        
        for plot_key in config['plot_keys']:
            fig, ax_main = plt.subplots(1, figsize=(8, 6)) 
            fig_name = config['fig_name']

            title = f'{fig_name}' 
            key_names, hyperparameters, single_elements = extract_hyperparameters(base_config['algorithm_config'])

            combinations = list(itertools.product(*hyperparameters))
            trial_names, label_names = [], []
            for comb in combinations:
                trial_names.append([f'{name}={str(element)}_' for (name, element) in zip(key_names, comb)])
                label_names.append([str(element) for element in comb])

        
            types = {} 
            labels = []
            for i, name in enumerate(trial_names):
                types[str(i)] = name
                labels.append(label_names[i])

            plot_keys = config['plot_keys']

            results_file_name = "/result.json"
            result_files  = [f.path for f in os.scandir(path) if f.is_dir() ]
            results = [pd.read_json(f + results_file_name,lines=True) for f in result_files]

            for key, value in types.items():
                for i, file in enumerate(result_files):
                # for i, data_exp in enumerate(results):
                    if all(x in file for x in value):
                        data_exp = pd.read_json(file + results_file_name,lines=True)
                        ax_main.plot(qubits, data_exp[plot_key].values[0], label=f'{labels[int(key)][0]} - {plot_key}')
                        # ax_main.plot(qubits, data_exp[plot_key].values[0], label=f'hea - {plot_key}')

            ax_main.set_xlabel("$qubits$", fontsize=13)
            ax_main.set_ylabel("$variance$", fontsize=15)
            ax_main.set_yscale('log')
            ax_main.set_ylim(1e-5)
            ax_main.set_title(title + ' ' + plot_key, fontsize=15)
            ax_main.legend(fontsize=12, loc='lower left')
            ax_main.minorticks_on()
            ax_main.grid(which='both', alpha=0.4)
            fig.tight_layout()
            plt.savefig(f'{path}/{fig_name}_{plot_key}.png')

    print('Done')

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default= 'configs/qpg/qpg_fp_gk.yml', 
                        metavar="FILE", help="path to config file", type=str)
    args = parser.parse_args()
    variance_calculation(args.path)



