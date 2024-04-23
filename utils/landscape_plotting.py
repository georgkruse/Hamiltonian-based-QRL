"""
File that generates an energy landscape with random directions in the space of 
optimization.
"""
import os
import ray
import yaml
import torch
import argparse
from numpy import dot
from numpy.linalg import norm
import pennylane as qml
import numpy as np
from pyqubo import Binary
import pandas as pd
from pennylane import qaoa
from pennylane import numpy as qnp
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import minimize

from ray.util.multiprocessing import Pool
from ray.rllib.policy.policy import Policy
from circuits.quantum_circuits import vqc_switch
from utils.config.create_env import wrapper_switch

from circuits.vqe.vqe import circuit_vqe
from circuits.qaoa.qaoa import circuit_qaoa
from circuits.qnn.qnn import circuit_qnn
from circuits._custom.custom import custom_circuit


def landscape_plotting(path):

    # Load config and specify some details
    with open(path) as f:
        base_config = yaml.load(f, Loader=yaml.FullLoader)
    config = base_config['evaluation']['landscape_plotting']

    num_qubits = config['num_qubits']
    num_layers = config['num_layers']
    evaluation_steps = config['evaluation_steps']

    if config['mode'] in ['qrl', 'qnn']:
        with open(f"{config['qrl_config_path']}/alg_config.yml") as f:
            _base_config = yaml.load(f, Loader=yaml.FullLoader)
        algorithm_config = _base_config['algorithm_config']
        algorithm_config['num_qubits'] = algorithm_config['vqc_type'][1]
        dev = qml.device(algorithm_config['backend_name'], wires=num_qubits)
    else:
        dev = qml.device('lightning.qubit', wires=num_qubits)

    env = wrapper_switch[base_config['env']](base_config['env_config'])

    theta, _ = env.reset()
    # Do some reshaping due to batch dimension (not always required, depends on vqc structure)
    theta['linear_0'] = np.reshape(theta['linear_0'], (-1, *theta['linear_0'].shape))
    theta['quadratic_0'] = np.reshape(theta['quadratic_0'], (-1, *theta['quadratic_0'].shape))

    # Get hamiltonian terms (at timestep int)
    H = env.get_hamiltonian(0)
    ray.init(num_cpus=base_config['total_num_cpus'], 
             local_mode=base_config['ray_local_mode'],
             _temp_dir=os.path.dirname(os.path.dirname(os.getcwd())) + '/' + 'ray_logs')


    # np.random.seed(694)
    fig_name = config['plot_name']
    fig_title =  config['plot_title']
    
    num_div = config['num_div'] # Number of divisions to search (granularity of the grid / number of data points)
    
    if config['mode'] == 'qaoa':
        num_params = config['num_layers']*config['num_params']
    elif config['mode'] == 'vqe':
        num_params = config['num_layers']*config['num_params']*num_qubits
    elif config['mode'] == 'qrl':
        agent = Policy.from_checkpoint(config['center_params_path'])['default_policy']     
        num_params = algorithm_config['num_layers']*config['num_params']*num_qubits
    # Size of the surrounding of the center parameter
    number_dimension = num_params
    if number_dimension == 2:
        scalor_1 = config['scalor_1']
        scalor_2 = config['scalor_2']
    else:
        scalor_1 = scalor_2 = config['scalor_1']
    # Initialization of the grid
    arr_alpha1 = np.reshape(np.linspace(-scalor_1, scalor_1, num_div),(-1,1))
    arr_alpha2 = np.reshape(np.linspace(-scalor_2, scalor_2, num_div),(-1,1))
    len_alpha1= len(arr_alpha1)
    len_alpha2= len(arr_alpha2)

    # Create coordinate-grid
    param_grid_array = []
    for i, alpha1 in enumerate(arr_alpha1):
        for j, alpha2 in enumerate(arr_alpha2):
            param_grid_array.append([alpha1[0], alpha2[0]])
    param_grid = np.reshape(param_grid_array, (num_div*num_div, -1))

    # Generate random vectors to span the space (necessary if number of dimensions>2)
    # Cosine similarity of the vectors should be close to 0
    cos_sim = 1.
    while np.abs(cos_sim) >= 0.15:
        vec_delta = np.random.normal(size=(num_params))
        vec_eta = np.random.normal(size=(num_params)) 
        cos_sim = dot(vec_delta, vec_eta)/(norm(vec_delta)*norm(vec_eta))
        print('cosine similarity is:', cos_sim)

    vec_delta = np.reshape(vec_delta, (1,-1))
    vec_eta = np.reshape(vec_eta, (1,-1))
    var_params = np.empty((0,num_params))

    
    if isinstance(config['center_params'], float):
        center_params = np.zeros((len(arr_alpha1), num_params))
        scaling = np.ones(num_qubits)
    elif isinstance(config['center_params'], list):
        center_params = np.ones((len(arr_alpha1), num_params))*np.array(*config['center_params'])
        scaling = np.ones(num_qubits)
    elif isinstance(config['center_params'], str):
        csv_file_path = config['center_params_path']
        if config['mode'] == 'qrl':
            center_params = agent.model.variables()
            scaling = center_params[-1]
            # center_params = torch.concatenate([torch.concatenate([center_params[i-1][:5], center_params[i][:15]]) for i in range(1,6,2)]).detach().numpy()
            center_params = torch.concatenate([torch.concatenate([center_params[i][:2]]) for i in range(1,6,2)]).detach().numpy()
        # For QAOA, VQE or QNN:
        elif config['mode'] in ['qaoa', 'vqe', 'qnn']:
            center_params = np.loadtxt(csv_file_path)
    
    # Create weight-grid 
    if number_dimension == 2:
        var_params = []
        for i, alpha1 in enumerate(arr_alpha1):
            for j, alpha2 in enumerate(arr_alpha2):
                var_params.append([alpha1[0]+center_params[0], center_params[1]+alpha2[0]])
        var_params = np.reshape(var_params, (num_div*num_div, -1))
    else:
        for col_alpha2 in arr_alpha2:
            mat_alpha2 = np.ones((len_alpha1,1))*col_alpha2[0]
            var_params = np.concatenate((var_params,center_params+vec_delta*arr_alpha1+vec_eta*mat_alpha2))


    if config['mode'] == 'qrl':
        ######################################################
        # QRL
        ######################################################
        # qnode = qml.QNode(custom_circuit, dev, interface='torch')     
        qnode = qml.QNode(vqc_switch[config['vqc_type'][0]], dev, interface=config['interface'], diff_method=config['diff_method']) #, argnum=0)

        @ray.remote(num_cpus=1)
        def evaluate(theta, params, scaling):
            accuracy = 0
            total_reward = 0
            done = False
            theta, _ = env.reset()

            for timestep in range(evaluation_steps):
                theta['linear_0'] = np.reshape(theta['linear_0'], (-1, *theta['linear_0'].shape))
                theta['quadratic_0'] = np.reshape(theta['quadratic_0'], (-1, *theta['quadratic_0'].shape))
                prob = qnode(theta, params, config, H)
                prob = (prob + torch.ones(num_qubits))/2
                prob = torch.reshape(prob, (-1, num_qubits)).repeat_interleave(2, dim=1)*torch.tensor([1., -1.]).repeat(num_qubits)
                prob = prob + torch.tensor([0., 1.]).repeat(num_qubits)
                prob = prob * scaling
                action = torch.argmax(torch.reshape(prob, (-1, 2)), axis=1).detach().numpy()                
                theta, reward, done, _, _ = env.step(action)
                target = env.env.optimal_actions[timestep]
                total_reward += reward
                accuracy += int(all(action == target))
                if done:
                    theta, _ = env.reset()
                    done = False

            if config['metric'] == 'accuracy':
                return accuracy
            elif config['metric'] == 'reward':
                return total_reward
            
        metric = config['metric']
        futures = [evaluate.remote(theta, params, scaling) for params in var_params]

    elif config['mode'] == 'qaoa':
        ######################################################
        # QAOA 
        ######################################################

        H_mixer =  qml.Hamiltonian(
            [1 for _ in range(num_qubits)],
            [qml.PauliX(i) for i in range(num_qubits)]
        )
        qnode = qml.QNode(circuit_qaoa, dev)  

        @ray.remote(num_cpus=1)
        def evaluate_qaoa(params, H):
            output = qnode(params, num_layers, num_qubits, H, H_mixer, 'exp')
            return output
        
        metric = 'energy'
        futures = [evaluate_qaoa.remote(params, H) for params in var_params]
    
    elif config['mode'] == 'vqe':
        ######################################################
        # VQE 
        ######################################################
        qnode = qml.QNode(circuit_vqe, dev)  

        @ray.remote(num_cpus=1)
        def evaluate_vqe(params, H):
            output = qnode(params, num_layers, num_qubits, H, 'exp')
            return output
        metric = 'energy'
        futures = [evaluate_vqe.remote(params, H) for params in var_params]

    elif config['mode'] == 'qnn':
        ######################################################
        # QNN
        ######################################################
       
        qnode = qml.QNode(circuit_qnn, dev)  
        
        # create test dataset for qnn
        test_dataset = []
        for _ in range(evaluation_steps):
            theta, _ = env.reset()
            for timestep in range(env.episode_length):
                linear, quadratic = env.get_ising(timestep)
                env.calculate_optimal_actions(timestep)
                target = env.optimal_actions[timestep]
                test_dataset.append([linear, quadratic, target])

        @ray.remote(num_cpus=1)
        def evaluate(params):
            accuracy = 0
            # Test accuracy
            for _ in range(evaluation_steps):
                idx = np.random.randint(0, len(test_dataset))
                sample = test_dataset[idx]
                linear, quadratic, target = sample
                prob = qnode(params, num_layers, num_qubits, linear, quadratic)
                prob = (prob + np.ones(num_qubits))/2
                action = np.concatenate([np.rint(prob)]).astype(int)
                accuracy += int(all(action == target))
            return accuracy
        metric = 'accuracy'
        futures = [evaluate.remote(params) for params in var_params]

    data = np.reshape(ray.get(futures), (-1))

    if config['save_to_json']:
        # If calculation takes long, also save the result to a json file
        df_res = pd.DataFrame(columns= ['alpha1', 'alpha2', 'vec_delta', 'vec_eta', 'output'])
        for idx, output in enumerate(data):
            df_res.loc[len(df_res)] = [param_grid_array[idx][0], 
                                       param_grid_array[idx][1], 
                                       var_params[idx][0], 
                                       var_params[idx][1], 
                                       output]
        df_res.to_json(f"{config['plot_path']}/surface_plot.json")


    fig, ax = plt.subplots(nrows=1, subplot_kw={'projection': '3d'},
                            figsize=(12, 8.5))
    # Use the non-transformed grid for plotting
    ax.plot_trisurf(param_grid[:,0], param_grid[:,1], data, cmap=cm.viridis)
    if config['mark_optimal'][0]:
        ax.scatter(param_grid[np.where(data == config['mark_optimal'][1]), 0],
                   param_grid[np.where(data == config['mark_optimal'][1]), 1],
                   data[np.where(data == config['mark_optimal'][1])], c='red')
    ax.set_title(f'{fig_title} \n {metric}')
    ax.set_xlabel("$\\alpha_1$")
    ax.set_ylabel("$\\alpha_2$")
    ax.set_zlabel("$C(\\alpha_1,\\alpha_2)$")
    ax.set_proj_type('ortho')
    ax.view_init(elev=60)
    fig.tight_layout()
    fig.savefig(f"{config['plot_path']}/{fig_name}.png")
    ax.view_init(elev=90)
    fig.savefig(f"{config['plot_path']}/{fig_name}_rot.png")
    plt.show()
    print('Done')


if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default= 'configs/qpg/qpg_fp_gk.yml', 
                        metavar="FILE", help="path to config file", type=str)
    args = parser.parse_args()
    landscape_plotting(args.path)

