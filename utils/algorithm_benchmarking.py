import os
import yaml
import torch
import shutil
import datetime
import argparse
import numpy as np
import pennylane as qml
from pennylane import qaoa
from ray.rllib.policy.policy import Policy
from torch.optim import Adam
import matplotlib.pyplot as plt

from utils.config.create_env import wrapper_switch
from utils.landscape_plotting import landscape_plotting
from circuits.vqe.vqe import circuit_vqe
from circuits.qaoa.qaoa import circuit_qaoa
from circuits.qnn.qnn import circuit_qnn
from circuits._custom.custom import custom_circuit

def algorithm_benchmarking(path):
    with open(path) as f:
        base_config = yaml.load(f, Loader=yaml.FullLoader)

    if base_config['evaluation']['set_seed']:
        np.random.seed(base_config['evaluation']['seed'])
        torch.manual_seed(base_config['evaluation']['seed'])

    config = base_config['evaluation']['algorithm_benchmarking']
    env = wrapper_switch[base_config['env']](base_config['env_config'])

    num_qubits = config['num_qubits']
    dev = qml.device("lightning.qubit", wires=num_qubits)
    alg_to_evaluate = config['alg_to_evaluate']
    alg_to_train = config['alg_to_train']
    optimizer_steps = config['optimizer_steps']
    layer_vqe = config['layer']['layer_vqe']
    layer_qaoa = config['layer']['layer_qaoa']
    layer_qnn = config['layer']['layer_qnn']
    plot_vqe_probs = config['plotting']['plot_vqe_probs']
    plot_qaoa_probs = config['plotting']['plot_qaoa_probs']
    plot_qrl_probs = config['plotting']['plot_qrl_probs']

    if isinstance(config['initialization']['init_vqe'], list):
        params_vqe = torch.tensor(config['initialization']['init_vqe'], requires_grad=True, dtype=torch.double)
    else:
        params_vqe = torch.tensor([config['initialization']['init_vqe'] for _ in range(config['params_vqe']*num_qubits*layer_vqe)], requires_grad=True, dtype=torch.double)

    if isinstance(config['initialization']['init_qaoa'], list):
        params_qaoa = torch.tensor(config['initialization']['init_qaoa'], requires_grad=True, dtype=torch.double)
    else:
        params_qaoa = torch.tensor([config['initialization']['init_qaoa'] for _ in range(2*layer_qaoa)], requires_grad=True, dtype=torch.double)

    if isinstance(config['initialization']['init_qnn'], list):
        params_qnn = torch.tensor(config['initialization']['init_qnn'], requires_grad=True, dtype=torch.double)
    else:
        params_qnn = torch.tensor([config['initialization']['init_qnn'] for _ in range(config['params_qnn']*num_qubits*layer_qnn)], requires_grad=True, dtype=torch.double)


    optim_vqe = Adam([params_vqe], lr=config['lr']['lr_vqe'])
    optim_qaoa = Adam([params_qaoa], lr=config['lr']['lr_vqe'])
    optim_qnn = Adam([params_qnn], lr=config['lr']['lr_vqe'])

    mode = base_config['env_config']['mode']
    date = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    folder = f'logs/{base_config["env"]}/{mode}/{date}'
    os.makedirs(os.path.dirname(folder + '/'), exist_ok=True)
    shutil.copy(path, folder + '/alg_config.yml')

    if 'vqe' in alg_to_evaluate or 'vqe' in alg_to_train:
        qnode_vqe_exp = qml.QNode(circuit_vqe, dev, interface='torch', diff_method='best') 
        qnode_vqe_probs = qml.QNode(circuit_vqe, dev, interface='torch') 

    if 'qaoa' in alg_to_evaluate or 'qaoa' in alg_to_train:
        qnode_qaoa_exp = qml.QNode(circuit_qaoa, dev, interface='torch', diff_method='best') 
        qnode_qaoa_probs = qml.QNode(circuit_qaoa, dev, interface='torch') 

        H_mixer =  qml.Hamiltonian(
            [1 for _ in range(num_qubits)],
            [qml.PauliX(i) for i in range(num_qubits)]
        )

    if 'qnn' in alg_to_evaluate or 'qnn' in alg_to_train:
        qnode_qnn = qml.QNode(circuit_qnn, dev, interface='torch', diff_method='adjoint') 

    if 'qrl' in alg_to_evaluate:
        # Not to be trained here, just evaluated against other algorithms in n episodes
        path = config['qrl_path']
        agent = Policy.from_checkpoint(path)['default_policy']  

    landscape_paths = []
    mean_final_loss_vqe = []
    mean_final_loss_qaoa = []
    mean_final_loss_qnn = []
    mean_final_loss_qrl = []

    found_optimal_vqe = 0
    found_optimal_qaoa = 0
    found_optimal_qnn = 0
    found_optimal_qrl = 0

    optim_cost_vqe = []
    optim_cost_qaoa = []
    optim_cost_qnn = []
    optim_cost_qrl = []
    optimal_cost = []

    theta, _ = env.reset()

    for timestep in range(config['evaluation_steps']):

        # Provide here the optimal scores (if possible)
        min_cost_index = env.calculate_optimal_actions(timestep)

        H = env.get_hamiltonian(timestep)
        linear, quadratic = env.get_ising(timestep)
        theta = env.get_theta(timestep)
    
        losses_vqe, losses_qaoa, losses_qnn = [], [], []
        if 'vqe' in alg_to_evaluate:
            if 'vqe' in alg_to_train:
                print('Started optimizing VQE')
                for i in range(optimizer_steps):
                    optim_vqe.zero_grad()
                    loss = qnode_vqe_exp(params_vqe, layer_vqe, num_qubits, H, 'exp')
                    loss.backward()
                    optim_vqe.step()
                    losses_vqe.append(loss.detach().numpy())
                    if i % 10 == 0:
                        print(loss)
                mean_final_loss_vqe.append(losses_vqe[-1])
                vqe_path = f'{folder}/params_vqe_qubits_{num_qubits}_layer_{layer_vqe}_timestep_{timestep}.csv'
                landscape_paths.append(vqe_path)
                np.savetxt(vqe_path, params_vqe.detach().numpy())
            else:
                params_vqe = torch.tensor(np.loadtxt(config['vqe_path']), requires_grad=True, dtype=torch.double)
            probs_vqe = qnode_vqe_probs(params_vqe, layer_vqe, num_qubits, H, 'probs')

        if 'qaoa' in alg_to_evaluate:
            if 'qaoa' in alg_to_train:
                print('Started optimizing QAOA')
                for i in range(optimizer_steps):
                    optim_qaoa.zero_grad()
                    loss = qnode_qaoa_exp(params_qaoa, layer_qaoa, num_qubits, H, H_mixer, 'exp')
                    loss.backward()
                    optim_qaoa.step()
                    losses_qaoa.append(loss.detach().numpy())
                    if i % 10 == 0:
                        print(loss)

                mean_final_loss_qaoa.append(losses_qaoa[-1])
                qaoa_path = f'{folder}/params_qaoa_qubits_{num_qubits}_layer_{layer_qaoa}_timestep_{timestep}.csv'
                landscape_paths.append(qaoa_path)
                np.savetxt(qaoa_path, params_qaoa.detach().numpy())
            else:
                params_qaoa = torch.tensor(np.loadtxt(config['qaoa_path']), requires_grad=True, dtype=torch.double)
            probs_qaoa = qnode_qaoa_probs(params_qaoa, layer_qaoa, num_qubits, H, H_mixer, 'probs')

        if 'qnn' in alg_to_train:
            print('Started optimizing QNN')
            for i in range(optimizer_steps):
                target = torch.from_numpy(env.optimal_actions[timestep])
                optim_qnn.zero_grad()
                prob = qnode_qnn(params_qnn, layer_qnn, num_qubits, linear, quadratic)
                prob = (prob + torch.ones(num_qubits))/2
                loss = torch.mean(torch.square(target - prob))
                loss.backward()
                optim_qnn.step()
                losses_qnn.append(loss.detach().numpy())
                if i % 10 == 0:
                    print(loss)
            
            action = np.concatenate([np.rint(prob.detach().numpy())]).astype(int)
            index = int(''.join(map(lambda action: str(int(action)), action)), 2)

            optim_cost_qnn.append(env.data[index])
            mean_final_loss_qnn.append(losses_qnn[-1])
            qnn_path = f'{folder}/params_qnn_qubits_{num_qubits}_layer_{layer_qnn}.csv'
            landscape_paths.append(qnn_path)
            np.savetxt(qnn_path, params_qnn.detach().numpy())

        if 'qrl' in alg_to_evaluate:
            print('Selected action QRL')
            # theta['linear_0'] = np.reshape(theta['linear_0'], (-1, *theta['linear_0'].shape))
            # theta['quadratic_0'] = np.reshape(theta['quadratic_0'], (-1, *theta['quadratic_0'].shape))
            action, x, y = agent.compute_single_action(theta)
            # action = agent.compute_actions(theta)
            theta, reward, done, _, _ = env.step(action)
            cost = env.actual_cost
            if cost == env.optimal_costs[timestep]:
                found_optimal_qrl += 1
            optim_cost_qrl.append(cost)

        if ('vqe' in alg_to_train) and ('qaoa' in alg_to_train):

            fig, ax = plt.subplots(layout='constrained')
            ax.plot(range(optimizer_steps), losses_vqe, label='VQE')
            ax.plot(range(optimizer_steps), losses_qaoa, label='QAOA')
            ax.set_ylabel('mean commitment cost')
            ax.set_xlabel('optimization step')
            ax.set_title(f'Comparision VQE and QAOA: UC {mode} timestep {timestep}')
            ax.legend()
            ax.grid(visible=True)
            fig.savefig(f'{folder}/comparision_vqe_qaoa_{mode}_timestep_{timestep}.png')
            fig.clf()
        
        if ('vqe' in alg_to_evaluate) and plot_vqe_probs:
            fig, ax = plt.subplots(layout='constrained')
            x = []
            width = 0.1  # the width of the bars
            multiplier = 0
            x_label = [format(i, f"0{num_qubits}b") for i in range(2**num_qubits)] 
            selected_x_label = []
            sub_one_percent_costs_vqe = []
            for idx, value in enumerate(probs_vqe):
                if value >= 0.01:
                    offset = width * multiplier
                    if idx == min_cost_index:
                        color = 'green'
                        found_optimal_vqe += 1
                    else:
                        color = 'blue'
                    rects = ax.bar(offset, value.detach().numpy(), width, color=color)
                    ax.bar_label(rects, padding=3)
                    multiplier += 1
                    x.append(offset)
                    selected_x_label.append(x_label[idx])
                    sub_one_percent_costs_vqe.append(env.data[idx])

            if len(sub_one_percent_costs_vqe) >= 1:
                optim_cost_vqe.append(min(sub_one_percent_costs_vqe))
            else:
                optim_cost_vqe.append(0.0)

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel('probability')
            ax.set_xlabel('binary label')
            ax.set_title(f'Probabilities VQE: UC {mode} timestep {timestep}')
            ax.set_xticks(x, selected_x_label, rotation=45)
            fig.savefig(f'{folder}/uc_vqe_comp_{mode}_timestep_{timestep}.png')
            fig.clf()

        if ('qaoa' in alg_to_evaluate) and plot_qaoa_probs:

            fig, ax = plt.subplots(layout='constrained')
            x = []
            width = 0.1  # the width of the bars
            multiplier = 0
            x_label = [format(i, f"0{num_qubits}b") for i in range(2**num_qubits)] 
            selected_x_label = []
            sub_one_percent_costs_qaoa = []
            for idx, value in enumerate(probs_qaoa):
                if value >= 0.01:
                    offset = width * multiplier
                    if idx == min_cost_index:
                        color = 'green'
                        found_optimal_qaoa += 1
                    else:
                        color = 'blue'
                    rects = ax.bar(offset, value.detach().numpy(), width, color=color)
                    ax.bar_label(rects, padding=3)
                    multiplier += 1
                    x.append(offset)
                    selected_x_label.append(x_label[idx])
                    sub_one_percent_costs_qaoa.append(env.data[idx])

            if len(sub_one_percent_costs_qaoa) >= 1:
                optim_cost_qaoa.append(min(sub_one_percent_costs_qaoa))
            else:
                optim_cost_qaoa.append(0.0)

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel('probability')
            ax.set_xlabel('binary label')
            ax.set_title(f'Probabilities QAOA: UC {mode} timestep {timestep}')
            ax.set_xticks(x, selected_x_label, rotation=45)
            fig.savefig(f'{folder}/uc_qaoa_comp_{mode}_timestep_{timestep}.png')
            fig.clf()

    fig, ax1 = plt.subplots(layout='constrained')
    # ax2 = ax1.twinx()
    x = []
    width = 0.1  # the width of the bars
    multiplier = 0

    if ('vqe' in alg_to_evaluate) and plot_vqe_probs:
        # ax1.scatter(range(config['evaluation_steps']), mean_final_loss_vqe, label='VQE')
        ax1.scatter([*range(config['evaluation_steps'])], optim_cost_vqe, marker='s', label='VQE')

    if ('qaoa' in alg_to_evaluate) and plot_qaoa_probs:
        # ax1.scatter(range(config['evaluation_steps']), mean_final_loss_qaoa, label='QAOA')
        ax1.scatter([*range(config['evaluation_steps'])], optim_cost_qaoa, marker='s', label='QAOA')

    if ('qrl' in alg_to_evaluate) and plot_qrl_probs:
        ax1.scatter([*range(config['evaluation_steps'])], optim_cost_qrl, label='QRL')
    # if ('qnn' in alg_to_evaluate) and plot_qnn_probs:
    #     ax2.scatter([*range(config['evaluation_steps'])], optim_cost_qnn, label='QNN')
        
    ax1.plot([*range(config['evaluation_steps'])], env.optimal_costs[:config['evaluation_steps']], label='optimal', color='black')
    ax1.grid(visible=True)

    ax1.set_ylabel('mean commitment cost')
    # ax2.set_ylabel('minimal commitment cost')
    ax1.set_xlabel('timestep')
    ax1.set_title(f'UC {mode} all timesteps \n found optimal: VQE {found_optimal_vqe} ; QAOA {found_optimal_qaoa} ; QRL {found_optimal_qrl}')
    ax1.legend()
    # ax2.legend()
    fig.savefig(f'{folder}/comparision_vqe_qaoa_{mode}_all_timesteps.png')

    print('Done')

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default= 'configs/_eva.yml', 
                        metavar="FILE", help="path to config file", type=str)
    args = parser.parse_args()
    algorithm_benchmarking(args.path)