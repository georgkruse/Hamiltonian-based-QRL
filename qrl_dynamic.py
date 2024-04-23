import numpy as np
from ray.rllib.policy.policy import Policy
from games.knapsack_wrapper import KNAPSACKSEQUENTIALDYNAMICVALIDATIONWRAPPER
import yaml
import numpy as np
import os
import math
from collections import Counter
import time
import multiprocessing as mp

class QRLDynamicTesting():
    """
    This class calculates the approximation ratios, probability of picking the optimal and valid actions for trained QRL agents.
    It also calculates how much better than random sampling it is to sample from each of the QRL agent's distributions.

    To calculate these metrics, the QRL agent is loaded from the checkpoint saved at the end of training and 100 episodes are
    simulated, where each action is accounted for. At the end, one has the probability of all the actions the QRL agent took and
    can use it to calculate the probability of taking the optimal action and the other relevant metrics. 

    Args:
    root_paths: list of paths to checkpoints of the trained QRL agents
    types: list of unique strings that distinguish each of the trained models
    instances_size: A list of ints indicating the size of the problem instances
    dataset_path: Path to the dataset containing each of the problem instances


    Returns:
    Saves a npy file corresponding to a dictionary containing all the relevant metrics.
    """
    def __init__(self,root_paths,types,instances_size,dataset_path):
        self.root_paths = root_paths
        self.types = types
        self.instances_size = instances_size
        self.dataset_path = dataset_path #"games/knapsack/KP_dataset.npy"
        self.dataset = np.load(self.dataset_path, allow_pickle=True).item()
        self.dataset_size = 100
        self.dataset_types = ["test", "validation"]
        
        self.results = {"approximation_ratios": {}, "optimal_probability": {}, "valid_probability":{}, "better_than_random": {}}

        for i,instance_size in enumerate(self.instances_size):
            self.results["approximation_ratios"][f"{instance_size}"] = {}
            self.results["optimal_probability"][f"{instance_size}"] = {}
            self.results["valid_probability"][f"{instance_size}"] = {}
            self.results["better_than_random"][f"{instance_size}"] = {}
            for type_ in self.types:
                self.results["approximation_ratios"][f"{instance_size}"][type_] = {}
                self.results["optimal_probability"][f"{instance_size}"][type_] = {}
                self.results["valid_probability"][f"{instance_size}"][type_] = {}
                self.results["better_than_random"][f"{instance_size}"][type_] = {}
                for dataset_type in self.dataset_types:
                    approximation_ratios, optimal_probability, valid_probability, better_than_random = self.calculate_approx_ratio(self.root_paths[i],instance_size,type_,dataset_type)
                    self.results["approximation_ratios"][f"{instance_size}"][type_][dataset_type] = approximation_ratios
                    self.results["optimal_probability"][f"{instance_size}"][type_][dataset_type] = optimal_probability
                    self.results["valid_probability"][f"{instance_size}"][type_][dataset_type] = valid_probability
                    self.results["better_than_random"][f"{instance_size}"][type_][dataset_type] = better_than_random

        np.save(f"final_results.npy", self.results)

    def calculate_approx_ratio(self,root_path,instance_size,type_, dataset_type):
        
        with open(root_path + '/alg_config.yml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        config["env_config"]["dataset_type"] = dataset_type
        config["algorithm_config"]["output_scaling_update"] = 0.0

        folders = []
        for root, dirs, files in os.walk(root_path):
            for dir_name in dirs:
                if type_ in dir_name:
                    folders.append(os.path.join(root,dir_name))

        checkpoints = []
        for folder in folders:
            for root, dirs, files in os.walk(folder):
                for dir_name in dirs:
                    if "checkpoint" in dir_name:
                        checkpoints.append(os.path.join(root,dir_name))

        checkpoints_approx_ratio = []
        checkpoints_optimal_probability = []
        checkpoints_valid_probability = []
        checkpoints_better_than_random = []


        for i in range(len(checkpoints)):
            start = time.time()
            optimal_probability, valid_probability, better_than_random,approx_ratio = self.calculate_probabilities(config,checkpoints[i])
            checkpoints_optimal_probability.append(optimal_probability)
            checkpoints_valid_probability.append(valid_probability)
            checkpoints_better_than_random.append(better_than_random)
            checkpoints_approx_ratio.append(approx_ratio)
            print(f"This checkpoint for instance size {instance_size} took {time.time()-start} seconds to be processed.")
            print("###############################################################################################")

        approx_ratio = np.mean(checkpoints_approx_ratio,axis = 0)
        optimal_probability = np.mean(checkpoints_optimal_probability, axis = 0)
        valid_probability = np.mean(checkpoints_valid_probability,axis = 0)
        better_than_random = np.mean(checkpoints_better_than_random,axis = 0)

        return approx_ratio, optimal_probability, valid_probability, better_than_random
    
    def calculate_probabilities(self,config,checkpoint):
        optimal_probability = np.zeros((100))
        better_than_random = np.zeros((100))
        valid_probability = np.zeros((100))
        average_random_contribution = np.zeros((100))
        qrl_contribution = np.zeros((100))
        approx_ratio = np.zeros((100))

        env = KNAPSACKSEQUENTIALDYNAMICVALIDATIONWRAPPER(config["env_config"])
        agent = Policy.from_checkpoint(checkpoint)['default_policy']
        agent.config["model"]["custom_model_config"]["output_scaling_update"] = 0.0

        for episode in range(self.dataset_size):
            approx_ratio_episode = []
            actions_taken = []
            optimal_action = env.env.solution_str[episode]
            optimal_action = list(optimal_action)
            optimal_action = [int(a) for a in optimal_action]
            optimal_reward = env.env.calculate_reward(optimal_action,timestep = episode)
            assert optimal_reward > 0
            for idx in range(100):
                theta, _ = env.reset(use_specific_timestep=True,timestep=episode)
                done = False
                decisions = [0 for i in range(env.env.instances_size)]
                while not done:
                    actions = agent.compute_single_action(theta)
                    action = actions[0]
                    theta, reward, done, _, _ = env.step(action)
                    if done == True:
                        actions_taken.append(decisions)
                        approx_ratio_episode.append(reward/optimal_reward)
                        break
                    else:
                        decisions[action] = 1
                
            approx_ratio[episode] = np.mean(approx_ratio_episode)
            
            counter = Counter(tuple(lst) for lst in actions_taken)
            action_probabilities = [(list(lst), count / 100) for lst, count in counter.items()]

            action_probabilities.sort(key=lambda x: x[1], reverse=True)
            random_probability = 1/(2**env.env.instances_size)
            actions = np.reshape(np.unpackbits(np.arange(2**env.env.instances_size).astype('>i8').view(np.uint8)), (-1, 64))[:,-env.env.instances_size:]
            for i,prob in enumerate(action_probabilities):
                # Store probability of optimal action
                if action_probabilities[i][0] == optimal_action:
                    optimal_probability[episode] = action_probabilities[i][1]
                contribution = env.env.calculate_reward(action_probabilities[i][0],timestep=episode)
                if not env.env.broke_constraint(action_probabilities[i][0],timestep=episode):
                    valid_probability[episode] += action_probabilities[i][1]
                    qrl_contribution[episode] += action_probabilities[i][1] * contribution
            for i, action in enumerate(actions):
                action_contrib = env.env.calculate_reward(action,timestep=episode)
                average_random_contribution[episode] += random_probability * action_contrib

        better_than_random = qrl_contribution/average_random_contribution
        return optimal_probability, valid_probability, better_than_random,approx_ratio
        

if __name__ == "__main__":
    
    paths = [
        "logs/knapsack/dynamic/final_results/init_constant/ising/3/2024-03-25--16-25-29_QRL_QPG",
        "logs/knapsack/dynamic/final_results/init_constant/ising/4/2024-03-25--16-25-19_QRL_QPG",
        "logs/knapsack/dynamic/final_results/init_constant/ising/5/2024-03-26--09-55-46_QRL_QPG",
        "logs/knapsack/dynamic/final_results/init_constant/ising/6/2024-03-26--09-55-45_QRL_QPG",
        "logs/knapsack/dynamic/final_results/init_constant/ising/7/2024-03-25--16-24-59_QRL_QPG",
        "logs/knapsack/dynamic/final_results/init_constant/ising/8/2024-03-26--09-55-15_QRL_QPG",
        "logs/knapsack/dynamic/final_results/init_constant/ising/9/2024-03-26--09-55-03_QRL_QPG",
        "logs/knapsack/dynamic/final_results/init_constant/ising/10/2024-03-26--09-54-51_QRL_QPG"
    ]

    types = [
        "=s-ppgl",
        "=m-ppgl",
        "=h-ppgl"
    ]

    instances_size = [3,4,5,6,7,8,9,10]


    QRLDynamicTesting(root_paths=paths,types=types,instances_size=instances_size)