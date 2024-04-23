from ray.rllib.policy.policy import Policy
from games.knapsack_wrapper import KNAPSACKWRAPPER
import yaml
import numpy as np
import itertools
import os
import matplotlib.pyplot as plt

root_paths = [
    "logs/knapsack/3items/2024-03-06--12-15-39_QRL_QPG",
    "logs/knapsack/4items/2024-03-06--12-15-37_QRL_QPG",
    "logs/knapsack/5items/2024-03-06--12-15-22_QRL_QPG",
    "logs/knapsack/6items/2024-03-06--12-15-05_QRL_QPG",
    "logs/knapsack/7items/2024-03-06--12-13-43_QRL_QPG"
]

types = ["KNAPSACK"]
results_optimal = []
results_valid = []
results_better_than_average = []

for root_path in root_paths:

    aux = []
    percentages_optimal = []
    percentages_valid = []
    percentages_better_than_average = []

    folders = []

    with open(root_path + '/alg_config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for type in types:
        percentages_type_optimal = []
        percentages_type_valid = []
        percentages_type_better_than_average = []
        folders = []
        for root, dirs, files in os.walk(root_path):
            for dir_name in dirs:
                if type in dir_name:
                    folders.append(os.path.join(root,dir_name))

        for folder in folders:
            checkpoints = []
            checkpoint_probs_optimal = 0
            checkpoint_probs_valid = 0
            for root, dirs, files in os.walk(folder):
                for dir_name in dirs:
                    if "checkpoint" in dir_name:
                        checkpoints.append(os.path.join(root,dir_name))
            
            env = KNAPSACKWRAPPER(config['env_config'])
            theta, _ = env.reset()
            optimal_action = list(env.env.solve_knapsack(env.env.KP(env.env.values,env.env.weights,env.env.maximum_weight)))
            optimal_action = [int(optimal_action[i]) for i in range(len(optimal_action))]
            theta['linear_0'] = np.reshape(theta['linear_0'], (1,env.env.items,2))
            theta['quadratic_0'] = np.reshape(theta['quadratic_0'], (1,len(env.env.state["quadratic_0"]),3))

            for checkpoint in checkpoints:

                agent = Policy.from_checkpoint(checkpoint)['default_policy']

                # Do some reshaping due to batch dimension (not always required, depends on vqc structure)
                actions = agent.compute_actions(theta)
                logits = actions[2]["action_dist_inputs"].reshape(env.env.items,2)
                probabilities = np.zeros((env.env.items,2))

                def softmax(logits):
                    exp_logits = np.exp(logits)
                    return exp_logits / np.sum(exp_logits)

                for i in range(len(logits)):
                    probabilities[i] = softmax(logits[i])
                probabilities = probabilities[:,1]
                # Generate all possible combinations of actions
                actions = list(itertools.product([0, 1], repeat=len(probabilities)))
                # Calculate the probability for each combination of actions

                action_probabilities = []
                for action in actions:
                    prob = 1.0
                    for i, val in enumerate(action):
                        prob *= probabilities[i] if val == 1 else (1 - probabilities[i])
                    action_probabilities.append((action, prob))

                # Sort the action probabilities by probability in descending order
                action_probabilities.sort(key=lambda x: x[1], reverse=True)
                average_value = 0
                average_random_value = 0
                random_probability = 1/(2**env.env.items)
                actions = np.reshape(np.unpackbits(np.arange(2**env.env.items).astype('>i8').view(np.uint8)), (-1, 64))[:,-env.env.items:]
                for i,prob in enumerate(action_probabilities):
                    # Store probability of optimal action
                    if action_probabilities[i][0] == tuple(optimal_action):
                        checkpoint_probs_optimal = action_probabilities[i][1]
                    if sum(action_probabilities[i][0][j] * env.env.weights[j] for j in range(env.env.items)) <= env.env.maximum_weight:
                        checkpoint_probs_valid += action_probabilities[i][1]
                        average_contribution = action_probabilities[i][1] * sum(action_probabilities[i][0][j] * env.env.values[j] for j in range(env.env.items))
                        average_value += average_contribution
                        average_random_value += random_probability * average_contribution

            percentages_type_optimal.append(checkpoint_probs_optimal)
            percentages_type_valid.append(checkpoint_probs_valid)
            percentages_better_than_average.append(average_value/average_random_value)

        percentages_optimal.append(np.mean(percentages_type_optimal,axis=0))
        percentages_valid.append(np.mean(percentages_type_valid,axis = 0))
        percentages_better_than_average.append(np.mean(percentages_type_better_than_average,axis=0))

    results_optimal.append(percentages_optimal)
    results_valid.append(percentages_valid)
    results_better_than_average.append(percentages_better_than_average)