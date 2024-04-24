import numpy as np
from ray.rllib.policy.policy import Policy
from games.knapsack_wrapper import KNAPSACKSEQUENTIALWRAPPER
import yaml
import numpy as np
import itertools
import os
import matplotlib.pyplot as plt
from collections import Counter
import math

class QAOAVSQRL():
    def __init__(self,root_paths, name_roots, types,cases,path_to_qaoa_results, title):
        self.root_paths = root_paths
        self.name_roots = name_roots
        self.root_paths = list(zip(self.root_paths,self.name_roots))
        self.types = types
        self.problems = self.problems_definitions()
        self.cases = cases
        self.path_to_qaoa_results = path_to_qaoa_results
        self.title = title
        self.plot_results()

    def get_qrl_probabilities(self):
        results_optimal = {}
        results_valid = {}
        results_better_than_random = {}

        for root_path,name_root in self.root_paths:
        
            percentages_optimal = {}
            percentages_valid = {}
            percentages_better_than_random = {}

            folders = []

            with open(root_path + '/alg_config.yml') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)

            for type_ in self.types:
                percentages_type_optimal = []
                percentages_type_valid = []
                percentages_type_better_than_random = []
                folders = []
                for root, dirs, files in os.walk(root_path):
                    for dir_name in dirs:
                        if type_ in dir_name:
                            folders.append(os.path.join(root,dir_name))

                for folder in folders:
                    checkpoints = []
                    checkpoint_probs_optimal = 0
                    checkpoint_probs_valid = 0
                    for root, dirs, files in os.walk(folder):
                        for dir_name in dirs:
                            if "checkpoint" in dir_name:
                                checkpoints.append(os.path.join(root,dir_name))

                    env = KNAPSACKSEQUENTIALWRAPPER(config['env_config'])
                    theta, _ = env.reset()
                    optimal_action = list(env.env.solve_knapsack(env.env.KP(env.env.values,env.env.weights,env.env.maximum_weight)))
                    optimal_action = [int(optimal_action[i]) for i in range(len(optimal_action))]
                    theta['linear_0'] = np.reshape(theta['linear_0'], (1,env.env.items,2))
                    theta['quadratic_0'] = np.reshape(theta['quadratic_0'], (1,len(env.env.state["quadratic_0"]),3))
                    if "annotations" in theta:
                        theta['annotations'] = np.reshape(theta['annotations'], (1,env.env.items,2))

                    for checkpoint in checkpoints:
                    
                        agent = Policy.from_checkpoint(checkpoint)['default_policy']

                        if "Sequential" in name_root:
                            actions_taken = []
                            for iter in range(5000):
                                theta, _ = env.reset()
                                theta['linear_0'] = np.reshape(theta['linear_0'], (1,env.env.items,2))
                                theta['quadratic_0'] = np.reshape(theta['quadratic_0'], (1,int(math.factorial(env.env.items)/((math.factorial(env.env.items - 2)) * math.factorial(2))),3))
                                if "annotations" in theta:
                                    theta['annotations'] = np.reshape(theta['annotations'], (1,env.env.items,2))
                                done = False
                                decisions = [0 for i in range(env.env.items)]
                                while not done:
                                    actions = agent.compute_actions(theta)
                                    action = actions[0][0]
                                    theta, reward, done, _, _ = env.step(action)
                                    if "annotations" in theta:
                                        theta['annotations'] = np.reshape(theta['annotations'], (1,env.env.items,2))
                                    if done == True:
                                        actions_taken.append(decisions)
                                        break
                                    else:
                                        decisions[action] = 1
                            
                            counter = Counter(tuple(lst) for lst in actions_taken)
                            action_probabilities = [(list(lst), count / 5000) for lst, count in counter.items()]
                        
                        else:

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
                            if action_probabilities[i][0] == optimal_action:
                                checkpoint_probs_optimal = action_probabilities[i][1]
                            if sum(action_probabilities[i][0][j] * env.env.weights[j] for j in range(env.env.items)) <= env.env.maximum_weight:
                                checkpoint_probs_valid += action_probabilities[i][1]
                                contribution = sum(action_probabilities[i][0][j] * env.env.values[j] for j in range(env.env.items))
                                average_value += action_probabilities[i][1] * contribution
                                average_random_value += random_probability * contribution

                    percentages_type_optimal.append(checkpoint_probs_optimal)
                    percentages_type_valid.append(checkpoint_probs_valid)
                    percentages_type_better_than_random.append(average_value/average_random_value)

                if f'{type_}' not in percentages_optimal:
                    percentages_optimal[f'{type_}'] = []
                percentages_optimal[f'{type_}'].append(np.mean(percentages_type_optimal,axis=0))

                if f'{type_}' not in percentages_valid:
                    percentages_valid[f'{type_}'] = []
                percentages_valid[f'{type_}'].append(np.mean(percentages_type_valid,axis = 0))

                if f'{type_}' not in percentages_better_than_random:
                    percentages_better_than_random[f'{type_}'] = []
                percentages_better_than_random[f'{type_}'].append(np.mean(percentages_type_better_than_random,axis=0))

            results_optimal[f'{name_root}'] = percentages_optimal
            results_valid[f'{name_root}'] = percentages_valid
            results_better_than_random[f'{name_root}'] = percentages_better_than_random
        
        return results_optimal, results_valid, results_better_than_random
        
    def problems_definitions(self):
        problems = {}

        # Problem instance with 3 nodes
        problems["3"] = {}
        problems['3']["values"] = [5,3,19]
        problems['3']["weights"] = [3,7,25]
        problems['3']["maximum_weight"] = 11

        #Problem instance with 4 nodes
        problems["4"] = {}
        problems['4']["values"] = [3,5,2,10]
        problems['4']["weights"] = [10,3,7,9]
        problems['4']["maximum_weight"] = 20

        # Problem instance with 5 nodes
        problems["5"] = {}
        problems['5']["values"] = [1,10,5,20,17]
        problems['5']["weights"] = [5,9,10,15,3]
        problems['5']["maximum_weight"] = 21

        # Problem instance with 6 nodes
        problems["6"] = {}
        problems['6']["values"] = [20,16,5,9,10,11]
        problems['6']["weights"] = [5,4,10,16,1,17]
        problems['6']["maximum_weight"] = 32

        # Problem instance with 7 nodes
        problems["7"] = {}
        problems['7']["values"] = [20,10,4,18,5,9,18]
        problems['7']["weights"] = [15,10,3,17,9,10,17]
        problems['7']["maximum_weight"] = 37

        # Problem instance with 8 nodes
        problems["8"] = {}
        problems['8']["values"] = [3,5,10,12,22,11,4,19]
        problems['8']["weights"] = [20,15,10,11,3,7,9,16]
        problems['8']["maximum_weight"] = 30

        return problems

    def get_dict_qaoavsqrl(self):
        results_optimal, results_valid, results_better_than_random = self.get_qrl_probabilities()
        qaoa_results = np.load(self.path_to_qaoa_results, allow_pickle=True).item()

        prop_dict = {"slack":{}, "unbalanced":{}, "qrl":{}}

        prop_dict["slack"]["optimal_probability"] = {}
        prop_dict["unbalanced"]["optimal_probability"] = {}
        prop_dict["qrl"]["optimal_probability"] = results_optimal

        prop_dict["slack"]["valid_probability"] = {}
        prop_dict["unbalanced"]["valid_probability"] = {}
        prop_dict["qrl"]["valid_probability"] = results_valid

        prop_dict["slack"]["better_than_random"] = {}
        prop_dict["unbalanced"]["better_than_random"] = {}
        prop_dict["qrl"]["better_than_random"] = results_better_than_random

        for i, problem_size in enumerate(self.cases):
            values = self.problems[f'{problem_size}']["values"]
            weights = self.problems[f'{problem_size}']["weights"]
            maximum_weight = self.problems[f'{problem_size}']["maximum_weight"]
            items = len(values)
            actions = np.reshape(np.unpackbits(np.arange(2**items).astype('>i8').view(np.uint8)), (-1, 64))[:,-items:]
            random_probability = 1/(2**items)

            probabilities_slack = qaoa_results["slack"][problem_size]["probabilities"]
            probabilities_unbalanced = qaoa_results["unbalanced"][problem_size]["probabilities"]

            prop_dict["slack"]["optimal_probability"][problem_size] = qaoa_results["slack"][problem_size]["probability"]
            prop_dict["unbalanced"]["optimal_probability"][problem_size] = qaoa_results["unbalanced"][problem_size]["probability"]

            probability_valid_slack = 0
            probability_valid_unbalanced = 0
            average_random_contribution = 0
            slack_contribution = 0
            unbalanced_contribution = 0

            for k, action in enumerate(actions):
                if sum(action[j] * weights[j] for j in range(items)) <= maximum_weight:
                    probability_valid_slack += probabilities_slack[k]
                    probability_valid_unbalanced += probabilities_unbalanced[k]
                    contribution = sum(action[j] * values[j] for j in range(items))
                    average_random_contribution += contribution * random_probability
                    slack_contribution += probabilities_slack[k] * contribution
                    unbalanced_contribution += probabilities_unbalanced[k] * contribution

            prop_dict["slack"]["valid_probability"][problem_size] = probability_valid_slack
            prop_dict["unbalanced"]["valid_probability"][problem_size] = probability_valid_unbalanced

            prop_dict["slack"]["better_than_random"][problem_size] = slack_contribution / average_random_contribution
            prop_dict["unbalanced"]["better_than_random"][problem_size] = unbalanced_contribution / average_random_contribution
        
        return prop_dict

    def plot_results(self):
        prop_dict = self.get_dict_qaoavsqrl()

        qrl_dict = prop_dict["qrl"]

        fig, ax = plt.subplots(len(self.types), 3, figsize=(14, 4), tight_layout = True)

        qrl_optimal_results = {}
        qrl_valid_results = {}
        qrl_better_than_random = {}

        for type_ in self.types:
            qrl_optimal_results[f"{type_}"] = []
            qrl_valid_results[f"{type_}"] = []
            qrl_better_than_random[f"{type_}"] = []
            for i, root in enumerate(self.root_paths):
                qrl_optimal_results[f"{type_}"].append(qrl_dict["optimal_probability"][root[1]][type_][0])
                qrl_valid_results[f"{type_}"].append(qrl_dict["valid_probability"][root[1]][type_][0])
                qrl_better_than_random[f"{type_}"].append(qrl_dict["better_than_random"][root[1]][type_][0])

        if len(self.types) == 1:
            for type_ in self.types:
                ax[0].plot(self.cases, list(prop_dict["unbalanced"]["optimal_probability"].values()), label="unbalanced", marker="o", markeredgecolor="black", markersize=8)
                ax[0].plot(self.cases, list(prop_dict["slack"]["optimal_probability"].values()), label="slack", marker="^", markeredgecolor="black", markersize=8)
                ax[0].plot(self.cases, qrl_optimal_results[type_], label = f"qrl: {type_}", marker="x", markersize=8)
                ax[1].plot(self.cases, list(prop_dict["unbalanced"]["valid_probability"].values()), label="unbalanced", marker="o", markeredgecolor="black", markersize=8)
                ax[1].plot(self.cases, list(prop_dict["slack"]["valid_probability"].values()), label="slack", marker="^", markeredgecolor="black", markersize=8)
                ax[1].plot(self.cases, qrl_valid_results[type_], label = f"qrl: {type_}", marker="x", markersize=8)
                ax[2].plot(self.cases, list(prop_dict["unbalanced"]["better_than_random"].values()), label="unbalanced", marker="o", markeredgecolor="black", markersize=8)
                ax[2].plot(self.cases, list(prop_dict["slack"]["better_than_random"].values()), label="slack", marker="^", markeredgecolor="black", markersize=8)
                ax[2].plot(self.cases, qrl_better_than_random[type_], label = f"qrl: {type_}", marker="x", markersize=8)
            ax[0].set_xlabel("Problem Size")
            ax[0].set_ylabel("Optimal Probability")
            ax[0].set_xticks(self.cases)
            ax[0].grid()
            ax[0].set_yscale("log")
            ax[1].set_xlabel("Problem Size")
            ax[0].set_ylabel("Valid Probability")
            ax[1].set_xticks(self.cases)
            ax[1].grid()
            ax[2].set_xlabel("Problem Size")
            ax[0].set_ylabel("Better Than Random")
            ax[2].set_xticks(self.cases)
            ax[2].grid()
        
        else:
            for i, type_ in enumerate(self.types):
                ax[i,0].plot(self.cases, list(prop_dict["unbalanced"]["optimal_probability"].values()), label="unbalanced", marker="o", markeredgecolor="black", markersize=8)
                ax[i,0].plot(self.cases, list(prop_dict["slack"]["optimal_probability"].values()), label="slack", marker="^", markeredgecolor="black", markersize=8)
                ax[i,0].plot(self.cases, qrl_optimal_results[type_], label = f"qrl: {type_}", marker="x", markersize=8)
                ax[i,1].plot(self.cases, list(prop_dict["unbalanced"]["valid_probability"].values()), label="unbalanced", marker="o", markeredgecolor="black", markersize=8)
                ax[i,1].plot(self.cases, list(prop_dict["slack"]["valid_probability"].values()), label="slack", marker="^", markeredgecolor="black", markersize=8)
                ax[i,1].plot(self.cases, qrl_valid_results[type_], label = f"qrl: {type_}", marker="x", markersize=8)
                ax[i,2].plot(self.cases, list(prop_dict["unbalanced"]["better_than_random"].values()), label="unbalanced", marker="o", markeredgecolor="black", markersize=8)
                ax[i,2].plot(self.cases, list(prop_dict["slack"]["better_than_random"].values()), label="slack", marker="^", markeredgecolor="black", markersize=8)
                ax[i,2].plot(self.cases, qrl_better_than_random[type_], label = f"qrl: {type_}", marker="x", markersize=8)
                ax[i,0].set_xlabel("Problem Size")
                ax[i,0].set_ylabel("Optimal Probability")
                ax[i,0].set_xticks(self.cases)
                ax[i,0].grid()
                ax[i,0].set_yscale("log")
                ax[i,1].set_xlabel("Problem Size")
                ax[i,0].set_ylabel("Valid Probability")
                ax[i,1].set_xticks(self.cases)
                ax[i,1].grid()
                ax[i,2].set_xlabel("Problem Size")
                ax[i,0].set_ylabel("Better Than Random")
                ax[i,2].set_xticks(self.cases)
                ax[i,2].grid()

        fig.savefig(f"{self.title}.png", dpi=500, bbox_inches="tight")

if __name__ == "__main__":
    root_paths = ["/home/users/coelho/quantum-computing/QRL/logs/knapsack/sequential/qrlvsqaoa/3/2024-03-19--16-47-14_QRL_QPG",
                  "/home/users/coelho/quantum-computing/QRL/logs/knapsack/sequential/qrlvsqaoa/4/2024-03-19--16-48-48_QRL_QPG",
                  "/home/users/coelho/quantum-computing/QRL/logs/knapsack/sequential/qrlvsqaoa/5/2024-03-19--16-47-54_QRL_QPG",
                  "/home/users/coelho/quantum-computing/QRL/logs/knapsack/sequential/qrlvsqaoa/6/2024-03-19--16-48-02_QRL_QPG",
                  "/home/users/coelho/quantum-computing/QRL/logs/knapsack/sequential/qrlvsqaoa/7/2024-03-19--16-48-48_QRL_QPG",
                  "/home/users/coelho/quantum-computing/QRL/logs/knapsack/sequential/qrlvsqaoa/8/2024-03-19--16-49-04_QRL_QPG"
                  ]
    
    name_roots = [
        "Sequential 3",
        "Sequential 4",
        "Sequential 5",
        "Sequential 6",
        "Sequential 7",
        "Sequential 8"
    ]

    types = [
        "=s-ppgl,"
        #"=m-ppgl,",
        #"=h-ppgl,"
    ]
    cases = [3,4,5,6,7,8]
    path_to_qaoa_results = "results_qaoa_comparing.npy"
    title = "Sequential QRL VS QAOA"

    QAOAVSQRL(root_paths,name_roots,types,cases,path_to_qaoa_results,title)

