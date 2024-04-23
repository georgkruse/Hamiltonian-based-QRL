import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
from collections import OrderedDict
from pyqubo import Binary
import numpy as np
from openqaoa.problems import FromDocplex2IsingModel
from docplex.mp.model import Model

class Knapsack(gym.Env):
    def __init__(self,env_config):
        self.config = env_config
        self.values = self.config["values"]
        self.weights = self.config["weights"]
        self.maximum_weight = self.config["maximum_weight"]
        self.lambdas = self.config['lambdas']
        #self.reward_scaling = self.config["reward_scaling"]
        self.items = len(self.values)

        self.state = OrderedDict()
        self.mdl = self.KP(self.values,self.weights,self.maximum_weight)
        self.solution_str = self.solve_knapsack(self.mdl)
        self.optimal_action = list(self.solution_str)
        self.optimal_action = [int(a) for a in self.optimal_action]
        #self.optimal_reward = 

        ising_hamiltonian = FromDocplex2IsingModel(self.mdl,
                                   unbalanced_const=True,
                                   strength_ineq=[self.lambdas[0],self.lambdas[1]]).ising_model
        
        list_terms_coeffs = list(zip(ising_hamiltonian.terms, ising_hamiltonian.weights))
        linear_terms = [[i[0],w] for i,w in list_terms_coeffs if len(i) == 1]
        quadratic_terms = [[i,w] for i,w in list_terms_coeffs if len(i) == 2]

        linear = {str(item[0]): item[1] for item in linear_terms}
        quadratic = {tuple(str(item[0][i]) for i in range(len(item[0]))): item[1] for item in quadratic_terms}

        self.state['linear_0'] = np.stack([[int(key[0]), value] for key, value in zip(linear.keys(),linear.values())])
        self.state['quadratic_0'] = np.stack([[int(key[0]), int(key[1]), value] for key, value in zip(quadratic.keys(),quadratic.values())])

        spaces = {}
        spaces['linear_0'] = Box(-np.inf, np.inf, shape = (self.items,2), dtype='float64')
        spaces['quadratic_0'] = Box(-np.inf, np.inf, shape = (len(self.state["quadratic_0"]),3), dtype='float64')

        self.observation_space = Dict(spaces = spaces)
        self.action_space = MultiDiscrete([2 for _ in range(self.items)])
    
    def KP(self,values: list, weights: list, maximum_weight: int):
        """
        Crete a Docplex model of the Knapsack problem:

        args:
            values - list containing the values of the items
            weights - list containing the weights of the items
            maximum_weight - maximum weight allowed in the "backpack"
        """

        mdl = Model("Knapsack")
        num_items = len(values)
        x = mdl.binary_var_list(range(num_items), name = "x")
        cost = -mdl.sum(x[i] * values[i] for i in range(num_items))
        mdl.minimize(cost)
        mdl.add_constraint(
            mdl.sum(x[i] * weights[i] for i in range(num_items)) <= maximum_weight
        )
        return mdl
    
    def solve_knapsack(self,mdl):
        docplex_sol = mdl.solve()
        solution = ""
        for ii in mdl.iter_binary_vars():
            solution += str(int(np.round(docplex_sol.get_value(ii),1)))
        return solution
    
    def calculate_reward(self,action):
        reward = sum(action[i] * self.values[i] for i in range(self.items))

        # Check if constraint was broken
        weight = sum(action[i]*self.weights[i] for i in range(self.items))
        if weight > self.maximum_weight:
            reward = 0
        return reward

    
    def reset(self, seed = None, options = None):
        return self.state, {}
    
    def step(self,action):
        done = True

        reward = self.calculate_reward(action)

        return self.state, reward, done, False, {}


if __name__ == "__main__":
    config = {
        "values" : [1,2,3],
        "weights":  [3,2,1],
        "maximum_weight":   5,
        "lambdas": [0.96,0.03],
        "reward_scaling": 2
    }

    env = Knapsack(config)

    for i in range(10):
        state,_ = env.reset()
        done = False
        while not done:
            action = [0,1,1]
            state, reward, done, _, _ = env.step(action)
            print(reward)
            if done == True:
                break