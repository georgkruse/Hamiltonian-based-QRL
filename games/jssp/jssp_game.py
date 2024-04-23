import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyqubo import Binary
import pennylane as qml
from copy import deepcopy
from collections import OrderedDict
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box, Dict, Discrete
from games.uc.uc_utils import kazarlis_uc, objective_uc_qubo
from pyqubo import Binary, solve_ising, solve_qubo


from games.jssp.jssp_scheduler import get_jss_H

class JSSP_game(gym.Env):

    def __init__(self, env_config):
        
        self.config = env_config
        self.num_variables = env_config['num_variables']
        self.action_space_type = env_config['action_space_type']    
        self.episode_length = env_config['episode_length']
        self.lagrange_precedence = env_config['lagrange_precedence']
        self.lagrange_one_hot = env_config['lagrange_one_hot']
        self.lagrange_share = env_config['lagrange_share']
        self.power_scaling = env_config['power_scaling']

        self.duration = [[1, 1], [1, 2], [2, 1], [2, 2], [1, 3], [3, 1]]
        jobs = {
            "j1": [("m1", self.duration[0][0])],
            "j2": [("m1", self.duration[0][1])]
        }

        self.max_time = 4	  # Put an upperbound on how long the schedule can be
        H = get_jss_H(jobs,
                      self.max_time, 
                      lagrange_one_hot=self.lagrange_one_hot, 
                      lagrange_share=self.lagrange_share, 
                      lagrange_precedence=self.lagrange_precedence)
        
        H = H * self.power_scaling        
        self.model = H.compile()
        linear, quadratic, offset = self.model.to_ising()
        spaces = {}
        spaces[f'linear_{0}'] = Box(-np.inf, np.inf, shape = (self.num_variables,2), dtype='float64')
        spaces[f'quadratic_{0}'] = Box(-np.inf, np.inf, shape = (sum(range(self.num_variables)),3), dtype='float64')
        self.observation_space = Dict(spaces=spaces)

        if self.action_space_type == 'multi_discrete':
            self.action_space = MultiDiscrete([2 for _ in range(env_config['num_variables'])])
        elif self.action_space_type == 'discrete':
            self.action_space = Discrete(2**env_config['num_variables'])
    

    def reset(self, seed=None, options=None):
        self.history = []
        self.timestep = 0
        state = OrderedDict()

        jobs = {
                "j0": [("m1", self.duration[self.timestep][0])],
                "j1": [("m1", self.duration[self.timestep][1])]
                }

        H = get_jss_H(jobs,
                      self.max_time, 
                      lagrange_one_hot=self.lagrange_one_hot, 
                      lagrange_share=self.lagrange_share, 
                      lagrange_precedence=self.lagrange_precedence)
        
        H = H * self.power_scaling        
        self.model = H.compile()

        linear, quadratic, offset = self.model.to_ising()
        self.env_keys = {'j0_0,0': 0, 'j0_0,1': 1, 'j0_0,2': 2, 'j0_0,3': 3, 'j1_0,0': 4, 'j1_0,1': 5,  'j1_0,2': 6, 'j1_0,3': 7} 
       

        if len(quadratic.keys()) < sum(range(self.num_variables)):
            for key0 in self.env_keys:
                for key1 in self.env_keys:
                    if key0 != key1:
                        if (key0, key1) not in quadratic.keys():
                            if (key1, key0) not in quadratic.keys():
                                quadratic[(key0, key1)] = 0
        
        self.linear = linear 
        self.quadratic = quadratic
        linear_values = - np.array([*linear.values()])
        state[f'linear_{0}'] = np.stack([[self.env_keys[key] for key in linear.keys()], linear_values], axis=1)
        state[f'quadratic_{0}'] = np.stack([[self.env_keys[key[0]], self.env_keys[key[1]], value] for key, value in zip(quadratic.keys(),quadratic.values())])
        
        self.optimal_solution = solve_ising(linear, quadratic)
        decoded_sample = self.model.decode_sample(self.optimal_solution, vartype='SPIN')
        self.optimal_energy = decoded_sample.energy
        return state, {}
    
    def step(self, action_):
        
        done = False
        next_state = OrderedDict()
        action = deepcopy(action_)
        if isinstance(self.action_space, Discrete):
            binary_actions = np.reshape(np.unpackbits(np.arange(2**self.n).astype('>i8').view(np.uint8)), (-1, 64))[:,-self.num_variables:]
            action = binary_actions[action]
        
        action[action == 0] = -1
        self.optimal_solution = solve_ising(self.linear, self.quadratic)
        decoded_sample = self.model.decode_sample(self.optimal_solution, vartype='SPIN')
        self.optimal_energy = decoded_sample.energy

        decoded_sample = self.model.decode_sample(action, vartype='SPIN')
        reward = decoded_sample.energy
        

        if self.config['reward_mode'] == 'optimal':
            reward = reward - self.optimal_energy
            reward = - reward 
        elif self.config['reward_mode'] == 'optimal_log':
            reward = reward - self.optimal_energy + 1e-9
            reward = - np.log(reward) 
            reward = np.min([10.0, reward]) 

        
        self.timestep +=1

        jobs = {
                "j0": [("m1", self.duration[self.timestep][0])],
                "j1": [("m1", self.duration[self.timestep][1])]
                }

        H = get_jss_H(jobs,
                      self.max_time, 
                      lagrange_one_hot=self.lagrange_one_hot, 
                      lagrange_share=self.lagrange_share, 
                      lagrange_precedence=self.lagrange_precedence)
        
        H = H * self.power_scaling        
        self.model = H.compile()      
        linear, quadratic, offset = self.model.to_ising()

        self.linear = linear 
        self.quadratic = quadratic

        if len(quadratic.keys()) < sum(range(self.num_variables)):
            for key0 in self.env_keys:
                for key1 in self.env_keys:
                    if key0 != key1:
                        if (key0, key1) not in quadratic.keys():
                            if (key1, key0) not in quadratic.keys():
                                quadratic[(key0, key1)] = 0

        linear_values = - np.array([*linear.values()])
        next_state[f'linear_{0}'] = np.stack([[self.env_keys[key] for key in linear.keys()], linear_values], axis=1)
        next_state[f'quadratic_{0}'] = np.stack([[self.env_keys[key[0]], self.env_keys[key[1]], value] for key, value in zip(quadratic.keys(),quadratic.values())])
        
        if self.timestep >= self.episode_length:
            done = True
            self.timestep = 0

        return next_state, reward, done, False, {}


    def get_hamiltonian(self, timestep):
        linear, quadratic = self.get_ising(timestep)
        H_linear = qml.Hamiltonian(
            [linear[key] for key in linear.keys()],
            [qml.PauliZ(int(key)) for key in linear.keys()]
        )

        H_quadratic = qml.Hamiltonian(
            [quadratic[key] for key in quadratic.keys()],
            [qml.PauliZ(int(key[0]))@qml.PauliZ(int(key[1])) for key in quadratic.keys()]
        )

        H = H_quadratic - H_linear
        return H
    
    def get_ising(self, timestep):
        y = [Binary(f'{i}') for i in range(self.n)]
        y = [Binary(f'{i}') for i in range(self.n)]
        qubo = self.objective_fp_qubo(y, self.units, self.C, self.T, self.lambda_1)
        model = qubo.compile()
        linear, quadratic, _ = model.to_ising()
        return linear, quadratic
    
   

