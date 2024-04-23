import torch
import numpy as np
import pandas as pd
from pyqubo import Binary
import gymnasium as gym
from gymnasium.spaces import MultiBinary, Box

from games.uc.uc_utils import kazarlis_uc, objective_uc_qubo

class UC_game(gym.Env):

    def __init__(self, env_config):
        
        self.action_space = MultiBinary(env_config['num_generators'])
        self.observation_space = Box(-np.inf, np.inf, shape = (env_config['num_generators'],), dtype='float64')
        
        self.n = env_config['num_generators']
        self.p_min, self.p_max, A_org, B_org, self.C = kazarlis_uc(self.n) 
        list_power_scaling = [1e-2]
        list_L_org = [0, 100, 300, 600] #,500,625,800]
        continuous_params_random_org = [100, 200, 300, 100, 150, 75, 75, 50,50,50,300,250,125,125,100,75]
        self.lambda_ = 1.0

        for power_scaling in list_power_scaling:
            # Scaling input parameters according to power_scaling
            A = np.array(A_org)*power_scaling**2
            self.A = A.tolist()
            B = np.array(B_org)*power_scaling
            self.B = B.tolist()

            list_L = np.array(list_L_org)*power_scaling
            self.list_L = list_L.tolist()

            continuous_params_random = np.array(continuous_params_random_org)*power_scaling
            self.continuous_params_random = continuous_params_random.tolist()


    def reset(self, seed=None, options=None):
        self.timestep = 0
        self.day_demand = np.random.choice(self.list_L, 24) 
        target = self.day_demand[self.timestep]
        
        # Use Pyqubo to easily convert your QUBO to a Ising formulation
        y = [Binary(f'{i}') for i in range(self.n)]
        qubo = objective_uc_qubo(y, self.A, self.B, self.C, target, self.lambda_, self.n, self.continuous_params_random)

        model = qubo.compile()
        linear, quadratic, offset = model.to_ising()
        if target == 0.:
            target_ = torch.tensor([-1., -1., -1.])
        elif target == 1.:
            target_ = torch.tensor([1., -1., -1.])
        elif target == 3.:
            target_ = torch.tensor([1., 1., -1.])
        elif target == 6.:
            target_ = torch.tensor([1., 1., 1.])
        return linear, quadratic, offset, target_
    
    def step(self, action):
        self.counter += 1
        done = False
        next_state, reward, done1, done2, info = self.env.step(action)
        if done1 or done2:
            done = True
        next_state *= self.norm
        # <obs>, <reward: float>, <terminated: bool>, <truncated: bool>, <info: dict>
        return next_state, reward, done, False, info

    def render(self):
        return self.env.render()
    