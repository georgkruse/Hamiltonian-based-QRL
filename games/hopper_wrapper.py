import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete

class Hopper_Wrapper(gym.Env):
    def __init__(self, env_config):
        if 'render_mode' in env_config.keys():
            self.env = gym.make('Hopper-v4', render_mode = env_config['render_mode'])
        else:
            self.env = gym.make('Hopper-v4')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space         
        self.env_config = env_config

    def reset(self):
        self.counter = 0
        obs = self.env.reset()[0]
        return obs
    
    def step(self, action):
        self.counter += 1
        done = False
        next_state, reward, done1, done2, info = self.env.step(action)
        if done1 or done2:
            done = True

        return next_state, reward, done, info

    def render(self):
        return self.env.render()
    
class Hopper_Wrapper_discrete(gym.Env):
    def __init__(self, env_config):
        self.env_config = env_config
        self.env = gym.make('Hopper-v4')
        self.action_space = MultiDiscrete([8, 8, 8])
        self.observation_space = self.env.observation_space 
        
    def reset(self):
        self.counter = 0
        obs = self.env.reset()[0]
        return obs
    
    def step(self, action):

        action_space_car = np.linspace(-1.,  1., 8)
        a1 = action_space_car[int(action[0])]
        a2 = action_space_car[int(action[1])]
        a3 = action_space_car[int(action[2])]
        self.counter += 1
        done = False
        next_state, reward, done1, done2, info = self.env.step(np.array([a1, a2, a3]))
        if done1 or done2:
            done = True

        return next_state, reward, done, info

    def render(self):
        return self.env.render()

