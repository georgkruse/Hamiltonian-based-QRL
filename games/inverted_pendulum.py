import gymnasium as gym
import numpy as np

class Inverted_Pendulum(gym.Env):
    def __init__(self, env_config):
        self.env = gym.make('InvertedPendulum-v4')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        # self.state_bounds = np.array([2.4, 2.5, 0.405, 2.5])
    def reset(self):
        self.counter = 0
        obs = self.env.reset()[0]
        # print(obs)
        # obs = obs / self.state_bounds
        return obs
    def step(self, action):
        self.counter += 1
        next_state, reward, done, _, info = self.env.step(action)
        # next_state = next_state / self.state_bounds
        # print(next_state)
        if self.counter >= 200:
            done = True
        return next_state, reward, done, info
    
