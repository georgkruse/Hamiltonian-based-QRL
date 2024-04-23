import gymnasium as gym
import numpy as np

class MountainCar_Wrapper(gym.Env):
    def __init__(self, env_config):
        self.env = gym.make('MountainCarContinuous-v0')
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
        done = False
        # next_state, reward, done, _, info = self.env.step(action)
        # next_state = next_state / self.state_bounds
        # print(next_state)
        next_state, reward, done1, done2, info = self.env.step(action)
        if done1 or done2:
            done = True
        return next_state, reward, done, info