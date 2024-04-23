import gymnasium as gym
from gymnasium.spaces import MultiDiscrete

# class Walker2d_Wrapper(gym.Env):
#     def __init__(self, env_config):
#         self.env = gym.make('Walker2d-v4')
#         self.action_space = MultiDiscrete([3, 7])
#         self.observation_space = self.env.observation_space
#     def reset(self):
#         self.counter = 0
#         obs = self.env.reset()[0]
#         return obs
#     def step(self, action):
#         self.counter += 1
#         done = False
#         next_state, reward, done1, done2, info = self.env.step(action)
#         if done1 or done2:
#             done = True
#         return next_state, reward, done, info

class Walker2d_Wrapper(gym.Env):
    def __init__(self, env_config):
        self.env = gym.make('Walker2d-v4')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
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
