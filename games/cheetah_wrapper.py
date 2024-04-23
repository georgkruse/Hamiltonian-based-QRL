import gymnasium as gym

class Cheetah_Wrapper(gym.Env):
    def __init__(self, env_config):
        self.env = gym.make('HalfCheetah-v4')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    def reset(self):
        self.counter = 0
        obs = self.env.reset()[0]
        return obs
    def step(self, action):
        self.counter += 1
        next_state, reward, done, _, info = self.env.step(action)
        if self.counter >= 1000:
            done = True
        return next_state, reward, done, info

    # def reset(self):
    #     self.counter = 0
    #     obs = self.env.reset()
    #     print(obs)
    #     return obs
    # def step(self, action):
    #     self.counter += 1
    #     next_state, reward, done, info = self.env.step(action)
    #     if self.counter >= 1000:
    #         done = True
    #     return next_state, reward, done, info