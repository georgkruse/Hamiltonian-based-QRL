import gymnasium as gym
from games.mvc.mvc import MVC
from games.mvc.mvc_sequential import MVCSEQUENTIAL
from games.mvc.mvc_sequential_dataset import MVCSEQUENTIALDATASET

class MVC_Wrapper(gym.Env):
    def __init__(self,env_config):
        self.env = MVC(env_config = env_config)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed = None, options = None):
        return self.env.reset()
    
    def step(self,action):
        return self.env.step(action)
    
class MVCSEQUENTIAL_Wrapper(gym.Env):
    def __init__(self,env_config):
        self.env = MVCSEQUENTIAL(env_config = env_config)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed = None, options = None):
        return self.env.reset()
    
    def step(self,action):
        return self.env.step(action)
    
class MVCSEQUENTIALDATASET_Wrapper(gym.Env):
    def __init__(self,env_config):
        self.env = MVCSEQUENTIALDATASET(env_config = env_config)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed = None, options = None,use_specific_timestep = False, timestep = 0):
        return self.env.reset(use_specific_timestep = use_specific_timestep, timestep = timestep)
    
    def step(self,action):
        return self.env.step(action)