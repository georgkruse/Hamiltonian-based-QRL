import gymnasium as gym
from games.maxcut.maxcut import MAXCUT
from games.maxcut.maxcut_contextual_bandit import MAXCUTCONTEXTUALBANDIT
from games.maxcut.maxcut_sequential import MAXCUTSEQUENTIAL
from games.maxcut.maxcut_dynamic import MAXCUTDYNAMIC
from games.maxcut.maxcut_weighted_static import MAXCUTWEIGHTED
from games.maxcut.maxcut_weighted_dynamic import MAXCUTWEIGHTEDDYNAMIC

class MAXCUT_Wrapper(gym.Env):
    def __init__(self,env_config):
        self.env = MAXCUT(env_config = env_config)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed = None, options = None):
        return self.env.reset()
    
    def step(self,action):
        return self.env.step(action)
    
class MAXCUTCONTEXTUALBANDIT_Wrapper(gym.Env):
    def __init__(self,env_config):
        self.env = MAXCUTCONTEXTUALBANDIT(env_config = env_config)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed = None, options = None,use_specific_timestep = False, timestep = 0):
        return self.env.reset(use_specific_timestep=use_specific_timestep, timestep=timestep)
    
    def step(self,action):
        return self.env.step(action)
    
class MAXCUTSEQUENTIAL_Wrapper(gym.Env):
    def __init__(self,env_config):
        self.env = MAXCUTSEQUENTIAL(env_config = env_config)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed = None, options = None):
        return self.env.reset()
    
    def step(self,action):
        return self.env.step(action)
    
class MAXCUTDYNAMIC_Wrapper(gym.Env):
    def __init__(self,env_config):
        self.env = MAXCUTDYNAMIC(env_config = env_config)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed = None, options = None):
        return self.env.reset()
    
    def step(self,action):
        return self.env.step(action)

class MAXCUTWEIGHTEDDYNAMIC_Wrapper(gym.Env):
    def __init__(self,env_config):
        self.env = MAXCUTWEIGHTEDDYNAMIC(env_config = env_config)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed = None, options = None):
        return self.env.reset()
    
    def step(self,action):
        return self.env.step(action)

class MAXCUTWEIGHTED_Wrapper(gym.Env):
    def __init__(self,env_config):
        self.env = MAXCUTWEIGHTED(env_config = env_config)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed = None, options = None):
        return self.env.reset()
    
    def step(self,action):
        return self.env.step(action)
    
