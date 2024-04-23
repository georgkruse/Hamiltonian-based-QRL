import gymnasium as gym
# from games.knapsack.knapsack_bandit import Knapsack
# from games.knapsack.knapsack_bandit_custom_reward import KnapsackCUSTOM
# from games.knapsack.knapsack_sequential import KnapsackSequential
# from games.knapsack.knapsack_sequential_dynamic import KnapsackSequentialDynamic
# from games.knapsack.knapsack_sequential_dynamic_validation import KnapsackSequentialDynamicValidation

class KNAPSACKWRAPPER(gym.Env):
    def __init__(self,env_config):
        self.env = Knapsack(env_config = env_config)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed = None, options = None):
        return self.env.reset()
    
    def step(self,action):
        return self.env.step(action)
    
class KNAPSACKCUSTOMWRAPPER(gym.Env):
    def __init__(self,env_config):
        self.env = KnapsackCUSTOM(env_config = env_config)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed = None, options = None):
        return self.env.reset()
    
    def step(self,action):
        return self.env.step(action)
    
class KNAPSACKSEQUENTIALWRAPPER(gym.Env):
    def __init__(self,env_config):
        self.env = KnapsackSequential(env_config = env_config)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed = None, options = None):
        return self.env.reset()
    
    def step(self,action):
        return self.env.step(action)
    
class KNAPSACKSEQUENTIALDYNAMICWRAPPER(gym.Env):
    def __init__(self,env_config):
        self.env = KnapsackSequentialDynamic(env_config = env_config)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed = None, options = None, use_specific_timestep = False, timestep = 0):
        return self.env.reset(use_specific_timestep=use_specific_timestep, timestep=timestep)
    
    def step(self,action):
        return self.env.step(action)
    
class KNAPSACKSEQUENTIALDYNAMICVALIDATIONWRAPPER(gym.Env):
    def __init__(self,env_config):
        self.env = KnapsackSequentialDynamicValidation(env_config = env_config)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed = None, options = None, use_specific_timestep = False, timestep = 0):
        return self.env.reset(use_specific_timestep=use_specific_timestep, timestep=timestep)
    
    def step(self,action):
        return self.env.step(action)