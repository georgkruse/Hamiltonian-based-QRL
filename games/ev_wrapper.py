import gymnasium as gym
from games.ev.ev_game import EVGame
from games.ev.ev_game_qubo import EV_Game_QUBO
from games.ev.ev_game_qubo_bandit import EV_Game_QUBO_Bandit
from games.ev.ev_game_qubo_contextual_bandit import EV_Game_QUBO_Contextual_Bandit
from games.ev.ev_game_sequential import EVSEQUENTIALQUBO

class EV_Wrapper(gym.Env):
    def __init__(self,env_config):
        self.env = EVGame(env_config = env_config)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed = None, options = None):
        return self.env.reset()
    
    def step(self,action):
        return self.env.step(action)
    

class EV_Game_QUBO_Wrapper(gym.Env):
    def __init__(self,env_config):
        self.env = EV_Game_QUBO(env_config = env_config)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed = None, options = None):
        return self.env.reset()
    
    def step(self,action):
        return self.env.step(action)
    
class EV_Game_QUBO_Bandit_Wrapper(gym.Env):
    def __init__(self,env_config):
        self.env = EV_Game_QUBO_Bandit(env_config = env_config)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed = None, options = None):
        return self.env.reset()
    
    def step(self,action):
        return self.env.step(action)
    
class EV_Game_QUBO_Contextual_Bandit_Wrapper(gym.Env):
    def __init__(self,env_config):
        self.env = EV_Game_QUBO_Contextual_Bandit(env_config = env_config)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed = None, options = None, use_specific_timestep = False, specific_timestep = 0):
        return self.env.reset(use_specific_timestep=use_specific_timestep, specific_timestep=specific_timestep)
    
    def step(self,action):
        return self.env.step(action)
    
class EV_Game_QUBO_Sequential(gym.Env):
    def __init__(self,env_config):
        self.env = EVSEQUENTIALQUBO(env_config = env_config)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed = None, options = None):
        return self.env.reset()
    
    def step(self,action):
        return self.env.step(action)