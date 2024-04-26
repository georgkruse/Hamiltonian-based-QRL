
from games.cartpole_wrapper import Cartpole_Wrapper
from games.pendulum_wrapper import Pendulum_Wrapper, Pendulum_Wrapper_discrete, Pendulum_Wrapper_no_norm
from games.lunarlander_wrapper import LunarLander_Wrapper, LunarLander_Wrapper_discrete
from games.uc.uc_game import UC_game
from games.tsp.tsp_game import TSP
from games.maxcut.weighted_maxcut import WeightedMaxCut
from games.knapsack.knapsack_sequential_dynamic import KnapsackSequentialDynamic
from ray.tune.registry import register_env

wrapper_switch = {'CartPole-v1': Cartpole_Wrapper,
                  'Pendulum-v1_discrete': Pendulum_Wrapper_discrete,
                  'Pendulum-v1': Pendulum_Wrapper,
                  'Pendulum-v1-no-norm': Pendulum_Wrapper_no_norm,
                  'LunarLander-v2': LunarLander_Wrapper,
                  'LunarLander-v2_discrete': LunarLander_Wrapper_discrete,
                  'UC': UC_game,
                  'TSP': TSP,
                  'weightedMaxCut': WeightedMaxCut,
                  'KP': KnapsackSequentialDynamic}

def create_env(config):
    
    try: 
        env = wrapper_switch[config.env]
    except:
        raise NotImplementedError(\
            f"Cannot create '{config.env}' - enviroment.")
    
    register_env(config.env, env)
    print("Training on {} - game.".format(config.env))
    