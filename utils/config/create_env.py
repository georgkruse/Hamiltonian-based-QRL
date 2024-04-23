
from games.cartpole_wrapper import Cartpole_Wrapper
from games.cheetah_wrapper import Cheetah_Wrapper
from games.hopper_wrapper import Hopper_Wrapper, Hopper_Wrapper_discrete
from games.walker2d_wrapper import Walker2d_Wrapper
from games.pendulum_wrapper import Pendulum_Wrapper, Pendulum_Wrapper_discrete, Pendulum_Wrapper_no_norm
from games.mountaincar_wrapper import MountainCar_Wrapper
from games.lunarlander_wrapper import LunarLander_Wrapper, LunarLander_Wrapper_discrete
from games.ev_wrapper import EV_Wrapper, EV_Game_QUBO_Wrapper, EV_Game_QUBO_Bandit_Wrapper
from games.fp.fp_game import FP_game
from games.uc.uc_game import UC_game
from games.mvc_wrapper import MVCSEQUENTIAL_Wrapper, MVCSEQUENTIALDATASET_Wrapper
from games.maxcut_wrapper import MAXCUTSEQUENTIAL_Wrapper, MAXCUTDYNAMIC_Wrapper, MAXCUTWEIGHTED_Wrapper, MAXCUTWEIGHTEDDYNAMIC_Wrapper, MAXCUTCONTEXTUALBANDIT_Wrapper
from games.jssp.jssp_game import JSSP_game
from games.tsp.tsp_game import TSP
from games.knapsack_wrapper import KNAPSACKWRAPPER, KNAPSACKCUSTOMWRAPPER, KNAPSACKSEQUENTIALWRAPPER, KNAPSACKSEQUENTIALDYNAMICWRAPPER
from ray.tune.registry import register_env

wrapper_switch = {'CartPole-v1': Cartpole_Wrapper,
                  'HalfCheetah-v4': Cheetah_Wrapper,
                  'Hopper-v4': Hopper_Wrapper,
                  'Hopper-v4_discrete': Hopper_Wrapper_discrete,
                  'Walker2d-v4': Walker2d_Wrapper,
                  'Pendulum-v1_discrete': Pendulum_Wrapper_discrete,
                  'Pendulum-v1': Pendulum_Wrapper,
                  'Pendulum-v1-no-norm': Pendulum_Wrapper_no_norm,
                  'MountainCarContinuous-v0': MountainCar_Wrapper,
                  'LunarLander-v2': LunarLander_Wrapper,
                  'LunarLander-v2_discrete': LunarLander_Wrapper_discrete,
                  'UC': UC_game,
                  'JSSP': JSSP_game,
                  'EV': EV_Wrapper,
                  'FP': FP_game,
                  'EV': EV_Wrapper,
                  'TSP': TSP,
                  'EVQUBO': EV_Game_QUBO_Wrapper,
                  'EVQUBOBANDIT': EV_Game_QUBO_Bandit_Wrapper,
                  'MVCSEQUENTIAL': MVCSEQUENTIAL_Wrapper,
                  'MVCSEQUENTIALDATASET': MVCSEQUENTIALDATASET_Wrapper,
                  'MAXCUTSEQUENTIAL': MAXCUTSEQUENTIAL_Wrapper,
                  'MAXCUTDYNAMIC': MAXCUTDYNAMIC_Wrapper,
                  'MAXCUTWEIGHTED': MAXCUTWEIGHTED_Wrapper,
                  'MAXCUTWEIGHTEDDYNAMIC': MAXCUTWEIGHTEDDYNAMIC_Wrapper,
                  'MAXCUTCONTEXTUALBANDIT': MAXCUTCONTEXTUALBANDIT_Wrapper,
                  'KNAPSACKSEQUENTIALDYNAMIC': KNAPSACKSEQUENTIALDYNAMICWRAPPER,
                  'KNAPSACKSEQUENTIAL':KNAPSACKSEQUENTIALWRAPPER}

def create_env(config):
    
    try: 
        env = wrapper_switch[config.env]
    except:
        raise NotImplementedError(\
            f"Cannot create '{config.env}' - enviroment.")
    
    register_env(config.env, env)
    print("Training on {} - game.".format(config.env))
    