"""
@author: Georg Kruse
georg.kruse@iisb.fraunhofer.de
Fraunhofer Institute for Integrated Systems and Device Technology IISB
"""
import ray.rllib.algorithms.simple_q as simple_q
import ray.rllib.algorithms.ppo as ppo
import ray.rllib.algorithms.pg as pg

from utils.config.common import add_hyperparameters
from utils.config.create_env import create_env
from agents.quantum_ppo_agent import QuantumPPOModel
from agents.quantum_pg_agent import QuantumPGModel
from agents.quantum_dqn_agent import QuantumDQN_Model, CustomEpsilonGreedy
from agents.callbacks import *

from utils.config.create_env import create_env
from utils.config.common import *

def create_config(config, env_config):

    if config.type.lower() == 'ES'.lower() or config.type == 'GA'.lower():
        tune_config = {}
        tune_config = add_hyperparameters(tune_config, config)
        tune_config['env'] = create_env(config, env_config)
        tune_config['env_config'] = env_config
        return tune_config
    
    elif config.type.lower() == 'QRL'.lower():

        config_switch = {'QPPO': ppo.PPOConfig(), 
                         'QDQN': simple_q.SimpleQConfig(), 
                         'QPG': pg.PGConfig(), 
                        }
        
        model_switch = {'QPPO': QuantumPPOModel,
                        'QPG': QuantumPGModel,
                        'QDQN': QuantumDQN_Model
                        }
        callback_switch = {
                        'PG_MaxCut_Callbacks': PG_MaxCut_Callbacks,
                        'KPCallbacks':  KnapsackCallback,
                        'PG_TSP_Callbacks': PG_TSP_Callbacks,
        }
        
        # keys to lower-case for case insensitivity and improved fault tolerance
        config_switch = {k.lower(): v for k, v in config_switch.items()}

        if config.alg.lower() not in config_switch:
            err_msg = "There does not exist any default configuration for provided ray algorithm %s" \
                    " check whether it is a custom algorithm".format(config.alg)
            raise ValueError(err_msg)
        
        create_env(config)
        default_config = config_switch[config.alg.lower()]
        
        if 'callback' in config.env_config.keys():
            default_config = default_config.callbacks(callback_switch[config.env_config['callback']]).reporting(keep_per_episode_custom_metrics=True)

        alg_config = config.algorithm_config 
        tune_config = (
                default_config
                .framework(alg_config['framework'])
                .environment(env=config.env, env_config=env_config)
                .rl_module(_enable_rl_module_api=False)
                )

        alg_config = config.algorithm_config 
        alg_config = add_hyperparameters(alg_config)
       
        tune_config['model']['custom_model'] = model_switch[config.alg]
        tune_config['model']['custom_model_config'] = {}
        
        for key in alg_config.keys():
            if key in tune_config.keys():
                tune_config[key] = alg_config[key]
            else:
                tune_config['model']['custom_model_config'][key] = alg_config[key]

        tune_config['env_config'] = env_config

        if tune_config["exploration_config"]["type"] == "CustomEpsilonGreedy":
            tune_config["exploration_config"]["type"] = CustomEpsilonGreedy

        return tune_config

