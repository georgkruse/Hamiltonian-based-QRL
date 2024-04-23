from agents.quantum_ppo_agent import QRL_PPO
from agents.quantum_pg_agent import QRL_PG
from agents.evolution_strategy import Evolution_Strategy        
from agents.genetic_algorithm import Genetic_Algorithm 
from agents.quantum_dqn_agent import QRL_DQN

qrl_switch = {'QPPO': QRL_PPO, 'QPG': QRL_PG, 'QDQN':QRL_DQN}

def create_algorithm(config):
           
    if config.type =='ES':
        algorithm = Evolution_Strategy

    elif config.type == 'GA':
        algorithm = Genetic_Algorithm

    elif config.type == 'QRL':
        algorithm = qrl_switch[config.alg] 
        
    return algorithm