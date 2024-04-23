import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
from collections import OrderedDict
from pyqubo import Binary
import numpy as np


class MAXCUT(gym.Env):
    def __init__(self,env_config):
        self.config = env_config
        self.graph = self.config["graph"]
        self.nodes = self.config["nodes"]

        self.state = OrderedDict()

        y = [Binary(f"{i}") for i in range(self.nodes)]

        qubo = self.formulate_qubo(y)
        model = qubo.compile()
        linear,quadratic,offset = model.to_ising()

        spaces = {}
        spaces['linear_0'] = Box(-np.inf, np.inf, shape = (self.nodes,2), dtype='float64')
        spaces['quadratic_0'] = Box(-np.inf, np.inf, shape = (len(quadratic.values()),3), dtype='float64')

        self.observation_space = Dict(spaces = spaces)
        self.action_space = MultiDiscrete([2 for _ in range(self.nodes)])

        aux = []
        self.optimal_costs = []
        actions = np.reshape(np.unpackbits(np.arange(2**self.nodes).astype('>i8').view(np.uint8)), (-1, 64))[:,-self.nodes:]
        for action in actions:
            cost = self.formulate_qubo(action)
            aux.append(cost)
        self.optimal_costs = np.min(aux)
        self.optimal_actions = actions[np.argmin(aux)]
        self.max_cost = np.max(aux)

    def formulate_qubo(self,y):
        QUBO = 0

        for edge in self.graph:
            node_1, node_2 = edge[0],edge[1]
            QUBO -= y[node_1] + y[node_2] - 2 * y[node_1] * y[node_2]

        return QUBO
    
    def reset(self, seed = None, options = None):
        state = OrderedDict()
        
        y = [Binary(f"{i}") for i in range(self.nodes)]

        qubo = self.formulate_qubo(y)

        model = qubo.compile()
        linear,quadratic,offset = model.to_ising()
        for i in range(self.nodes):
            linear[str(i)] = 0
        linear_values = - np.array([*linear.values()])

        state['linear_0'] = np.stack([[int(key) for key in linear.keys()], linear_values], axis=1)
        state['quadratic_0'] = np.stack([[int(key[0]), int(key[1]), value] for key, value in zip(quadratic.keys(),quadratic.values())])

        return state, {}
    
    def step(self,action):
        next_state = OrderedDict()

        done = True

        y = [Binary(f"{i}") for i in range(self.nodes)]

        qubo = self.formulate_qubo(y)

        model = qubo.compile()
        linear,quadratic,offset = model.to_ising()
        
        linear_values = - np.array([*linear.values()])

        next_state['linear_0'] = np.stack([[int(key) for key in linear.keys()], linear_values], axis=1)
        next_state['quadratic_0'] = np.stack([[int(key[0]), int(key[1]), value] for key, value in zip(quadratic.keys(),quadratic.values())])

        reward = self.formulate_qubo(action)
        reward -= self.optimal_costs
        reward = -reward

        return next_state, reward, done, False, {}


if __name__ == "__main__":
    env_config = {
        "nodes": 5,
        "graph": [(0,1),(0,2),(1,3),(2,3),(2,4),(3,4)]
    }

    env = MAXCUT(env_config)

    for i in range(10):
        state,_ = env.reset()
        done = False
        while not done:
            action = [0,1,0,1]
            state, reward, done, _, _ = env.step(action)
            print(reward)
            if done == True:
                break