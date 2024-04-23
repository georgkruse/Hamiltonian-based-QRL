import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
from collections import OrderedDict
from pyqubo import Binary
import numpy as np


class MVCSEQUENTIAL(gym.Env):
    def __init__(self,env_config):
        self.config = env_config
        self.graph = self.config["graph"]
        self.nodes = self.config["nodes"]
        self.P = self.config["P"]
        self.use_linear_terms = self.config["use_linear_terms"]

        self.state = OrderedDict()
        self.timestep = 0

        self.a = {}
        for node in range(self.nodes):
            self.a[str(node)] = np.pi

        quadratic_values = len(self.graph)

        spaces = {}

        spaces['linear_0'] = Box(-np.inf, np.inf, shape = (self.nodes,2), dtype='float64')
        spaces['quadratic_0'] = Box(-np.inf, np.inf, shape = (quadratic_values,3), dtype='float64')
        spaces['a_0'] = Box(-np.inf, np.inf, shape = (self.nodes,2))

        self.observation_space = Dict(spaces = spaces)
        self.action_space = Discrete(self.nodes)

    def formulate_qubo(self,y):
        objective = 0

        for i in range(self.nodes):
            objective += y[i]

        edges_constraints = []

        for edge in self.graph:
            node_0,node_1 = edge[0],edge[1]
            edges_constraints.append(self.P * (1 - y[node_0] - y[node_1] + y[node_0]*y[node_1]))
        
        QUBO = objective + sum(edges_constraints)
        return QUBO
    
    def check_all_edges_covered(self):
        covered_edges = []
        accepted_nodes = []
        for key in self.a:
            if self.a[key] == 0:
                accepted_nodes.append(int(key))
        for node in accepted_nodes:
            for edge in self.graph:
                if node in edge:
                    if edge not in covered_edges:
                        covered_edges.append(edge)
        return len(covered_edges) == len(self.graph)
    
    def reset(self, seed = None, options = None):
        self.timestep = 0
        state = OrderedDict()

        for node in range(self.nodes):
            self.a[str(node)] = np.pi

        y = [Binary(f"{i}") for i in range(self.nodes)]

        qubo = self.formulate_qubo(y)

        model = qubo.compile()
        linear,quadratic,offset = model.to_ising()

        if self.use_linear_terms:
            linear_values = - np.array([*linear.values()])
        else:
            linear_values = np.zeros((self.nodes))

        state['linear_0'] = np.stack([[int(key) for key in linear.keys()], linear_values], axis=1)
        state['quadratic_0'] = np.stack([[int(key[0]), int(key[1]), value] for key, value in zip(quadratic.keys(),quadratic.values())])
        state['a_0'] = np.stack([[int(key), value] for key, value in zip(self.a.keys(),self.a.values())])

        return state, {}
    
    def step(self,action):

        reward = -1

        self.a[str(action)] = 0

        next_state = OrderedDict()

        done = False

        y = [Binary(f"{i}") for i in range(self.nodes)]

        qubo = self.formulate_qubo(y)

        model = qubo.compile()
        linear,quadratic,offset = model.to_ising()

        if self.use_linear_terms:
            linear_values = - np.array([*linear.values()])
        else:
            linear_values = np.zeros((self.nodes))

        next_state['linear_0'] = np.stack([[int(key) for key in linear.keys()], linear_values], axis=1)
        next_state['quadratic_0'] = np.stack([[int(key[0]), int(key[1]), value] for key, value in zip(quadratic.keys(),quadratic.values())])
        next_state['a_0'] = np.stack([[int(key), value] for key, value in zip(self.a.keys(),self.a.values())])

        self.timestep += 1

        if self.check_all_edges_covered():
            done = True
            self.timestep = 0
            for node in range(self.nodes):
                self.a[str(node)] = np.pi

        return next_state, reward, done, False, {}


#if __name__ == "__main__":
#    env_config = {
#        "nodes": 8,
#        "graph": [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [1, 4], [1, 5], [2, 4], [2, 6], [2, 7], [3, 7], [4, 5], [4, 7], [5, 6]],
#        "P": 12,
#        "use_linear_terms": True
#    }
#
#    env = MVCSEQUENTIAL(env_config)
#
#    for i in range(10):
#        state,_ = env.reset()
#        done = False
#        while not done:
#            action = 0
#            state, reward, done, _, _ = env.step(action)
#            print(reward)
#            if done == True:
#                break