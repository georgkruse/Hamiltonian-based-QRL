import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete
from collections import OrderedDict
from pyqubo import Binary
import numpy as np
from docplex.mp.model import Model


class MAXCUTWEIGHTED(gym.Env):
    def __init__(self,env_config):
        self.config = env_config
        self.graph = self.config["graph"]
        self.nodes = self.config["nodes"]

        self.state = OrderedDict()
        self.timestep = 0
        self.cut_weight = 0

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
        QUBO = 0

        for edge in self.graph:
            node_1, node_2, weight = edge[0],edge[1],edge[2]
            QUBO -= weight * (y[node_1] + y[node_2] - 2 * y[node_1] * y[node_2])

        return QUBO

    def optimal_cost(self,timestep = None):

        aux = []
        actions = np.reshape(np.unpackbits(np.arange(2**self.nodes).astype('>i8').view(np.uint8)), (-1, 64))[:,-self.nodes:]
        for action in actions:
            cost = self.formulate_qubo(action)
            aux.append(cost)
        
        return abs(np.min(aux))
    
    def calculate_cut_weight(self):
        cut_weight = 0
        accepted_nodes = []
        for key in self.a:
            if self.a[key] == 0:
                accepted_nodes.append(int(key))
        
        for edge in self.graph:
            node1,node2,weight = edge

            if (node1 in accepted_nodes and node2 not in accepted_nodes) or (node1 not in accepted_nodes and node2 in accepted_nodes):
                cut_weight += weight
        
        change_cut_weight = cut_weight - self.cut_weight if cut_weight - self.cut_weight > 0 else 0

        return cut_weight <= self.cut_weight, change_cut_weight, cut_weight
    
    def reset(self, seed = None, options = None):
        self.timestep = 0
        state = OrderedDict()
        self.cut_weight = 0

        for node in range(self.nodes):
            self.a[str(node)] = np.pi

        linear = {}
        quadratic = {}

        for node in range(self.nodes):
            linear[str(node)] = 0
        
        for edge in self.graph:
            node1, node2, weight = edge
            quadratic[(f'{node1}', f'{node2}')] = weight

        state['linear_0'] = np.stack([[int(key), value] for key, value in zip(linear.keys(),linear.values())])
        state['quadratic_0'] = np.stack([[int(key[0]), int(key[1]), value] for key, value in zip(quadratic.keys(),quadratic.values())])
        state['a_0'] = np.stack([[int(key), value] for key, value in zip(self.a.keys(),self.a.values())])

        return state, {}
    
    def step(self,action):

        self.a[str(action)] = 0

        self.did_cut_not_change, change_in_cut_weight, self.cut_weight = self.calculate_cut_weight()

        reward = change_in_cut_weight

        next_state = OrderedDict()

        done = False

        linear = {}
        quadratic = {}

        for node in range(self.nodes):
            linear[str(node)] = 0
        
        for edge in self.graph:
            node1, node2, weight = edge
            quadratic[(f'{node1}', f'{node2}')] = weight

        next_state['linear_0'] = np.stack([[int(key), value] for key, value in zip(linear.keys(),linear.values())])
        next_state['quadratic_0'] = np.stack([[int(key[0]), int(key[1]), value] for key, value in zip(quadratic.keys(),quadratic.values())])
        next_state['a_0'] = np.stack([[int(key), value] for key, value in zip(self.a.keys(),self.a.values())])

        self.timestep += 1

        if self.did_cut_not_change:
            done = True
            self.timestep = 0
            for node in range(self.nodes):
                self.a[str(node)] = np.pi
            self.cut_weight = 0

        return next_state, reward, done, False, {}


if __name__ == "__main__":
    env_config = {
        "nodes": 4,
        "graph": [[0,1,5],[0,3,10],[1,2,7],[2,3,4]]
    }

    env = MAXCUTWEIGHTED(env_config)

    for i in range(10):
        state,_ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = np.random.randint(low = 0, high = 5)
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            if done == True:
                print(total_reward)
                break