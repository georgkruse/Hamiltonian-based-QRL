import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
from collections import OrderedDict
from pyqubo import Binary
import numpy as np
from docplex.mp.model import Model


class MAXCUTDYNAMIC(gym.Env):
    def __init__(self,env_config):
        self.config = env_config
        self.graph = self.config["graph"]
        self.nodes = self.config["nodes"]

        self.state = OrderedDict()
        self.timestep = np.random.randint(low = 0, high = len(self.graph))
        self.cut_weight = 0

        self.a = {}
        for node in range(self.nodes):
            self.a[str(node)] = np.pi

        num_quadratic_terms = max([len(self.graph[idx]) for idx in range(len(self.graph))])

        spaces = {}

        spaces['linear_0'] = Box(-np.inf, np.inf, shape = (self.nodes,2), dtype='float64')
        spaces['quadratic_0'] = Box(-np.inf, np.inf, shape = (num_quadratic_terms,3), dtype='float64')
        spaces['a_0'] = Box(-np.inf, np.inf, shape = (self.nodes,2))

        self.observation_space = Dict(spaces = spaces)
        self.action_space = Discrete(self.nodes)

    def formulate_qubo(self,y,timestep):
        QUBO = 0

        for edge in self.graph[timestep]:
            node_1, node_2 = edge[0],edge[1]
            QUBO -= y[node_1] + y[node_2] - 2 * y[node_1] * y[node_2]

        return QUBO

    def optimal_cost(self,timestep = None):
        if timestep == None:
            timestep = self.timestep

        aux = []
        actions = np.reshape(np.unpackbits(np.arange(2**self.nodes).astype('>i8').view(np.uint8)), (-1, 64))[:,-self.nodes:]
        for action in actions:
            cost = self.formulate_qubo(action, timestep=timestep)
            aux.append(cost)
        
        return abs(np.min(aux))
    
    def calculate_cut_weight(self,timestep):
        cut_weight = 0
        accepted_nodes = []
        for key in self.a:
            if self.a[key] == 0:
                accepted_nodes.append(int(key))
        
        for edge in self.graph[timestep]:
            node1,node2 = edge

            if (node1 in accepted_nodes and node2 not in accepted_nodes) or (node1 not in accepted_nodes and node2 in accepted_nodes):
                cut_weight += 1
        
        change_cut_weight = cut_weight - self.cut_weight if cut_weight - self.cut_weight > 0 else 0

        return cut_weight < self.cut_weight, change_cut_weight, cut_weight
    
    def reset(self, seed = None, options = None,use_specific_timestep = False, timestep = 0):
        if use_specific_timestep:
            self.timestep = timestep
        else:
            self.timestep = np.random.randint(low = 0, high = len(self.graph))

        self.cut_weight = 0

        state = OrderedDict()

        for node in range(self.nodes):
            self.a[str(node)] = np.pi

        y = [Binary(f"{i}") for i in range(self.nodes)]

        qubo = self.formulate_qubo(y,self.timestep)

        model = qubo.compile()
        linear,quadratic,offset = model.to_ising()

        for node in range(self.nodes):
            linear[str(node)] = 0

        state['linear_0'] = np.stack([[int(key), value] for key, value in zip(linear.keys(),linear.values())])
        state['quadratic_0'] = np.stack([[int(key[0]), int(key[1]), value] for key, value in zip(quadratic.keys(),quadratic.values())])
        desired_rows = self.observation_space["quadratic_0"].shape[0]
        rows_to_pad = desired_rows - state["quadratic_0"].shape[0]
        state["quadratic_0"] = np.pad(state["quadratic_0"], ((0,rows_to_pad), (0,0)), mode = "constant")
        state['a_0'] = np.stack([[int(key), value] for key, value in zip(self.a.keys(),self.a.values())])

        return state, {}
    
    def step(self,action):

        self.a[str(action)] = 0

        next_state = OrderedDict()

        self.did_cut_not_change, change_in_cut_weight, self.cut_weight = self.calculate_cut_weight(timestep=self.timestep)

        reward = change_in_cut_weight

        done = False

        y = [Binary(f"{i}") for i in range(self.nodes)]

        qubo = self.formulate_qubo(y,self.timestep)

        model = qubo.compile()
        linear,quadratic,offset = model.to_ising()

        for node in range(self.nodes):
            linear[str(node)] = 0

        next_state['linear_0'] = np.stack([[int(key), value] for key, value in zip(linear.keys(),linear.values())])
        next_state['quadratic_0'] = np.stack([[int(key[0]), int(key[1]), value] for key, value in zip(quadratic.keys(),quadratic.values())])
        desired_rows = self.observation_space["quadratic_0"].shape[0]
        rows_to_pad = desired_rows - next_state["quadratic_0"].shape[0]
        next_state["quadratic_0"] = np.pad(next_state["quadratic_0"], ((0,rows_to_pad), (0,0)), mode = "constant")
        next_state['a_0'] = np.stack([[int(key), value] for key, value in zip(self.a.keys(),self.a.values())])

        if self.did_cut_not_change:
            done = True
            self.timestep = np.random.randint(low = 0, high = len(self.graph))
            for node in range(self.nodes):
                self.a[str(node)] = np.pi
            self.cut_weight = 0

        return next_state, reward, done, False, {}


#if __name__ == "__main__":
#    env_config = {
#        "nodes": 4,
#        "graph": [[[0,1],[0,3],[1,2],[2,3]],[[0,1],[0,2],[0,3],[2,3]]]
#    }
#
#    env = MAXCUTSEQUENTIALDATASET(env_config)
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