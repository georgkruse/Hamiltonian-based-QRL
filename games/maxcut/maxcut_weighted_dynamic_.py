import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
from collections import OrderedDict
from pyqubo import Binary
import numpy as np
from copy import deepcopy
from docplex.mp.model import Model


class MAXCUTWEIGHTEDDYNAMIC(gym.Env):
    def __init__(self,env_config):
        self.config = env_config
        self.graph = self.config["graph"]
        self.nodes = self.config["nodes"]
        self.constant = self.config["constant"]

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
            node_1, node_2, weight = edge[0],edge[1], edge[2]
            QUBO -= weight * (y[node_1] + y[node_2] - 2 * y[node_1] * y[node_2])

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
    
    def calculate_cut_weight(self,timestep, action):

        self.nodes = [i for i in range(5)]
        self.edges = []
        self.cutted_edges = []
        for i in range(5):
            for j in range(5):
                if i > j:
                    self.edges.append([i,j, -1, 0.25])

        self.current_edge = 0
        for idx, edge in enumerate(self.edges):
            if edge[0] != self.current_edge and edge[1] != self.current_edge:
                self.edges[idx][3] = - 100
        action = np.argmax(self.edges[2])

        self.cutted_edges.append(action)
        cut_weight = 0
        accepted_nodes = []
        
        # if self.current_node != -1:
        #     if self.current_node in self.not_cut
        
        if self.c[action] != -1:
            node1, node2, weight = self.graph[timestep][action]
            self.current_node = node2



        # for key in self.a:
        #     if self.a[key] == 2*np.pi:
        #         accepted_nodes.append(int(key))
        
        # for edge in self.graph[timestep]:
        #     node1, node2, weight = edge

        #     if (node1 in accepted_nodes and node2 not in accepted_nodes) or (node1 not in accepted_nodes and node2 in accepted_nodes):
        #         cut_weight += weight
        
        # change_cut_weight = cut_weight - self.cut_weight if cut_weight - self.cut_weight > 0 else 0

        # return cut_weight <= self.cut_weight, change_cut_weight, cut_weight
    
    def reset(self, seed = None, options = None,use_specific_timestep = False, timestep = 0):
        if use_specific_timestep:
            self.timestep = timestep
        else:
            self.timestep = np.random.randint(low = 0, high = len(self.graph))
        if self.constant:
            self.timestep = 0 # np.random.randint(low = 0, high = 2)
        self.cut_weight = 0

        state = OrderedDict()

        for node in range(self.nodes):
            self.a[str(node)] = np.pi

        linear = {}
        quadratic = {}

        for node in range(self.nodes):
            linear[str(node)] = 0
        
        for edge in self.graph[self.timestep]:
            node1, node2, weight = edge
            quadratic[(f'{node1}', f'{node2}')] = weight

        state['linear_0'] = np.stack([[int(key), value] for key, value in zip(linear.keys(),linear.values())])
        state['quadratic_0'] = np.stack([[int(key[0]), int(key[1]), value] for key, value in zip(quadratic.keys(),quadratic.values())])
        desired_rows = self.observation_space["quadratic_0"].shape[0]
        rows_to_pad = desired_rows - state["quadratic_0"].shape[0]
        state["quadratic_0"] = np.pad(state["quadratic_0"], ((0,rows_to_pad), (0,0)), mode = "constant")
        state['a_0'] = np.stack([[int(key), value] for key, value in zip(self.a.keys(),self.a.values())])

        self.not_cut = state["quadratic_0"]
        # state['cuts'] = []
        self.g = {}
        for node in range(self.nodes):
            self.g[str(node)] = -1
        self.c = {}
        for edge in range(self.graph[self.timestep]):
            self.c[str(edge)] = -1
                
    
        return state, {}
    
    def step(self, action):
        
        # self.a[str(action)] = 2*np.pi
        self.not_cut
        next_state = OrderedDict()

        self.did_cut_not_change, change_in_cut_weight, self.cut_weight = self.calculate_cut_weight(self.timestep, action)


        reward = change_in_cut_weight
        done = False

        linear = {}
        quadratic = {}

        for node in range(self.nodes):
            linear[str(node)] = 0
        
        for edge in self.graph[self.timestep]:
            node1, node2, weight = edge
            quadratic[(f'{node1}', f'{node2}')] = weight

        next_state['linear_0'] = np.stack([[int(key), value] for key, value in zip(linear.keys(),linear.values())])
        next_state['quadratic_0'] = np.stack([[int(key[0]), int(key[1]), value] for key, value in zip(quadratic.keys(),quadratic.values())])
        desired_rows = self.observation_space["quadratic_0"].shape[0]
        rows_to_pad = desired_rows - next_state["quadratic_0"].shape[0]
        next_state["quadratic_0"] = np.pad(next_state["quadratic_0"], ((0,rows_to_pad), (0,0)), mode = "constant")
        next_state['a_0'] = np.stack([[int(key), value] for key, value in zip(self.a.keys(),self.a.values())])

        if self.did_cut_not_change:
            done = True
            self.timestep = np.random.randint(low = 0, high = len(self.graph))
            if self.constant:
                self.timestep = 0 #np.random.randint(low = 0, high = 2)
            for node in range(self.nodes):
                self.a[str(node)] = np.pi
            self.cut_weight = 0
        

        return next_state, reward, done, False, {}


#if __name__ == "__main__":
#    env_config = {
#        "nodes": 5,
#        "graph": [[[0, 1, 2.1551119149511924], [0, 3, 3.322649139999425], [0, 2, 5.571739281202778], [1, 5, 5.2037381841714705], [1, 4, 0.6621322959701119], [3, 4, 3.0497920493284267], [3, 2, 5.788103492964022], [4, 5, 3.237548617449324], [5, 2, 5.26116362855287]], [[0, 1, 2.095319077662616], [0, 3, 2.0341311984079207], [0, 2, 0.7113576191369215], [1, 5, 5.113868421313215], [1, 4, 4.736083602071193], [2, 4, 3.148723166358647], [2, 5, 4.945166875390806], [4, 3, 0.5181850498933085], [3, 5, 3.2445528958808083]]]
#    }
#
#    env = MAXCUTWEIGHTEDDYNAMIC(env_config)
#
#    for i in range(10):
#        state,_ = env.reset()
#        done = False
#        env.optimal_cost(env.timestep)
#        while not done:
#            action = 0
#            state, reward, done, _, _ = env.step(action)
#            print(reward)
#            if done == True:
#                break