import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
from collections import OrderedDict
from pyqubo import Binary
import numpy as np
from docplex.mp.model import Model
from copy import deepcopy
import pickle
from itertools import combinations



class MAXCUTWEIGHTEDDYNAMIC(gym.Env):
    def __init__(self,env_config):
        self.config = env_config
        self.nodes = self.config['nodes']
        self.reward_mode = self.config['reward_mode']
        self.constant = self.config['constant']
        self.reward_at_end = self.config['reward_at_end']
        self.linear_terms = self.config['linear_terms']
        if self.config['a_value'] == '2pi':
            self.a_value = 2*np.pi
        else:
            self.a_value = 0
        # if self.nodes == 5:
        #     file = open("/home/users/kruse/quantum-computing/QRL/games/maxcut/data/tsp_5_reduced_train.pickle",'rb')
        #     object_file = pickle.load(file)
        #     self.graph = object_file['x_train']
        # elif self.nodes == 10:
        #     file = open("/home/users/kruse/quantum-computing/QRL/games/maxcut/data/tsp_10_reduced_train.pickle",'rb')
        #     object_file = pickle.load(file)
        #     self.graph = object_file['x_train']
        # self.fully_connected_qubits = list(combinations(list(range(self.nodes)), 2))
        # graphs = []
        # for tsp_graph_nodes in self.graph:
        #     fully_connected_edges = []
        #     edge_weights_ix = []
        #     for edge in self.fully_connected_qubits:
        #         fully_connected_edges.append((tsp_graph_nodes[edge[0]], tsp_graph_nodes[edge[1]]))
        #         edge_distance = np.linalg.norm(
        #             np.asarray(tsp_graph_nodes[edge[0]]) - np.asarray(tsp_graph_nodes[edge[1]]))
        #         edge_weights_ix.append([edge[0], edge[1], edge_distance])
        #     graphs.append(edge_weights_ix)
        # with open('maxcut_10.txt', 'w') as outfile:
        #     for graph in graphs:
        #         np.savetxt(outfile, graph)
        if self.nodes == 5:
            path = "/home/users/kruse/quantum-computing/QRL/games/maxcut/data/maxcut_5.txt"
            self.graph = np.loadtxt(path).reshape((1000,10,3))[:100]        
        elif self.nodes == 10:
            path = "/home/users/kruse/quantum-computing/QRL/games/maxcut/data/maxcut_10.txt"
            self.graph = np.loadtxt(path).reshape((100,45,3)) 
        self.state = OrderedDict()
        self.timestep = np.random.randint(low = 0, high = len(self.graph))
        self.cut_weight = 0
        self.mode = self.config['mode']

        self.a = {}
        for node in range(self.nodes):
            self.a[str(node)] = np.pi

        num_quadratic_terms = max([len(self.graph[idx]) for idx in range(len(self.graph))])

        spaces = {}
        spaces['linear_0'] = Box(-np.inf, np.inf, shape = (self.nodes,2), dtype='float64')
        spaces['quadratic_0'] = Box(-np.inf, np.inf, shape = (num_quadratic_terms,3), dtype='float64')
       
        if self.config['observation_space'] == 'node_wise':
            spaces['annotations'] = Box(-np.inf, np.inf, shape = (self.nodes,2))
        
        spaces['edges'] = Box(-np.inf, np.inf, shape = (num_quadratic_terms,3))

        self.observation_space = Dict(spaces = spaces)

        if self.config['action_space'] == 'discrete_nodes':
            self.action_space = Discrete(self.nodes)
        elif self.config['action_space'] == 'discrete_edges':
            self.action_space = Discrete(sum(range(self.nodes)))
        elif self.config['action_space'] == 'multi_discrete':
            self.action_space = Discrete()

    def formulate_qubo(self,y,timestep):
        QUBO = 0

        for edge in self.graph[timestep]:
            node_1, node_2, weight = int(edge[0]), int(edge[1]), edge[2]
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
    
    def calculate_cut_weight(self,timestep):
        cut_weight = 0
        accepted_nodes = []
        for key in self.a:
            if self.a[key] == 0:
                accepted_nodes.append(int(key))
        
        for edge in self.graph[timestep]:
            node1, node2, weight = edge

            if (int(node1) in accepted_nodes and int(node2) not in accepted_nodes) or (int(node1) not in accepted_nodes and int(node2) in accepted_nodes):
                cut_weight += weight
        
        change_cut_weight = cut_weight - self.cut_weight if cut_weight - self.cut_weight > 0 else 0

        return cut_weight <= self.cut_weight, change_cut_weight, cut_weight
    
    def reset(self, seed = None, options = None,use_specific_timestep = False, timestep = 0):
        
        if use_specific_timestep:
            self.timestep = timestep
        elif self.constant:
            self.timestep = 0
        else:
            self.timestep = np.random.randint(low = 0, high = len(self.graph))
        
        self.cut_weight = 0
        self.current_graph = deepcopy(self.graph[timestep])

        self.c = {}
        for node in range(self.nodes):
            self.c[str(node)] = None
        state = OrderedDict()

        for node in range(self.nodes):
            self.a[str(node)] = np.pi

        linear = {}
        quadratic = {}

        for node in range(self.nodes):
            linear[str(node)] = 0
        
        for edge in self.graph[self.timestep]:
            node1, node2, weight = edge
            quadratic[(f'{int(node1)}', f'{int(node2)}')] = weight

        state['linear_0'] = np.stack([[int(key), value] for key, value in zip(linear.keys(),linear.values())])
        state['quadratic_0'] = np.stack([[int(key[0]), int(key[1]), value] for key, value in zip(quadratic.keys(),quadratic.values())])
        desired_rows = self.observation_space["quadratic_0"].shape[0]
        rows_to_pad = desired_rows - state["quadratic_0"].shape[0]
        state["quadratic_0"] = np.pad(state["quadratic_0"], ((0,rows_to_pad), (0,0)), mode = "constant")
        if self.config['observation_space'] == 'node_wise':
            state['annotations'] = np.stack([[int(key), np.pi if value == np.pi else self.a_value] for key, value in zip(self.a.keys(),self.a.values())])
        
        state['edges'] = deepcopy(state['quadratic_0'])
        self.edges = state['edges']
        
        return state, {}
    
    def step(self, action):
        
        next_state = OrderedDict()

        if self.config['action_space'] == 'discrete_nodes':
            self.a[str(action)] = 0
            self.cut_weight_prev = deepcopy(self.cut_weight)
            self.did_cut_not_change, change_in_cut_weight, self.cut_weight = self.calculate_cut_weight(timestep=self.timestep)
            
        elif self.config['action_space'] == 'discrete_edges':
            self.cut_weight_prev = deepcopy(self.cut_weight)
            change_in_cut_weight = 0
            node1, node2, weight = self.current_graph[action]
            node1 = int(node1)
            node2 = int(node2)
            if self.c[str(node1)] == None and self.c[str(node2)] == None:
            
                self.a[str(node1)] = 0
                self.did_cut_not_change, change_in_cut_weight_1, cut_weight_node1 = self.calculate_cut_weight(self.timestep)
                self.a[str(node1)] = np.pi
                self.a[str(node2)] = 0
                self.did_cut_not_change, change_in_cut_weight_2, cut_weight_node2 = self.calculate_cut_weight(self.timestep)

                if cut_weight_node2 >= cut_weight_node1:
                    self.c[str(node1)] = 0
                    self.c[str(node2)] = 1
                    self.cut_weight = cut_weight_node2
                    change_in_cut_weight = change_in_cut_weight_2
                else:
                    self.c[str(node1)] = 1
                    self.c[str(node2)] = 0
                    
                    self.cut_weight = cut_weight_node1
                    change_in_cut_weight = change_in_cut_weight_1
                    
                    self.a[str(node1)] = 0
                    self.a[str(node2)] = np.pi

            elif self.c[str(node1)] == 0 and self.c[str(node2)] == None:
                self.c[str(node2)] = 1
                self.a[str(node2)] = 0
                self.did_cut_not_change, change_in_cut_weight, self.cut_weight = self.calculate_cut_weight(self.timestep)

            elif self.c[str(node1)] == 1 and self.c[str(node2)] == None:
                self.c[str(node2)] = 0
                self.did_cut_not_change, change_in_cut_weight, self.cut_weight = self.calculate_cut_weight(self.timestep)
            elif self.c[str(node2)] == 0 and self.c[str(node1)] == None:
                self.c[str(node1)] = 1
                self.a[str(node1)] = 0
                self.did_cut_not_change, change_in_cut_weight, self.cut_weight = self.calculate_cut_weight(self.timestep)
            elif self.c[str(node2)] == 1 and self.c[str(node1)] == None:
                self.c[str(node1)] = 0
                self.did_cut_not_change, change_in_cut_weight, self.cut_weight = self.calculate_cut_weight(self.timestep)
            elif self.c[str(node1)] != None and self.c[str(node2)] != None:
                self.did_cut_not_change, change_in_cut_weight, self.cut_weight = self.calculate_cut_weight(self.timestep)
            else:
                print('Invalid action.')
            
            nodes_0, nodes_1 = [], []
            for key in self.c:
                if self.c[key] == 0:
                    nodes_0.append(int(key))
            for key in self.c:
                if self.c[key] == 1:
                    nodes_1.append(int(key))

            for idx, edge in enumerate(self.edges):
                if (edge[0] == node1 and edge[1] == node2) or (edge[1] == node1 and edge[0] == node2):
                    self.edges[idx] = [edge[0], edge[1], -1]
                
                if (edge[0] in nodes_0) and (edge[1] in nodes_0):
                    self.edges[idx] = [edge[0], edge[1], -1]
                if (edge[0] in nodes_1) and (edge[1] in nodes_1):
                    self.edges[idx] = [edge[0], edge[1], -1]

            done = True if None not in self.c.values() else False


       

        reward = 0
        done = False
        linear = {}
        quadratic = {}

        for node in range(self.nodes):
            linear[str(node)] = 0
        
        for edge in self.graph[self.timestep]:
            node1, node2, weight = edge
            quadratic[(f'{int(node1)}', f'{int(node2)}')] = weight

        next_state['linear_0'] = np.stack([[int(key), value] for key, value in zip(linear.keys(),linear.values())])
        next_state['quadratic_0'] = np.stack([[int(key[0]), int(key[1]), value] for key, value in zip(quadratic.keys(),quadratic.values())])
        desired_rows = self.observation_space["quadratic_0"].shape[0]
        rows_to_pad = desired_rows - next_state["quadratic_0"].shape[0]
        next_state["quadratic_0"] = np.pad(next_state["quadratic_0"], ((0,rows_to_pad), (0,0)), mode = "constant")
        if self.config['observation_space'] == 'node_wise':
            next_state['annotations'] = np.stack([[int(key), np.pi if value == np.pi else self.a_value] for key, value in zip(self.a.keys(),self.a.values())])
        next_state['edges'] = deepcopy(self.edges)

        if self.did_cut_not_change:
            done = True
            self.optimal_cost_ = self.optimal_cost(self.timestep)
            if self.constant:
                self.timestep = 0
            else:
                self.timestep = np.random.randint(low = 0, high = len(self.graph))
            
            for node in range(self.nodes):
                self.a[str(node)] = np.pi
            for node in range(self.nodes):
                self.c[str(node)] = None
            self.cut_weight = 0

            if self.reward_mode == 'ratio':
                reward = deepcopy(self.cut_weight_prev/self.optimal_cost_)
            else:
                reward = deepcopy(self.cut_weight_prev)
        
        if not self.reward_at_end:
            reward = change_in_cut_weight

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