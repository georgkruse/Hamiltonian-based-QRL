import random
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
from collections import OrderedDict
from pyqubo import Binary
import numpy as np
from docplex.mp.model import Model
from copy import deepcopy
import pickle
from itertools import combinations
import copy

def compute_tour_length(nodes, tour):
    """
    Compute length of a tour, including return to start node.
    (If start node is already added as last node in tour, 0 will be added to tour length.)
    :param nodes: all nodes in the graph in form of (x, y) coordinates
    :param tour: list of node indices denoting a (potentially partial) tour
    :return: tour length
    """
    tour_length = 0
    for i in range(len(tour)):
        if i < len(tour)-1:
            tour_length += np.linalg.norm(np.asarray(nodes[tour[i]]) - np.asarray(nodes[tour[i+1]]))
        else:
            tour_length += np.linalg.norm(np.asarray(nodes[tour[-1]]) - np.asarray(nodes[tour[0]]))

    return tour_length

class TSP(gym.Env):
    def __init__(self,env_config):
        self.config = env_config
        self.nodes = self.config['nodes']
        self.fully_connected_qubits = list(combinations(list(range(self.nodes)), 2))
         
        with open(env_config['path'] + '/games/tsp/data/tsp_5_reduced_train.pickle', 'rb') as file:
            self.data = pickle.load(file)

        self.data_x = self.data['x_train']
        self.data_y = self.data['y_train']

        self.state = OrderedDict()
        self.timestep = np.random.randint(low = 0, high = len(self.data_x[0]))
        self.cut_weight = 0
        self.mode = 'mode'

        self.a = {}
        for node in range(self.nodes):
            self.a[str(node)] = np.pi

        spaces = {}
        spaces['linear_0'] = Box(-np.inf, np.inf, shape = (self.nodes,2), dtype='float64')
        spaces['quadratic_0'] = Box(-np.inf, np.inf, shape = (len(self.fully_connected_qubits),3), dtype='float64')       
        # spaces['partial_tour'] = Box(-np.inf, np.inf, shape = (len(self.fully_connected_qubits),3), dtype='float64')      
        spaces['annotations'] = Box(-np.inf, np.inf, shape = (self.nodes,2), dtype='float64')
        spaces['current_node'] = Box(-np.inf, np.inf, shape = (1,), dtype='float64')

        self.observation_space = Dict(spaces = spaces)

        if self.config['action_space'] == 'discrete_nodes':
            self.action_space = Discrete(self.nodes)
        elif self.config['action_space'] == 'discrete_edges':
            self.action_space = Discrete(self.nodes-1)
        elif self.config['action_space'] == 'multi_discrete':
            self.action_space = Discrete()

    @staticmethod
    def graph_to_list(
            nodes, fully_connected_edges, edge_weights, available_nodes, node_to_qubit_map):
        vals = []
        for node in nodes:
            vals.append(int(node_to_qubit_map[node] in available_nodes) * np.pi)

        for edge in fully_connected_edges:
            vals.append(np.arctan(edge_weights[edge]))

        return vals
    
    @staticmethod
    def cost(nodes, tour):
        return -compute_tour_length(nodes, tour)

    def compute_reward(self, nodes, old_state, state):
        return self.cost(nodes, state) - self.cost(nodes, old_state)
    
    def reset(self, seed = None, options = None,use_specific_timestep = False, timestep = 0):
        
        instance_number = random.randint(0, len(self.data_x)-1)
        self.tsp_graph_nodes = self.data_x[instance_number]
        self.optimal_tour_length = compute_tour_length(self.tsp_graph_nodes, [int(x - 1) for x in self.data_y[instance_number][:-1]])
        self.node_to_qubit_map = {}
        for i, node in enumerate(self.tsp_graph_nodes):
            self.node_to_qubit_map[node] = i

        self.fully_connected_edges = []
        self.edge_weights = {}
        self.edge_weights_ix = {}
        for edge in self.fully_connected_qubits:
            self.fully_connected_edges.append((self.tsp_graph_nodes[edge[0]], self.tsp_graph_nodes[edge[1]]))
            edge_distance = np.linalg.norm(
                np.asarray(self.tsp_graph_nodes[edge[0]]) - np.asarray(self.tsp_graph_nodes[edge[1]]))
            self.edge_weights[(self.tsp_graph_nodes[edge[0]], self.tsp_graph_nodes[edge[1]])] = edge_distance
            self.edge_weights_ix[edge] = edge_distance

        self.tour = [0]  # w.l.o.g. we always start at city 0
        self.tour_edges = []
        self.step_rewards = []
        self.available_nodes = list(range(1, self.nodes))

        self.prev_tour = copy.deepcopy(self.tour)
        self.state_list = self.graph_to_list(
            self.tsp_graph_nodes, self.fully_connected_edges, self.edge_weights,
            self.available_nodes, self.node_to_qubit_map)

        
        state = OrderedDict()

        linear = {}

        for node in range(self.nodes):
            linear[str(node)] = 0

        self.partial_tour = np.stack([[int(self.fully_connected_qubits[idx][0]),
                                        int(self.fully_connected_qubits[idx][1]),
                                        0] for idx in range(len(self.fully_connected_qubits))])
        
        state['linear_0'] = np.stack([[int(key), value] for key, value in zip(linear.keys(),linear.values())])

        state['quadratic_0'] = np.stack([[int(self.fully_connected_qubits[idx][0]),
                                          int(self.fully_connected_qubits[idx][1]),
                                          self.state_list[idx+self.nodes]] for idx in range(len(self.fully_connected_qubits))])
        
        if self.config['action_space'] == 'discrete_nodes':
            state['current_node'] = np.array([-1])
        else:
            state['current_node'] = np.array([0])

        self.current_node = state['current_node']
        if self.config['annotation_type'] == 'node_wise':
            state['annotations'] = np.stack([[idx, self.state_list[idx]] for idx in range(self.nodes)])

        return deepcopy(state), {}
    
    def step(self, action):        
        
        # write back this to state:  self.get_action(state_list, available_nodes, tour_edges, edge_weights_ix)
        # q_vals = self.q_vals_from_expectations(self.tour_edges, edge_weights, expectations)[0]
        # if action == 0:
        #     print('start up')
        #     next_node = 1

        if self.config['action_space'] == 'discrete_edges':
            potential_edges = []
            for (node1, node2) in self.fully_connected_qubits:
                if self.current_node == node1:
                    potential_edges.append([node1, node2])
                elif self.current_node == node2:
                    potential_edges.append([node1, node2])
            (node1, node2) = potential_edges[action]
            if self.current_node == node1:
                next_node = node2
            elif self.current_node == node2:
                next_node = node1

        elif self.config['action_space'] == 'discrete_nodes':
            if action == 0:
                print('stupid ray start up correction')
                next_node = 1
            else:
                next_node = action

        self.tour.append(next_node)
        self.tour_edges.append((deepcopy(self.tour[-1]), next_node))
        remove_node_ix = self.available_nodes.index(next_node)
        del self.available_nodes[remove_node_ix]

        if len(self.tour) > 1:
            reward = self.compute_reward(self.tsp_graph_nodes, self.prev_tour, self.tour)
            self.step_rewards.append(reward)
            done = False if len(self.available_nodes) > 1 else True
            

        if len(self.available_nodes) == 1:
            self.tour.append(self.available_nodes[0])
            self.tour.append(self.tour[0])
            reward = self.compute_reward(self.tsp_graph_nodes, self.prev_tour, self.tour)
            self.step_rewards.append(reward)
        
        if done:
            tour_length = compute_tour_length(self.tsp_graph_nodes, self.tour)
            self.ratio = tour_length/self.optimal_tour_length

        self.prev_tour = copy.deepcopy(self.tour)
        
        self.state_list = self.graph_to_list(
            self.tsp_graph_nodes, self.fully_connected_edges, self.edge_weights,
            self.available_nodes, self.node_to_qubit_map)

        next_state = OrderedDict()

        linear = {}

        for node in range(self.nodes):
            linear[str(node)] = 0
        
        next_state['linear_0'] = np.stack([[int(key), value] for key, value in zip(linear.keys(),linear.values())])

        next_state['quadratic_0'] = np.stack([[int(self.fully_connected_qubits[idx][0]),
                                          int(self.fully_connected_qubits[idx][1]),
                                          self.state_list[idx+self.nodes]] for idx in range(len(self.fully_connected_qubits))])
        
        if self.config['action_space'] == 'discrete_nodes':
            next_state['current_node'] = np.array([-1])
        else:
            next_state['current_node'] = np.array([self.tour[-1]])

        self.current_node = next_state['current_node']
        if self.config['annotation_type'] == 'node_wise':
            next_state['annotations'] = np.stack([[idx, self.state_list[idx]] for idx in range(self.nodes)])

        return deepcopy(next_state), reward, done, False, {}