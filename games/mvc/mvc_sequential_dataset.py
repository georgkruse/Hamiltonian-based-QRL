import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
from collections import OrderedDict
from pyqubo import Binary
import numpy as np
from docplex.mp.model import Model


class MVCSEQUENTIALDATASET(gym.Env):
    def __init__(self,env_config):
        self.config = env_config
        self.graph = self.config["graph"]
        self.nodes = self.config["nodes"]
        self.P = self.config["P"]
        self.use_linear_terms = self.config["use_linear_terms"]

        self.state = OrderedDict()
        self.timestep = np.random.randint(low = 0, high = len(self.graph))

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
        objective = 0

        for i in range(self.nodes):
            objective += y[i]

        edges_constraints = []

        for edge in self.graph[timestep]:
            node_0,node_1 = edge[0],edge[1]
            edges_constraints.append(self.P * (1 - y[node_0] - y[node_1] + y[node_0]*y[node_1]))
        
        QUBO = objective + sum(edges_constraints)
        return QUBO

    def minimum_vertex_cover(self,nodes,edges):

        model = Model(name = "MVC")

        vertex_vars = {node: model.binary_var(name = 'v{}'.format(node)) for node in range(nodes)}

        for edge in edges:
            model.add_constraint(
                vertex_vars[edge[0]] + vertex_vars[edge[1]] >= 1,
                ctname='edge_{}_{}'.format(edge[0],edge[1])
            )

        model.minimize(model.sum(vertex_vars.values()))

        return model

    def optimal_cost(self,timestep = None):
        if timestep == None:
            timestep = self.timestep
        
        edges = self.graph[timestep]
        nodes = self.nodes

        model = self.minimum_vertex_cover(nodes, edges)
        docplex_sol = model.solve()
        solution = ""
        for ii in model.iter_binary_vars():
            solution += str(int(np.round(docplex_sol.get_value(ii),1)))
        
        count = 0
        for bit in solution:
            if bit == "1":
                count += 1
        return count
    
    def check_all_edges_covered(self):
        covered_edges = []
        accepted_nodes = []
        for key in self.a:
            if self.a[key] == 0:
                accepted_nodes.append(int(key))
        for node in accepted_nodes:
            for edge in self.graph[self.timestep]:
                if node in edge:
                    if edge not in covered_edges:
                        covered_edges.append(edge)
        return len(covered_edges) == len(self.graph[self.timestep])
    
    def reset(self, seed = None, options = None,use_specific_timestep = False, timestep = 0):
        if use_specific_timestep:
            self.timestep = timestep
        else:
            self.timestep = np.random.randint(low = 0, high = len(self.graph))

        state = OrderedDict()

        for node in range(self.nodes):
            self.a[str(node)] = np.pi

        y = [Binary(f"{i}") for i in range(self.nodes)]

        qubo = self.formulate_qubo(y,self.timestep)

        model = qubo.compile()
        linear,quadratic,offset = model.to_ising()

        if self.use_linear_terms:
            linear_values = - np.array([*linear.values()])
        else:
            linear_values = np.zeros((self.nodes))

        state['linear_0'] = np.stack([[int(key) for key in linear.keys()], linear_values], axis=1)
        state['quadratic_0'] = np.stack([[int(key[0]), int(key[1]), value] for key, value in zip(quadratic.keys(),quadratic.values())])
        desired_rows = self.observation_space["quadratic_0"].shape[0]
        rows_to_pad = desired_rows - state["quadratic_0"].shape[0]
        state["quadratic_0"] = np.pad(state["quadratic_0"], ((0,rows_to_pad), (0,0)), mode = "constant")
        state['a_0'] = np.stack([[int(key), value] for key, value in zip(self.a.keys(),self.a.values())])

        return state, {}
    
    def step(self,action):

        reward = -1

        self.a[str(action)] = 0

        next_state = OrderedDict()

        done = False

        y = [Binary(f"{i}") for i in range(self.nodes)]

        qubo = self.formulate_qubo(y,self.timestep)

        model = qubo.compile()
        linear,quadratic,offset = model.to_ising()

        if self.use_linear_terms:
            linear_values = - np.array([*linear.values()])
        else:
            linear_values = np.zeros((self.nodes))

        next_state['linear_0'] = np.stack([[int(key) for key in linear.keys()], linear_values], axis=1)
        next_state['quadratic_0'] = np.stack([[int(key[0]), int(key[1]), value] for key, value in zip(quadratic.keys(),quadratic.values())])
        desired_rows = self.observation_space["quadratic_0"].shape[0]
        rows_to_pad = desired_rows - next_state["quadratic_0"].shape[0]
        next_state["quadratic_0"] = np.pad(next_state["quadratic_0"], ((0,rows_to_pad), (0,0)), mode = "constant")
        next_state['a_0'] = np.stack([[int(key), value] for key, value in zip(self.a.keys(),self.a.values())])

        if self.check_all_edges_covered():
            done = True
            self.timestep = np.random.randint(low = 0, high = len(self.graph))
            for node in range(self.nodes):
                self.a[str(node)] = np.pi

        return next_state, reward, done, False, {}


#if __name__ == "__main__":
#    env_config = {
#        "nodes": 5,
#        "graph": [[[0, 1], [0, 2], [0, 3], [1, 3], [1, 4], [3, 4]],[[0, 1], [0, 2], [1, 3], [1, 4], [2, 3], [3, 4]],[[0, 1], [0, 2], [1, 3], [2, 3], [2, 4], [3, 4]],[[0, 1], [0, 2], [0, 3], [0, 4], [1, 4], [2, 4]],[[0, 1], [0, 2], [0, 3], [0, 4], [2, 4], [3, 4]]],
#        "P": 8,
#        "use_linear_terms": True
#    }
#
#    env = MVCSEQUENTIALDATASET(env_config)
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