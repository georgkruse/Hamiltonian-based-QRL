import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
from collections import OrderedDict
from pyqubo import Binary
import numpy as np


class MAXCUTCONTEXTUALBANDIT(gym.Env):
    def __init__(self,env_config):
        self.config = env_config
        self.graph = self.config["graph"]
        self.nodes = self.config["nodes"]

        self.state = OrderedDict()
        self.timestep = 0

        max_quadratic_values = np.max([len(self.graph[i]) for i in range(len(self.graph))])

        spaces = {}
        spaces['linear_0'] = Box(-np.inf, np.inf, shape = (self.nodes,2), dtype='float64')
        spaces['quadratic_0'] = Box(-np.inf, np.inf, shape = (max_quadratic_values,3), dtype='float64')

        self.observation_space = Dict(spaces = spaces)
        self.action_space = MultiDiscrete([2 for _ in range(self.nodes)])

        self.optimal_costs,self.optimal_actions = [],[]
        actions = np.reshape(np.unpackbits(np.arange(2**self.nodes).astype('>i8').view(np.uint8)), (-1, 64))[:,-self.nodes:]

        for timestep in range(len(self.graph)):
            aux = []
            for i,action in enumerate(actions):
                cost = self.formulate_qubo(action,timestep)
                aux.append(cost)
            self.optimal_costs.append(np.min(aux))
            self.optimal_actions.append(actions[np.argmin(aux)])

    def formulate_qubo(self,y,timestep):
        QUBO = 0

        for edge in self.graph[timestep]:
            node_1, node_2 = edge[0],edge[1]
            QUBO -= y[node_1] + y[node_2] - 2 * y[node_1] * y[node_2]

        return QUBO
    
    def reset(self, seed = None, options = None, use_specific_timestep = False, timestep = 0):
        
        if use_specific_timestep:
            self.timestep = timestep
        else:
            self.timestep = np.random.randint(low = 0, high = len(self.graph))

        state = OrderedDict()
        
        y = [Binary(f"{i}") for i in range(self.nodes)]

        qubo = self.formulate_qubo(y,timestep = self.timestep)

        model = qubo.compile()
        linear,quadratic,offset = model.to_ising()
        for i in range(self.nodes):
            linear[str(i)] = 0
        linear_values = - np.array([*linear.values()])

        state['linear_0'] = np.stack([[int(key) for key in linear.keys()], linear_values], axis=1)
        state['quadratic_0'] = np.stack([[int(key[0]), int(key[1]), value] for key, value in zip(quadratic.keys(),quadratic.values())])
        desired_rows = self.observation_space["quadratic_0"].shape[0]
        rows_to_pad = desired_rows - state["quadratic_0"].shape[0]
        state["quadratic_0"] = np.pad(state["quadratic_0"], ((0,rows_to_pad), (0,0)), mode = "constant")

        return state, {}
    
    def step(self,action):
        next_state = OrderedDict()

        done = True

        y = [Binary(f"{i}") for i in range(self.nodes)]

        qubo = self.formulate_qubo(y,timestep = self.timestep)

        model = qubo.compile()
        linear,quadratic,offset = model.to_ising()
        for i in range(self.nodes):
            linear[str(i)] = 0
        linear_values = - np.array([*linear.values()])

        next_state['linear_0'] = np.stack([[int(key) for key in linear.keys()], linear_values], axis=1)
        next_state['quadratic_0'] = np.stack([[int(key[0]), int(key[1]), value] for key, value in zip(quadratic.keys(),quadratic.values())])
        desired_rows = self.observation_space["quadratic_0"].shape[0]
        rows_to_pad = desired_rows - next_state["quadratic_0"].shape[0]
        next_state["quadratic_0"] = np.pad(next_state["quadratic_0"], ((0,rows_to_pad), (0,0)), mode = "constant")

        reward = self.formulate_qubo(action,self.timestep)
        reward -= self.optimal_costs[self.timestep]
        reward = -reward

        self.timestep = np.random.randint(low = 0, high = len(self.graph))

        return next_state, reward, done, False, {}


if __name__ == "__main__":
    env_config = {
        "nodes": 5,
        "graph": [[(0,1),(0,4),(1,2),(2,3),(3,4)],
                  [(0,1),(0,2),(0,3),(0,4),(2,3),(2,4)],
                  [(0,1),(0,2),(0,3),(1,3),(1,4),(3,4)],
                  [(0,1),(0,2),(0,3),(0,4)],
                  [(0,1),(0,2),(0,3),(0,4)]]
    }

    env = MAXCUTCONTEXTUALBANDIT(env_config)

    for i in range(10):
        state,_ = env.reset()
        done = False
        while not done:
            action = [0,1,0,1,0]
            state, reward, done, _, _ = env.step(action)
            print(reward)
            if done == True:
                break