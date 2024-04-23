import numpy as np
from pyqubo import Binary
from collections import OrderedDict
import gymnasium as gym
from gymnasium.spaces import Box, Dict, MultiBinary,MultiDiscrete
import pennylane as qml

class EV_Game_QUBO_Bandit(gym.Env):

    def __init__(self,env_config):
        self.config = env_config
        self.n = self.config["number_cars"]
        self.k = self.config["number_timesteps"]
        self.u = np.array(self.config["u"]) # A list containing the u vectors of the self.n cars
        self.v = self.config["v"] # A list containing the v vectors of the self.n cars
        self.C = self.config["max_capacity"]
        self.P = self.config["max_power"]
        self.lambda_1 = self.config["lambda_1"]
        self.lambda_2 = self.config["lambda_2"]
        self.use_capacity_constraints = self.config["use_capacity_constraints"]
        self.use_power_constraints = self.config["use_power_constraints"]
        self.negative_reward_value = self.config["negative_reward_value"]
        self.normalization = self.config["normalization"]

        # cost -> reward based on the optimal cost
        # problem -> reward based on the problem
        self.reward_type = self.config["reward_type"]

        self.timestep = 0
        self.episodes = 0

        self.maxi = - np.inf
        self.mini = np.inf

        self.state = OrderedDict()

        y = [Binary(f"{i}") for i in range(self.n)]

        qubo = self.objective_ev_qubo(y, total = True)

        model = qubo.compile()
        linear,quadratic,offset = model.to_ising()
        self.linear, self.quadratic = linear, quadratic

        if len(linear.keys()) < self.n:
            for i in range(self.n):
                if str(i) not in linear.keys():
                    linear[str(i)] = 0

        linear_values = - np.array([*linear.values()])

        max_linear_values = max(linear_values)
        min_linear_values = min(linear_values)
        max_quadratic_values = max(quadratic.values())
        min_quadratic_values = min(quadratic.values())
        if max_linear_values > self.maxi:
            self.maxi = max_linear_values
        if max_quadratic_values > self.maxi:
            self.maxi = max_quadratic_values
        if min_linear_values < self.mini:
            self.mini = min_linear_values
        if min_quadratic_values < self.mini:
            self.mini = min_quadratic_values

        spaces = {}
        spaces['linear_0'] = Box(-np.inf, np.inf, shape = (self.n,2), dtype='float64')
        if self.use_capacity_constraints == False and self.use_power_constraints == False:
            spaces['quadratic_0'] = Box(-np.inf, np.inf, shape = (1,3), dtype='float64')
        else:
            spaces['quadratic_0'] = Box(-np.inf, np.inf, shape = (len(quadratic.values()),3), dtype='float64')
        
        self.observation_space = Dict(spaces = spaces)

        self.action_space = MultiDiscrete([2 for _ in range(self.n)])

        self.state['linear_0'] = np.stack([[int(key) for key in linear.keys()], linear_values], axis=1)
        if len(quadratic)>1:
            self.state['quadratic_0'] = np.stack([[int(key[0]), int(key[1]), value] for key, value in zip(quadratic.keys(),quadratic.values())])
        else:
            self.state['quadratic_0'] = np.array([[0,0,0]])

        aux = []
        self.optimal_costs = []
        actions = np.reshape(np.unpackbits(np.arange(2**self.n).astype('>i8').view(np.uint8)), (-1, 64))[:,-self.n:]
        for action in actions:
            cost = self.objective_ev_qubo(action,total=True)
            aux.append(cost)
        self.optimal_costs = np.min(aux)
        self.optimal_actions = actions[np.argmin(aux)]
        self.max_cost = np.max(aux)

    def reset(self, seed = None, options = None):
        self.timestep = 0
        state = OrderedDict()

        y = [Binary(f"{i}") for i in range(self.n)]

        qubo = self.objective_ev_qubo(y, total=True)

        model = qubo.compile()
        linear,quadratic,offset = model.to_ising()
        self.linear, self.quadratic = linear, quadratic

        if len(linear.keys()) < self.n:
            for i in range(self.n):
                if str(i) not in linear.keys():
                    linear[str(i)] = 0

        linear_values = - np.array([*linear.values()])

        linear_values -= self.mini
        linear_values = linear_values / (self.maxi - self.mini)
        
        if self.normalization == "2pi":
            linear_values *= 2*np.pi
        elif self.normalization == "pi/2":
            linear_values -= np.pi/2

        for key in quadratic:
            quadratic[key] -= self.mini
            quadratic[key] = quadratic[key] / (self.maxi - self.mini)
        
        if self.normalization == "2pi":
            quadratic[key] *= 2*np.pi
        elif self.normalization == "pi/2":
            quadratic[key] -= np.pi/2

        state['linear_0'] = np.stack([[int(key) for key in linear.keys()], linear_values], axis=1)
        if len(quadratic)>1:
            state['quadratic_0'] = np.stack([[int(key[0]), int(key[1]), value] for key, value in zip(quadratic.keys(),quadratic.values())])

            # Pad the array
            desired_rows = self.observation_space["quadratic_0"].shape[0]
            rows_to_pad = desired_rows - state["quadratic_0"].shape[0]
            state["quadratic_0"] = np.pad(state["quadratic_0"], ((0,rows_to_pad), (0,0)), mode = "constant")
        else:
            state['quadratic_0'] = np.array([[0,0,0]])

        return state, {}
    
    def step(self,action):

        self.timestep += 1

        next_state = OrderedDict()

        done = True
        
        y = [Binary(f"{i}") for i in range(self.n)]

        qubo = self.objective_ev_qubo(y, total=True)

        model = qubo.compile()
        linear,quadratic,offset = model.to_ising()
        self.linear, self.quadratic = linear, quadratic

        if len(linear.keys()) < self.n:
            for i in range(self.n):
                if str(i) not in linear.keys():
                    linear[str(i)] = 0

        linear_values = - np.array([*linear.values()])

        next_state["linear_0"] = np.stack([[int(key) for key in linear.keys()], linear_values], axis=1)
        if len(quadratic)>1:
            next_state["quadratic_0"] = np.stack([[int(key[0]), int(key[1]), value] for key, value in zip(quadratic.keys(),quadratic.values())])
            # Pad the array
            desired_rows = self.observation_space["quadratic_0"].shape[0]
            rows_to_pad = desired_rows - next_state["quadratic_0"].shape[0]
            next_state["quadratic_0"] = np.pad(next_state["quadratic_0"], ((0,rows_to_pad), (0,0)), mode = "constant")
        else:
            next_state["quadratic_0"] = np.array([[0,0,0]])

        self.timestep = 0
        if self.reward_type == "cost":
            reward = self.objective_ev_qubo(action,total=True)
            reward = reward - self.optimal_costs
            reward = -reward
        elif self.reward_type == "problem":
            reward = sum([action[i] * sum(self.v[i]) for i in range(self.n)])

            # Check if capacity_constraint was broken
            capacity_constraint_broken = False
            number_cars = 0
            for k in range(self.k):
                for i in range(self.n):
                    if self.u[i,k] != 0:
                        number_cars += action[i]
                if number_cars > self.C:
                    capacity_constraint_broken = True
                    pass
                else:
                    number_cars = 0

            power_constraint_broken = False
            power_consumed = 0
            for k in range(self.k):
                for i in range(self.n):
                    power_consumed += action[i] * self.v[i][k]
                if power_consumed > self.P:
                    power_constraint_broken = True
                    pass
                else:
                    power_consumed = 0
            
            if capacity_constraint_broken or power_constraint_broken:
                reward = - self.negative_reward_value

        return next_state, reward, done, False, {}

    def objective_ev_qubo(self,y,total = False):
        list_objs = []
        variables_timestep = []
        list_constraints_capacity = []
        list_constraints_power = []

        for i in range(self.n):
            list_objs.append(-sum(self.v[i]) * y[i])

        if total:
            EV_presence = np.array([i for i in range(len(self.u[self.timestep]))])
        else:
            #Create an array that contains the timesteps where the EV will be there to encode the constraints
            EV_presence = [index for index,value in enumerate(self.u[self.timestep]) if value == 1]

        if self.use_capacity_constraints:    
            for k in EV_presence:
                for i in range(self.n):
                    if self.u[i,k] != 0:
                        variables_timestep.append(y[i])
                list_constraints_capacity.append(variables_timestep)
                variables_timestep = []

            list_constraints_capacity = [self.C - np.sum(list_constraints_capacity[i]) for i in range(len(list_constraints_capacity))]
            list_constraints_capacity = [-self.lambda_1*list_constraints_capacity[i] + self.lambda_2*list_constraints_capacity[i]**2 for i in range(len(list_constraints_capacity))]

        if self.use_power_constraints:
            for k in EV_presence:
                for i in range(self.n):
                    variables_timestep.append(y[i]*self.v[i][k])
                list_constraints_power.append(variables_timestep)
                variables_timestep = []
        
            list_constraints_power = [self.P - np.sum(list_constraints_power[i]) for i in range(len(list_constraints_power))]
            list_constraints_power = [-self.lambda_1*list_constraints_power[i] + self.lambda_2*list_constraints_power[i]**2 for i in range(len(list_constraints_power))]

        H_Constraints_capacity = sum(list_constraints_capacity)
        H_Constraints_power = sum(list_constraints_power)
        H_Cost = sum(list_objs)

        QUBO = H_Cost + H_Constraints_capacity + H_Constraints_power
        return QUBO
    
    def get_ising(self):
        y = [Binary(f'{i}') for i in range(self.n)]
        qubo = self.objective_ev_qubo(y)
        model = qubo.compile()
        linear,quadratic,offset = model.to_ising()
        return linear, quadratic
    
    def get_hamiltonian(self):
        linear, quadratic = self.get_ising()
        H_linear = qml.Hamiltonian(
            [-linear[key] for key in linear.keys()],
            [qml.PauliZ(int(key)) for key in linear.keys()]
        )

        H_quadratic = qml.Hamiltonian(
            [quadratic[key] for key in quadratic.keys()],
            [qml.PauliZ(int(key[0]))@qml.PauliZ(int(key[1])) for key in quadratic.keys()]
        )

        H = H_quadratic - H_linear
        return H


#if __name__ == "__main__":
#    env_config = {
#        "number_cars": 5,
#        "number_timesteps": 5,
#        "u": [[1,1,0,0,1],[0,1,1,0,0],[1,0,1,0,0],[1,1,0,1,1],[0,0,1,1,1]],
#        "v": [[5,5,0,0,5],[0,10,10,0,0],[15,0,15,0,0],[5,5,0,5,5],[0,0,10,10,10]],
#        "max_capacity": 2,
#        "max_power": 20,
#        "lambda_1": 26.53,
#        "lambda_2": 24.47,
#        "use_capacity_constraints": True,
#        "use_power_constraints": False,
#        "reward_type": "cost"
#    }
#
#    env = EV_Game_QUBO_Bandit(env_config)
#
#    for i in range(10):
#        state,_ = env.reset()
#        done = False
#        while not done:
#            action = [0,1,0,1,0]
#            state, reward, done, _, _ = env.step(action)
#            print(reward)
#            if done == True:
#                break