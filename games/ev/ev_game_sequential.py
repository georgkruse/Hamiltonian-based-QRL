import numpy as np
from pyqubo import Binary
from collections import OrderedDict
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete
import pennylane as qml
from itertools import product, combinations
from docplex.mp.model import Model

class EVSEQUENTIALQUBO(gym.Env):

    def __init__(self,env_config):
        self.config = env_config
        self.n = self.config["number_cars"]        # Number of cars at each timestep
        self.k = self.config["number_timesteps"]   # Number of timesteps
        self.u = np.array(self.config["u"])        # A list containing the u vectors of the self.n cars
        self.v = self.config["v"]                  # A list containing the v vectors of the self.n cars
        self.C = self.config["max_capacity"]      # Capacity for EV cars
        self.P = self.config["max_power"]      # Max power per timestep
        self.lambda_1 = self.config["lambda_1"]
        self.lambda_2 = self.config["lambda_2"]
        self.use_capacity_constraints = self.config["use_capacity_constraints"]
        self.use_power_constraints = self.config["use_power_constraints"]
        self.parked_cars = []
        self.reward_type = self.config["reward_type"]
        self.negative_reward_value = self.config["negative_reward_value"]

        self.C_qubo = [self.C for i in range(self.n)]
        self.P_qubo = [self.P for i in range(self.n)]

        self.timestep = 0
        self.episodes = 0

        self.state = OrderedDict()

        self.cars_accepted = []
        self.action = []

        quadratic_combinations = len(list(combinations([f"x_{i}" for i in range(self.n)],2)))

        spaces = {}
        spaces['linear_0'] = Box(-np.inf, np.inf, shape = (self.n,2), dtype='float64')
        if self.use_capacity_constraints == False and self.use_power_constraints == False:
            spaces['quadratic_0'] = Box(-np.inf, np.inf, shape = (1,3), dtype='float64')
        else:
            spaces['quadratic_0'] = Box(-np.inf, np.inf, shape = (quadratic_combinations,3), dtype='float64')
        
        self.observation_space = Dict(spaces = spaces)

        self.action_space = Discrete(2)

        self.episode_rewards, self.episode_actions = [],[]

        actions = np.reshape(np.unpackbits(np.arange(2**self.n).astype('>i8').view(np.uint8)), (-1, 64))[:,-self.n:]
        #actions = [[0,1,0,1,1]]
        for action in actions:
            reward = 0
            for timestep in range(self.n):
                reward += self.problem_reward(action[timestep],timestep = timestep)
            self.action = []

            self.episode_rewards.append(reward)
        
        self.optimal_reward = np.max(self.episode_rewards)
        self.optimal_action = actions[np.argmax(self.episode_rewards)]

    def update_C_and_P(self,timestep):

        accepted_car = (self.u[timestep],self.v[timestep])

        for i in range(self.k):
            self.C_qubo[i] -= accepted_car[0][i]
            self.P_qubo[i] -= accepted_car[1][i]
            self.C_qubo[i] = max(0,self.C_qubo[i])
            self.P_qubo[i] = max(0,self.P_qubo[i]) 
        

    def reset(self, seed = None, options = None):
        self.action = []
        self.timestep = 0
        state = OrderedDict()

        y = [Binary(f"{i}") for i in range(self.n)]

        qubo = self.objective_ev_qubo(y,range(0,self.k),self.C_qubo[self.timestep],self.P_qubo[self.timestep],timestep = self.timestep)

        model = qubo.compile()
        linear,quadratic,offset = model.to_ising()
        self.linear, self.quadratic = linear, quadratic

        if len(linear.keys()) < self.n:
            for i in range(self.n):
                if str(i) not in linear.keys():
                    linear[str(i)] = 0

        linear_values = - np.array([*linear.values()])

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
        
        if self.reward_type == "cost":
            reward = self.objective_ev_qubo(action, range(0,self.k),self.C[self.timestep],self.P[self.timestep],self.timestep)
            reward = reward - self.optimal_costs[self.timestep]
            reward = -reward
        elif self.reward_type == "problem":
            reward = self.problem_reward(action)
        
        if action == 1:
            self.update_C_and_P(self.timestep)

        self.timestep += 1

        next_state = OrderedDict()

        done = False
        
        y = [Binary(f"{i}") for i in range(self.n)]

        if self.timestep < len(self.u):
            qubo = self.objective_ev_qubo(y,range(0,self.k),self.C_qubo[self.timestep],self.P_qubo[self.timestep],self.timestep)
        else:
            qubo = self.objective_ev_qubo(y,range(0,self.k),self.C_qubo[self.timestep-1],self.P_qubo[self.timestep-1],self.timestep-1)

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

        if self.timestep >= len(self.u):
            done =  True
            self.action = []
            self.timestep = 0

        return next_state, reward, done, False, {}
    
    def problem_reward(self,action,timestep = None):
        if timestep == None:
            timestep = self.timestep

        self.action.append(action)
        
        reward = self.action[timestep] * sum(self.v[timestep])

        # Check if capacity_constraint was broken
        capacity_constraint_broken = False
        number_cars = 0
        for k in range(self.k):
            for i in range(len(self.action)):
                if self.u[i,k] != 0:
                    number_cars += self.action[i]
            if number_cars > self.C:
                capacity_constraint_broken = True
                pass
            else:
                number_cars = 0

        power_constraint_broken = False
        power_consumed = 0
        for k in range(self.k):
            for i in range(len(self.action)):
                power_consumed += self.action[i] * self.v[i][k]
            if power_consumed > self.P:
                power_constraint_broken = True
                pass
            else:
                power_consumed = 0

        if capacity_constraint_broken or power_constraint_broken:
            reward = 0
        
        return reward

    def objective_ev_qubo(self,y,timesteps_to_encode,C,P,timestep):
        list_objs = []
        variables_timestep = []
        list_constraints_capacity = []
        list_constraints_power = []

        for i in range(self.n):
            list_objs.append(-sum(self.v[i]) * y[i])

        if self.use_capacity_constraints:    
            for k in timesteps_to_encode:
                for i in range(self.n):
                    if self.u[i,k] != 0:
                        variables_timestep.append(y[i])
                list_constraints_capacity.append(variables_timestep)
                variables_timestep = []

            list_constraints_capacity = [C - np.sum(list_constraints_capacity[i]) for i in range(len(list_constraints_capacity))]
            list_constraints_capacity = [-self.lambda_1*list_constraints_capacity[i] + self.lambda_2*list_constraints_capacity[i]**2 for i in range(len(list_constraints_capacity))]

        if self.use_power_constraints:
            for k in timesteps_to_encode:
                for i in range(self.n):
                    variables_timestep.append(y[i]*self.v[i][k])
                list_constraints_power.append(variables_timestep)
                variables_timestep = []

            list_constraints_power = [P - np.sum(list_constraints_power[i]) for i in range(len(list_constraints_power))]
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
    
    def KP(self,timestep = None) -> Model:

        if timestep is None:
            timestep = self.timestep

        mdl = Model("2k-Knapsack")

        x = {
            i: mdl.binary_var(name=f"x_{i}") for i in range(len(self.u[timestep]))
        }  # variables that represent the items

        mdl.maximize(
            mdl.sum(sum(self.v[i]) * x[i] for i in x)
        )  # indicate the objective function

        for j in range(len(self.u[0])):
            mdl.add_constraint(
                mdl.sum(self.u[i][j] * x[i] for i in x) <= self.C
            )  # add  the constraint for knapsack

        for j in range(len(self.u[0])):
            mdl.add_constraint(
                mdl.sum(self.v[i][j]*x[i] for i in x) <= self.P
            )

        return mdl
    
    def solve_knapsack(self,mdl):
        docplex_sol = mdl.solve()
        solution = ""
        for ii in mdl.iter_binary_vars():
            solution += str(int(np.round(docplex_sol.get_value(ii),1)))
        return solution


#if __name__ == "__main__":
#    env_config = {
#        "number_cars": 5,
#        "number_timesteps": 5,
#        "u": [[1,1,0,0,1],[0,1,1,0,0],[1,0,1,0,0],[1,1,0,1,1],[0,0,1,1,1]],
#        "v": [[5,5,0,0,5],[0,10,10,0,0],[15,0,15,0,0],[5,5,0,5,5],[0,0,10,10,10]],
#        "max_capacity": 2,
#        "max_power": 20,
#        "lambda_1": 3.54,
#        "lambda_2": 0.21,
#        "use_capacity_constraints": True,
#        "use_power_constraints": True,
#        "reward_type": "problem",
#        "negative_reward_value": 50
#    }
#
#    env = EVSEQUENTIALQUBO(env_config)
#
#    for i in range(10):
#        state,_ = env.reset()
#        done = False
#        while not done:
#            action = np.random.randint(2)
#            state, reward, done, _, _ = env.step(action)
#            if done == True:
#                break