import numpy as np
from pyqubo import Binary
from collections import OrderedDict
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
import pennylane as qml
from itertools import product

class EV_Game_QUBO(gym.Env):

    def __init__(self,env_config):
        self.config = env_config
        self.n = self.config["number_cars"]        # Number of cars at each timestep
        self.k = self.config["number_timesteps"]   # Number of timesteps
        self.u = np.array(self.config["u"])        # A list containing the u vectors of the self.n cars
        self.v = self.config["v"]                  # A list containing the v vectors of the self.n cars
        self.C = self.config["max_capacity"]       # Capacity for EV cars
        self.P = self.config["max_power"]          # Max power per timestep
        self.lambda_1 = self.config["lambda_1"]
        self.lambda_2 = self.config["lambda_2"]
        self.use_capacity_constraints = self.config["use_capacity_constraints"]
        self.use_power_constraints = self.config["use_power_constraints"]
        self.parked_cars = []
        self.reward_type = self.config["reward_type"]

        self.timestep = 0
        self.episodes = 0

        self.state = OrderedDict()

        self.cars_accepted = []

        y = [Binary(f"{i}") for i in range(self.n)]

        qubos = []

        for i in range(len(self.u)):
            qubo = self.objective_ev_qubo(y, range(0,self.k),self.C[i],self.P[i],timestep = i)
            qubos.append(qubo)

        models = [qubo.compile() for qubo in qubos]
        linear, quadratic = [],[]

        for i in range(len(models)):
            linear_value,quadratic_value,offset_value = models[i].to_ising()
            linear.append(linear_value)
            quadratic.append(quadratic_value)
        
        max_quadratic_values = np.max([len(quadratic[i]) for i in range(len(quadratic))])

        spaces = {}
        spaces['linear_0'] = Box(-np.inf, np.inf, shape = (self.n,2), dtype='float64')
        if self.use_capacity_constraints == False and self.use_power_constraints == False:
            spaces['quadratic_0'] = Box(-np.inf, np.inf, shape = (1,3), dtype='float64')
        else:
            spaces['quadratic_0'] = Box(-np.inf, np.inf, shape = (max_quadratic_values,3), dtype='float64')
        
        self.observation_space = Dict(spaces = spaces)

        self.action_space = MultiDiscrete([2 for _ in range(self.n)])

        self.optimal_costs,self.optimal_actions = [],[]
        actions = np.reshape(np.unpackbits(np.arange(2**self.n).astype('>i8').view(np.uint8)), (-1, 64))[:,-self.n:]

        for timestep in range(len(self.u)):
            aux = []
            for i,action in enumerate(actions):
                cost = self.objective_ev_qubo(action,range(0, self.k),self.C[timestep],self.P[timestep],timestep = timestep)
                aux.append(cost)
            self.optimal_costs.append(np.min(aux))
            self.optimal_actions.append(actions[np.argmin(aux)])

    #def calculate_C_and_P(self,timestep,action):
#
    #    cars_parked_at_each_timestep = []
    #    power_consumed_at_each_timestep = []
#
    #    # Store the indexes of the accepted cars
    #    indexes_accepted_cars = [index for index,value in enumerate(action) if value == 1]
#
    #    # Update the list of parked cars with the newly accepted cars
    #    for index in indexes_accepted_cars:
    #        self.parked_cars.append([self.u[timestep][index],self.v[timestep][index]])
#
    #    # Update the capacity and the power at this timestep based on these cars
    #    for i,car in enumerate(self.parked_cars):
    #        cars_parked_at_this_timestep += car[0][timestep]
    #        power_consumed_at_this_timestep += car[1][timestep]
    #    
    #    self.current_C = self.C - cars_parked_at_this_timestep
    #    self.current_P = self.P - power_consumed_at_this_timestep
#
    #    return self.current_C, self.current_P
             

    def reset(self, seed = None, options = None):
        self.timestep = 0
        state = OrderedDict()

        y = [Binary(f"{i}") for i in range(self.n)]

        qubo = self.objective_ev_qubo(y,range(0,self.k),self.C[self.timestep],self.P[self.timestep],timestep = self.timestep)

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
            reward = sum([action[i] * sum(self.v[self.timestep][i]) for i in range(self.n)])

            # Check if capacity_constraint was broken
            capacity_constraint_broken = False
            number_cars = 0
            for k in range(self.k):
                for i in range(self.n):
                    if self.u[self.timestep,i,k] != 0:
                        number_cars += action[i]
                if number_cars > self.C[self.timestep]:
                    capacity_constraint_broken = True
                    pass
                else:
                    number_cars = 0

            power_constraint_broken = False
            power_consumed = 0
            for k in range(self.k):
                for i in range(self.n):
                    power_consumed += action[i] * self.v[self.timestep][i][k]
                if power_consumed > self.P[self.timestep]:
                    power_constraint_broken = True
                    pass
                else:
                    power_consumed = 0
            
            if capacity_constraint_broken or power_constraint_broken:
                reward = -1000

        self.timestep += 1

        next_state = OrderedDict()

        done = False
        
        y = [Binary(f"{i}") for i in range(self.n)]

        if self.timestep < len(self.u):
            qubo = self.objective_ev_qubo(y,range(0,self.k),self.C[self.timestep],self.P[self.timestep],self.timestep)
        else:
            qubo = self.objective_ev_qubo(y,range(0,self.k),self.C[self.timestep-1],self.P[self.timestep-1],self.timestep-1)

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
            self.timestep = 0

        return next_state, reward, done, False, {}

    def objective_ev_qubo(self,y,timesteps_to_encode,C,P,timestep):
        list_objs = []
        variables_timestep = []
        list_constraints_capacity = []
        list_constraints_power = []

        for i in range(self.n):
            list_objs.append(-sum(self.v[timestep][i]) * y[i])

        if self.use_capacity_constraints:    
            for k in timesteps_to_encode:
                for i in range(self.n):
                    if self.u[timestep,i,k] != 0:
                        variables_timestep.append(y[i])
                list_constraints_capacity.append(variables_timestep)
                variables_timestep = []

            list_constraints_capacity = [C - np.sum(list_constraints_capacity[i]) for i in range(len(list_constraints_capacity))]
            list_constraints_capacity = [-self.lambda_1*list_constraints_capacity[i] + self.lambda_2*list_constraints_capacity[i]**2 for i in range(len(list_constraints_capacity))]

        if self.use_power_constraints:
            for k in timesteps_to_encode:
                for i in range(self.n):
                    variables_timestep.append(y[i]*self.v[timestep][i][k])
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


if __name__ == "__main__":
    env_config = {
        "number_cars": 5,
        "number_timesteps": 5,
        "u": [[[1,1,0,0,1],[0,1,1,0,0],[1,0,1,0,0],[1,1,0,1,1],[0,0,1,1,1],[1,1,1,1,1]],
              [[1,1,1,1,1],[0,1,1,0,1],[0,0,1,1,1],[1,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1]]],
        "v": [[[5,5,0,0,5],[0,10,10,0,0],[15,0,15,0,0],[5,5,0,5,5],[0,0,10,10,10],[1,1,1,1,1]],
              [[10,5,2,7,15],[0,18,3,0,15],[0,0,7,7,7],[25,0,10,0,0],[0,5,10,10,0],[1,1,1,1,1]]],
        "max_capacity": [2,2],
        "max_power": [20,25],
        "lambda_1": 6.18,
        "lambda_2": 8.54,
        "use_capacity_constraints": True,
        "use_power_constraints": True,
        "reward_type": "problem"
    }

    env = EV_Game_QUBO(env_config)

    actions = np.reshape(np.unpackbits(np.arange(2**5).astype('>i8').view(np.uint8)), (-1, 64))[:,-5:]

    state,_ = env.reset()
    done = False
    while not done:
        action = actions[np.random.choice(actions.shape[0])]
        state, reward, done, _, _ = env.step(action)
        if done == True:
            break