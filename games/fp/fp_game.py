import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyqubo import Binary
import pennylane as qml
from copy import deepcopy
from collections import OrderedDict
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box, Dict, Discrete
from games.uc.uc_utils import kazarlis_uc, objective_uc_qubo

class FP_game(gym.Env):

    def __init__(self, env_config):

        self.config = env_config
        self.mode = env_config['mode']
        self.power_scaling = env_config['power_scaling']
        self.lambda_1 = env_config['lambda']
        self.action_space_type = env_config['action_space_type']
        self.episode_length = env_config['episode_length']
        self.reward_mode = env_config['reward_mode']
        self.units = env_config['units']
        self.n = self.units**2
       
        
        self.data_set = np.loadtxt(env_config['path'])
        index = np.random.randint(0, self.data_set.shape[1]-self.n)
        self.C = self.data_set[index:index+self.units,index:index+self.units]*0.01
        self.T = self.data_set[index+self.data_set.shape[1]:index+self.data_set.shape[1]+self.units,index:index+self.units]*0.01

        y = [Binary(f'{i}') for i in range(self.n)]
        qubo = self.objective_fp_qubo(y, self.units, self.C, self.T, self.lambda_1)
        model = qubo.compile()
        linear, quadratic, offset = model.to_ising()
        self.linear, self.quadratic = linear, quadratic

        spaces = {}
        spaces[f'linear_{0}'] = Box(-np.inf, np.inf, shape = (self.n,2), dtype='float64')
        spaces[f'quadratic_{0}'] = Box(-np.inf, np.inf, shape = (sum(range(self.n)),3), dtype='float64')
        self.observation_space = Dict(spaces=spaces)

        if self.action_space_type == 'multi_discrete':
            self.action_space = MultiDiscrete([self.units for _ in range(self.units)])
        elif self.action_space_type == 'discrete':
            self.action_space = Discrete(2**self.n)
        
        
        


    def reset(self, seed=None, options=None):
        self.history = []
        self.timestep = 0
        state = OrderedDict()
        self.optimal_costs, self.optimal_actions, self.optimal_action_index = [], [], []
        
        # Use Pyqubo to easily convert your QUBO to a Ising formulation
        y = [Binary(f'{i}') for i in range(self.n)]
        qubo = self.objective_fp_qubo(y, self.units, self.C, self.T, self.lambda_1)
        model = qubo.compile()
        linear, quadratic, offset = model.to_ising()

        self.linear = linear 
        self.quadratic = quadratic

        if len(quadratic.keys()) < sum(range(self.n)):
            for key0 in range(self.n):
                for key1 in range(self.n):
                    if key0 != key1:
                        if (str(key0), str(key1)) not in quadratic.keys():
                            if (str(key1), str(key0)) not in quadratic.keys():
                                quadratic[(str(key0), str(key1))] = 0

        state[f'linear_{0}'] = np.stack([[int(key), value] for (key, value) in linear.items()], axis=0)
        state[f'quadratic_{0}'] = np.stack([[int(key[0]), int(key[1]), value] for key, value in zip(quadratic.keys(),quadratic.values())])

        actions = np.reshape(np.unpackbits(np.arange(2**self.n).astype('>i8').view(np.uint8)), (-1, 64))[:,-self.n:]
        self.data = []
        for idx, action in enumerate(actions):
            cost = self.objective_fp_qubo(action, self.units, self.C, self.T, self.lambda_1)
            self.data.append(cost)
        self.optimal_costs.append(np.min(self.data))
        self.optimal_actions.append(actions[np.argmin(self.data)])
        self.optimal_action_index.append(np.argmin(self.data))

        return state, {}
    
    def step(self, raw_action):
        
        done = False
        next_state = OrderedDict()

        if isinstance(self.action_space, Discrete):
            binary_actions = np.reshape(np.unpackbits(np.arange(2**self.n).astype('>i8').view(np.uint8)), (-1, 64))[:,-self.n:]
            action = binary_actions[raw_action]
        elif isinstance(self.action_space, MultiDiscrete):
            action = np.zeros((self.units, self.units))
            action[range(len(raw_action)), raw_action] = 1.0
            action = np.reshape(action, (-1))

        reward = self.objective_fp_qubo(action, self.units, self.C, self.T, self.lambda_1)
      
        if self.config['reward_mode'] == 'optimal':
            reward = reward - self.optimal_costs[self.timestep]
            reward = - reward 
        elif self.config['reward_mode'] == 'optimal_spike':
            reward = reward - self.optimal_costs[self.timestep]
            if reward == 0:
                reward = 1
            else:
                reward = -1
        elif self.config['reward_mode'] == 'optimal_sqrt':
            reward = reward - self.optimal_costs[self.timestep]
            reward = - np.sqrt(reward) 
        elif self.config['reward_mode'] == 'optimal_log':
            reward = reward - self.optimal_costs[self.timestep] + 1e-9
            reward = - np.log(reward) 
            reward = np.min([5.0, reward]) 


        self.timestep +=1

        index = np.random.randint(0, self.data_set.shape[1]-self.n)
        self.C = self.data_set[index:index+self.units,index:index+self.units]*0.01
        self.T = self.data_set[index+self.data_set.shape[1]:index+self.data_set.shape[1]+self.units,index:index+self.units]*0.01
        # Use Pyqubo to easily convert your QUBO to a Ising formulation
        y = [Binary(f'{i}') for i in range(self.n)]
        qubo = self.objective_fp_qubo(y, self.units, self.C, self.T, self.lambda_1)
        model = qubo.compile()
        linear, quadratic, offset = model.to_ising()

        self.linear = linear 
        self.quadratic = quadratic

        if len(quadratic.keys()) < sum(range(self.n)):
            for key0 in range(self.n):
                for key1 in range(self.n):
                    if key0 != key1:
                        if (str(key0), str(key1)) not in quadratic.keys():
                            if (str(key1), str(key0)) not in quadratic.keys():
                                quadratic[(str(key0), str(key1))] = 0

        next_state[f'linear_{0}'] = np.stack([[int(key), value] for (key, value) in linear.items()], axis=0)
        next_state[f'quadratic_{0}'] = np.stack([[int(key[0]), int(key[1]), value] for key, value in zip(quadratic.keys(),quadratic.values())])

        actions = np.reshape(np.unpackbits(np.arange(2**self.n).astype('>i8').view(np.uint8)), (-1, 64))[:,-self.n:]
        self.data = []
        for idx, action in enumerate(actions):
            cost = self.objective_fp_qubo(action, self.units, self.C, self.T, self.lambda_1)
            self.data.append(cost)
        self.optimal_costs.append(np.min(self.data))
        self.optimal_actions.append(actions[np.argmin(self.data)])
        self.optimal_action_index.append(np.argmin(self.data))
        
        if self.timestep >= self.episode_length:
            done = True
            self.timestep = 0

        return next_state, reward, done, False, {}
    

            

    # calculate the objective/cost function of a bitstring
    def objective_fp_qubo(self, y, units, C, T, lambda_1):

        # result bitstring needs to be reversed (first qubit corrseponds to last bit in bitstring)
        obj_sum = 0
        
        # cost function
        for q in range(units):
            for p in range(units):
                for j in range(units):
                    for i in range(units):
                        Cij = C[i][j]
                        Tpq = T[p][q]
                        obj_sum += Cij*Tpq*y[units*p+i]*y[units*q+j]
        # now the constraints
        for i in range(units):
            p_sum = 0
            for p in range(units):
                p_sum += y[units*p+i]
            obj_sum += (lambda_1* (1-p_sum)**2)
        
        for p in range(units):
            i_sum = 0
            for i in range(units):
                i_sum += y[units*p+i]
            obj_sum += (lambda_1* (1-i_sum)**2)
            
        return (obj_sum)*self.power_scaling

    
    def get_hamiltonian(self, timestep):
        linear, quadratic = self.get_ising(timestep)
        H_linear = qml.Hamiltonian(
            [linear[key] for key in linear.keys()],
            [qml.PauliZ(int(key)) for key in linear.keys()]
        )

        H_quadratic = qml.Hamiltonian(
            [quadratic[key] for key in quadratic.keys()],
            [qml.PauliZ(int(key[0]))@qml.PauliZ(int(key[1])) for key in quadratic.keys()]
        )

        H = H_quadratic - H_linear
        return H
    
    def get_ising(self, timestep):
        y = [Binary(f'{i}') for i in range(self.n)]
        y = [Binary(f'{i}') for i in range(self.n)]
        qubo = self.objective_fp_qubo(y, self.units, self.C, self.T, self.lambda_1)
        model = qubo.compile()
        linear, quadratic, _ = model.to_ising()
        return linear, quadratic
    
    def objective_uc_qubo_docplex(self, A, B, C, Ls, lambda_1, n, p):
        
        doc_mdl = Model()
        # Create Variables
        objective = []
        UT = self.config['up_time']
        DT = self.config['down_time']
        u, v, w = [], [], []

        for t, L in enumerate(Ls):
            u.append([doc_mdl.binary_var(f"u_{g}_{t}") for g in range(n)])
            v.append([doc_mdl.binary_var(f"v_{g}_{t}") for g in range(n)])
            w.append([doc_mdl.binary_var(f"w_{g}_{t}") for g in range(n)])

        u = np.stack(u, axis=1)
        v = np.stack(v, axis=1)
        w = np.stack(w, axis=1)
        
        for t, L in enumerate(Ls):
            # Create Objective to minimize
            objective.append(doc_mdl.sum([(A[g] + B[g]*p[g] + C[g]*p[g]**2)*u[g][t] for g in range(n)]))
            # Create Constraint
            power_out = doc_mdl.sum([u[g][t]*p[g] for g in range(n)])
            doc_mdl.add_constraint(power_out >= L)
            if t > 0:
                for g in range(n):
                    doc_mdl.add_constraint(u[g][t] - u[g][t-1] == v[g][t] - w[g][t])

        for g in range(n):
            for t in range(UT[g],len(Ls)):
                gen_min_up_times = doc_mdl.sum([v[g][t_up] for t_up in range(t-UT[g]+1,t)])
                doc_mdl.add_constraint(gen_min_up_times <= u[g][t])

            for t in range(DT[g],len(Ls)):
                gen_min_down_times = doc_mdl.sum([w[g][t_down] for t_down in range(t-DT[g]+1,t)])
                doc_mdl.add_constraint(gen_min_down_times <= 1 - u[g][t])


        doc_mdl.minimize(doc_mdl.sum(objective))
        qp = from_docplex_mp(doc_mdl)
        return qp

    def objective_uc_qubo_docplex_cont(self, A, B, C, Ls, lambda_1, n, p):
        
        doc_mdl = Model()
        # Create Variables
        objective = []
        UT = self.config['up_time']
        DT = self.config['down_time']
        u, v, w, p = [], [], [], []

        for t, L in enumerate(Ls):
            u.append([doc_mdl.binary_var(f"u_{g}_{t}") for g in range(n)])
            v.append([doc_mdl.binary_var(f"v_{g}_{t}") for g in range(n)])
            w.append([doc_mdl.binary_var(f"w_{g}_{t}") for g in range(n)])
        
        for t, L in enumerate(Ls):
            p.append([[doc_mdl.continuous_var(lb=0.8, ub=1.4, name=f"p_{g}_{t}"), doc_mdl.continuous_var(lb=0.2, ub=0.6,name=f"p_{g+int(n/2)}_{t}")] for g in range(0,int(n/2))])
            # p.append([doc_mdl.continuous_var(lb=0.2, ub=0.6,name=f"p_{g}_{t}") for g in range(int(n/2), n)])

        p = np.reshape(p, (8, -1))
        u = np.stack(u, axis=1)
        v = np.stack(v, axis=1)
        w = np.stack(w, axis=1)
        
        for t, L in enumerate(Ls):
            # Create Objective to minimize
            objective.append(doc_mdl.sum([(A[g] + B[g]*p[g][t] + C[g]*p[g][t])*u[g][t] for g in range(n)]))
            # Create Constraint
            power_out = doc_mdl.sum([u[g][t]*p[g][t] for g in range(n)])
            doc_mdl.add_constraint(power_out >= L)
            if t > 0:
                for g in range(n):
                    doc_mdl.add_constraint(u[g][t] - u[g][t-1] == v[g][t] - w[g][t])

        for g in range(n):
            for t in range(UT[g],len(Ls)):
                gen_min_up_times = doc_mdl.sum([v[g][t_up] for t_up in range(t-UT[g]+1,t)])
                doc_mdl.add_constraint(gen_min_up_times <= u[g][t])

            for t in range(DT[g],len(Ls)):
                gen_min_down_times = doc_mdl.sum([w[g][t_down] for t_down in range(t-DT[g]+1,t)])
                doc_mdl.add_constraint(gen_min_down_times <= 1 - u[g][t])


        doc_mdl.minimize(doc_mdl.sum(objective))
        qp = from_docplex_mp(doc_mdl)
        return qp
        
    def classical_optimizer(self, qp):
      
        
        # qubo = self.objective_uc_qubo(y, self.A, self.B, self.C, self.time_series[self.timestep+idx], self.lambda_, self.n, self.generator_outputs)
        # define a problem
        
        # self.cplex_result = CplexOptimizer().solve(qp)
        self.gurobi_result = GurobiOptimizer().solve(qp)

        # print("cplex")
        # print(self.cplex_result.prettyprint())
        # print()
        print("gurobi")
        print(self.gurobi_result.prettyprint())
    
    def plot_result(self):



        x = np.arange(len(self.time_series))  # the label locations
        width = 0.1  # the width of the bars
        multiplier = 0

        fig, ax = plt.subplots(layout='constrained')
        x = []
        x_label = []
        for key, value in self.gurobi_result.variables_dict.items():
            if 'u' in key:
                offset = width * multiplier
                rects = ax.bar(offset, value*self.generator_outputs[int(key[2])], width, label=key)
                ax.bar_label(rects, padding=3)
                multiplier += 1
                x.append(offset)
                x_label.append(key)
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Power Generated')
        ax.set_title('Unit Commitment')
        ax.set_xticks(x, x_label)
        # ax.legend(loc='upper left', ncols=3)
        # ax.set_ylim(0, 250)
        plt.xticks(rotation=45)
        plt.savefig('uc_gorubi.png')
        print('done')

   

