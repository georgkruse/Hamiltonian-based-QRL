import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyqubo import Binary
import pennylane as qml
from collections import OrderedDict
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box, Dict, Discrete
from games.uc.uc_utils import kazarlis_uc, objective_uc_qubo
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.algorithms import CplexOptimizer, GurobiOptimizer
from docplex.mp.model import Model
from qiskit_optimization.translators import from_docplex_mp, from_gurobipy

class UC_game(gym.Env):

    def __init__(self, env_config):

        self.config = env_config

        self.n = env_config['num_generators']
        self.power_scaling = env_config['power_scaling']
        p_min_org, p_max_org, A_org, B_org, C_org = kazarlis_uc(10, env_config['path']) 
        self.lambda_ = env_config['lambda']
        self.action_space_type = env_config['action_space_type']
        self.constraint_mode = env_config['constraint_mode']
        # self.num_stacked_timesteps = env_config['num_stacked_timesteps']
        self.episode_length = env_config['episode_length']

        self.norm = 1.0
        # Scaling input parameters according to power_scaling
        self.A = np.array(A_org)*self.norm
        self.B = np.array(B_org)*self.norm
        self.C = np.array(C_org)*self.norm
        self.p_min = p_min_org*self.norm
        self.p_max = p_max_org*self.norm

        self.timestep = 0
        self.episodes = 0
      
        if self.config['mode'] in ['static_5', 'dynamic_5']:
            rng = np.random.RandomState(5)
            self.episode_length = 10
            self.num_stacked_timesteps = 1
            self.generator_outputs = (self.p_max[::2] - self.p_min[::2])
            self.power_demands = rng.uniform(min(self.generator_outputs),sum(self.generator_outputs), 10)
            self.time_series = self.power_demands
            self.power_scaling = 0.000005

        elif self.config['mode'] in ['static_10', 'dynamic_10']:
            rng = np.random.RandomState(10)
            self.episode_length = 10
            self.num_stacked_timesteps = 1
            self.generator_outputs = (self.p_max - self.p_min)
            self.power_demands = rng.uniform(min(self.generator_outputs),sum(self.generator_outputs), 10)
            self.time_series = self.power_demands
            self.power_scaling = 0.000005

        elif self.config['mode'] in ['static_15', 'dynamic_15']:
            rng = np.random.RandomState(15)
            self.episode_length = 10
            self.num_stacked_timesteps = 1
            self.generator_outputs = np.concatenate([(self.p_max - self.p_min), (self.p_max[::2] - self.p_min[::2])])
            self.power_demands = rng.uniform(min(self.generator_outputs),sum(self.generator_outputs), 10)
            self.time_series = self.power_demands
            self.A = np.concatenate([self.A, self.A[::2]])
            self.B = np.concatenate([self.B, self.B[::2]])
            self.C = np.concatenate([self.C, self.C[::2]])
            self.power_scaling = 0.000005
        
        elif self.config['mode'] == 'dynamic':
            rng = np.random.RandomState(43)
            self.episode_length = 10
            self.num_stacked_timesteps = 1
            self.generator_outputs = np.random.uniform(self.p_min[0], self.p_max[0], self.n)
            self.power_demands = rng.uniform(min(self.generator_outputs),sum(self.generator_outputs), 10)
            self.time_series = self.power_demands
            if self.n > 10:
                self.A = np.concatenate([self.A, self.A])
                self.B = np.concatenate([self.B, self.B])
                self.C = np.concatenate([self.C, self.C])
        else:
            self.power_demands = np.array(self.power_demands)
            self.generator_outputs = np.array(self.generator_outputs)
            self.time_series = self.power_demands 

        # Use Pyqubo to easily convert your QUBO to a Ising formulation
        y = [Binary(f'{i}') for i in range(self.n)]
        qubo = self.objective_uc_qubo(y, self.A, self.B, self.C, self.power_demands[0], self.lambda_, self.n, self.generator_outputs)
        model = qubo.compile()
        linear, quadratic, offset = model.to_ising()
        self.linear, self.quadratic = linear, quadratic
       
        spaces = {}
        for idx in range(self.num_stacked_timesteps):
            spaces[f'linear_{idx}'] = Box(-np.inf, np.inf, shape = (self.n,2), dtype='float64')
            spaces[f'quadratic_{idx}'] = Box(-np.inf, np.inf, shape = (len(quadratic.values()),3), dtype='float64')
        self.observation_space = Dict(spaces=spaces)

        if self.action_space_type == 'multi_discrete':
            self.action_space = MultiDiscrete([2 for _ in range(env_config['num_generators'])])
        elif self.action_space_type == 'discrete':
            self.action_space = Discrete(2**env_config['num_generators'])
    
        self.optimal_costs, self.optimal_actions, self.optimal_action_indices = [], [], []
        self.optimal_costs_tmp, self.optimal_actions_tmp, self.optimal_action_indices_tmp = [], [], []

    def calculate_optimal_actions(self, timestep):
        # Calculate optimal action
        actions = np.reshape(np.unpackbits(np.arange(2**self.n).astype('>i8').view(np.uint8)), (-1, 64))[:,-self.n:]
        self.data = []
        for idx, action in enumerate(actions):
            cost = self.objective_uc_qubo(action, self.A, self.B, self.C, self.power_demands[timestep], self.lambda_, self.n, self.generator_outputs)
            self.data.append(cost)
        self.optimal_costs_tmp.append(np.min(self.data))
        self.optimal_actions_tmp.append(actions[np.argmin(self.data)])
        self.optimal_action_indices_tmp.append(np.argmin(self.data))

        return np.argmin(self.data) 

    def reset(self, seed=None, options=None):
        self.history = []
        self.timestep = 0
        self.episodes += 1
        state = OrderedDict()

        if self.config['mode'] in ['dynamic', 'simple']:
            self.time_series = np.random.choice(self.power_demands, self.episode_length + self.num_stacked_timesteps + 1)
        elif self.config['mode'] in ['static', 'static_5', 'static_10', 'static_15', 'simple']:
            self.time_series = np.concatenate([self.power_demands, self.power_demands])
            self.num_stacked_timesteps = 1
        elif self.config['mode'] in ['dynamic_5', 'dynamic_10', 'dynamic_15']:
            if self.config['mode'] == 'dynamic_5':
                self.generator_outputs = np.random.uniform(self.p_min[::2], self.p_max[::2])
            elif self.config['mode'] == 'dynamic_10':
                self.generator_outputs = np.random.uniform(self.p_min, self.p_max)
            elif self.config['mode'] == 'dynamic_15':
                self.generator_outputs = np.random.uniform(np.concatenate([self.p_min, self.p_min[::2]]),
                                                           np.concatenate([self.p_max, self.p_max[::2]]))
            else:                                           
                self.generator_outputs = np.random.uniform(self.p_min[0], self.p_max[0], self.n)

            self.power_demands = np.random.uniform(min(self.generator_outputs),sum(self.generator_outputs), self.episode_length)
            self.time_series = np.concatenate([self.power_demands, self.power_demands])
            self.num_stacked_timesteps = 1
            
        self.optimal_costs, self.optimal_actions = [], []

        actions = np.reshape(np.unpackbits(np.arange(2**self.n).astype('>i8').view(np.uint8)), (-1, 64))[:,-self.n:]
        for timestep, power_demand in enumerate(self.power_demands):
            data = []
            for idx, action in enumerate(actions):
                cost = self.objective_uc_qubo(action, self.A, self.B, self.C, power_demand, self.lambda_, self.n, self.generator_outputs)
                data.append(cost)
            self.optimal_costs.append(np.min(data))
            self.optimal_actions.append(actions[np.argmin(data)])
            self.optimal_action_indices.append(np.argmin(data))

        # Use Pyqubo to easily convert your QUBO to a Ising formulation
        for idx in range(self.num_stacked_timesteps):
            y = [Binary(f'{i}') for i in range(self.n)]
            qubo = self.objective_uc_qubo(y, self.A, self.B, self.C, self.time_series[self.timestep+idx], self.lambda_, self.n, self.generator_outputs)
            model = qubo.compile()
            linear, quadratic, offset = model.to_ising()
            if len(linear.keys()) < self.n:
                for i in range(self.n):
                    if str(i) not in linear.keys():
                        linear[str(i)] = 0
            # Add a minus to the linear term, because of the way pyqubo converts to ising models
            self.linear = linear 
            self.quadratic = quadratic
            linear_values = - np.array([*linear.values()])
            state[f'linear_{idx}'] = np.stack([[int(key) for key in linear.keys()], linear_values], axis=1)
            state[f'quadratic_{idx}'] = np.stack([[int(key[0]), int(key[1]), value] for key, value in zip(quadratic.keys(),quadratic.values())])

        self.generator_status = np.zeros(self.n)
        self.current_constraint = np.zeros(self.n)
        
        return state, {}
    
    def step(self, action):
        
        done = False
        next_state = OrderedDict()

        if isinstance(self.action_space, Discrete):
            binary_actions = np.reshape(np.unpackbits(np.arange(2**self.n).astype('>i8').view(np.uint8)), (-1, 64))[:,-self.n:]
            action = binary_actions[action]

        self.generator_status = action

        self.history.append(sum(self.generator_status*self.generator_outputs))
        reward = self.objective_uc_qubo(self.generator_status, self.A, self.B, self.C, self.time_series[self.timestep], self.lambda_, self.n, self.generator_outputs)
        
        self.actual_cost = reward

        if self.config['reward_mode'] == 'optimal':
            reward = - reward 
        elif self.config['reward_mode'] == 'optimal_sqrt':
            reward = - np.sqrt(reward) 
        elif self.config['reward_mode'] == 'optimal_log':
            reward = - np.log(reward)
            reward = np.min([1.0, reward]) 

        self.timestep +=1

        for idx in range(self.num_stacked_timesteps):
            y = [Binary(f'{i}') for i in range(self.n)]
            qubo = self.objective_uc_qubo(y, self.A, self.B, self.C, self.time_series[self.timestep+idx], self.lambda_, self.n, self.generator_outputs)
            model = qubo.compile()
            linear, quadratic, offset = model.to_ising()
            if len(linear.keys()) < self.n:
                for i in range(self.n):
                    if str(i) not in linear.keys():
                        linear[str(i)] = 0

            # Add a minus to the linear term, because of the way pyqubo converts to ising models
            self.linear = linear 
            self.quadratic = quadratic
            linear_values = - np.array([*linear.values()])
            next_state[f'linear_{idx}'] = np.stack([[int(key) for key in linear.keys()], linear_values], axis=1)
            next_state[f'quadratic_{idx}'] = np.stack([[int(key[0]), int(key[1]), value] for key, value in zip(quadratic.keys(),quadratic.values())])
        
        if self.timestep >= self.episode_length:
            done = True
            self.timestep = 0

        return next_state, reward, done, False, {}
    
    def objective_uc_qubo(self, y, A, B, C, L, lambda_1, n, constants, eq_bits= None):
      
        list_obj = []
        list_const = []
        for i in range(n):
            # Generating Bit-Flips with the term (y-1)**2
            list_obj.append((A[i] + B[i]*self.generator_outputs[i] + C[i]*self.generator_outputs[i]**2)*(y[i])**2)
            list_const.append(self.generator_outputs[i]*(y[i])**2)
        objective = sum(list_obj)
        constraint_1 = lambda_1*((sum(list_const) - L)**2)
        if self.constraint_mode == 'demand_only':
            QUBO = constraint_1
        elif self.constraint_mode == 'demand_constraint':
            QUBO = objective + constraint_1
        return QUBO*self.power_scaling

    def compute_reward(self, y, timestep):
      
        list_obj = []
        list_const = []
        for i in range(self.n):
            # Generating Bit-Flips with the term (y-1)**2
            list_obj.append((self.A[i] + self.B[i]*self.generator_outputs[i] + self.C[i]*self.generator_outputs[i]**2)*(y[i])**2)
            list_const.append(self.generator_outputs[i]*(y[i])**2)
        objective = sum(list_obj)
        constraint_1 = self.lambda_*((sum(list_const) - self.time_series[timestep])**2)
        if self.constraint_mode == 'demand_only':
            QUBO = constraint_1
        elif self.constraint_mode == 'demand_constraint':
            QUBO = objective + constraint_1
        reward = QUBO - self.optimal_costs[timestep]
        reward = - reward 
        return reward        
    
    def get_qp_docplex(self):
        return self.qp
    
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
        qubo = self.objective_uc_qubo(y, self.A, self.B, self.C, self.time_series[timestep], self.lambda_, self.n, self.generator_outputs)
        model = qubo.compile()
        linear, quadratic, _ = model.to_ising()
        return linear, quadratic
    
    def get_theta(self, timestep):
        y = [Binary(f'{i}') for i in range(self.n)]
        qubo = self.objective_uc_qubo(y, self.A, self.B, self.C, self.time_series[timestep], self.lambda_, self.n, self.generator_outputs)
        model = qubo.compile()
        linear, quadratic, _ = model.to_ising()
        theta = {}
        if len(linear.keys()) < self.n:
            for i in range(self.n):
                if str(i) not in linear.keys():
                    linear[str(i)] = 0

        linear_values = - np.array([*linear.values()])
        theta[f'linear_0'] = np.stack([[int(key) for key in linear.keys()], linear_values], axis=1)
        theta[f'quadratic_0'] = np.stack([[int(key[0]), int(key[1]), value] for key, value in zip(quadratic.keys(),quadratic.values())])
        return theta
    
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

   

