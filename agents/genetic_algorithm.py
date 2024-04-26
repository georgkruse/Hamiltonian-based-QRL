from ray import tune 
import numpy as np 
import pennylane as qml 
import torch
from gymnasium.spaces import Box, Discrete

import ray
from collections import namedtuple
from copy import deepcopy 

import os
import numpy as np
import matplotlib.pyplot as plt    

# Take care, this has not been maintained in a while.


class Genetic_Algorithm(tune.Trainable):

    def setup(self, config):

        if isinstance(config['env'], str):        
            self.env = gym.make(config['env'], render_mode='rgb_array')
        else:
            self.env = config['env']({}) #(config['env_config'])

        self.config = config
        self.seed = config['seed']
        self.lr = config['lr']
        self.lr_decay_factor = config['lr_decay_factor']
        self.population_size = config['population_size']
        self.evaluate_num = config['evaluate_num']
        self.sigma = config['sigma']
        self.num_layers = int(config['num_layers'])
        self.l = config['l']
        self.qubits = self.env.observation_space.shape[0]*self.l
        self.config['qubits'] = self.qubits
        self.init_params = config['init_params']
        self.vqc_type = config['vqc_type']
        self.use_input_scaling = config['use_input_scaling']
        self.use_output_scaling_actor = config['use_output_scaling_actor']
        self.output_scaling_actor = config['output_scaling_actor']
        self.init_params_mode = config['init_params_mode']
        self.action_type = config['action_type']
        self.elite_num = config['elite_num']
        self.evaluate_num_elite = config['evaluate_num_elite']
        
        self.mode = config['mode']
        
        if self.vqc_type == 'rot_circuit':
            self.num_params = 3
        elif self.vqc_type == 'circuit_6':
            self.num_params = 5 + self.qubits
        else:
            self.num_params = 2

        self.t_step = 0
        self.agents = []
        self.circuit_evaluations = 0

        self.init_population()
        self.weight_space = self.agents[0].shape[0] 

    def init_population(self):

        if self.mode == 'classical':
            
            self.agents = []
            state = np.zeros((1, self.qubits),'float32')
            self.model = create_neural_network(self.seed)
            action = self.model(state)
            self.config['model'] = self.model

            for i in range(self.population_size):

                self.model = create_neural_network(self.seed)
                _ = self.model(state)
                self.seed += 1

                z = self.model.get_weights()
                z = np.concatenate((tf.reshape(z[0],-1), 
                                    tf.reshape(z[1],-1), 
                                    tf.reshape(z[2],-1), 
                                    tf.reshape(z[3],-1),
                                    tf.reshape(z[4],-1),
                                    tf.reshape(z[5],-1)))
                
                self.agents.append(z)

        elif self.mode == 'quantum':
            for _ in range(self.population_size):
                if self.init_params_mode == 'double_center':    
                    phi = np.random.uniform(-self.init_params+np.pi/2, self.init_params+np.pi/2, int(2*self.num_layers*self.qubits))
                    psi = np.random.uniform(-self.init_params-np.pi/2, self.init_params-np.pi/2, int(self.num_layers*self.qubits))
                    weights = np.concatenate([phi, psi])
                    np.random.shuffle(weights)
                elif self.init_params_mode == 'plus-minus-uniform':
                    weights = np.random.uniform(-self.init_params, self.init_params, int(self.num_params*self.num_layers*self.qubits))

                if self.use_input_scaling:
                    input_scaling = np.ones(self.num_layers*self.qubits)
                    weights = np.concatenate([weights, input_scaling])

                if self.use_output_scaling_actor:
                    if isinstance(self.output_scaling_actor, list):
                        if self.measurement_type == 'exp_@_exp':
                            weights = np.concatenate([weights, deepcopy(np.full(self.env.action_space.shape[0], self.output_scaling_actor))])
                        elif isinstance(self.env.action_space, Discrete):
                            weights = np.concatenate([weights, deepcopy(np.full(self.env.action_space.n, self.output_scaling_actor))])
                        else:
                            weights = np.concatenate([weights, deepcopy(np.full(self.env.action_space.shape[0]*2, self.output_scaling_actor))])
                    else:
                        weights = np.concatenate([weights, deepcopy(np.array([self.output_scaling_actor]))])
                
                self.agents.append(weights)
                

    @ray.remote(num_cpus=1)
    def evaluate(evaluate_num, weights, env, config):
        
        total_score = 0
        circuit_evaluations = 0

        if config['mode'] == 'classical':
            model = set_model_weights(config['model'], weights)
        elif config['mode'] == 'quantum':
            dev = qml.device(config['backend'], wires=config['qubits'])
            circuit = create_circuit(config, dev, weights, env)
        
        for _ in range(evaluate_num):

            done = False
            state = env.reset()
            while not done:

                if isinstance(env.action_space, Box):
                    if config['mode'] == 'classical':
                        prediction = model(tf.reshape(state, (1,-1)))[:env.action_space.shape[0]] # * 3
                        action = np.reshape(prediction, -1)
                        # print(prediction)

                        # action = []
                        # idx = 0
                        # for _ in range(env.action_space.shape[0]):
                        #     action.append(float(np.random.normal(prediction[idx], np.abs(prediction[idx+1]), 1)))
                        #     idx += 2
                    elif config['mode'] == 'quantum':
                        prediction = circuit(weights, state, config)
                        if config['measurement_type'] == 'exp':
                            prediction = prediction[:env.action_space.shape[0]*2]
                        elif config['measurement_type'] == 'exp_@_exp':
                            prediction = prediction[:env.action_space.shape[0]]
                        elif config['measurement_type'] == 'exp_/_var':
                            prediction = prediction[:env.action_space.shape[0]*2]
                        elif config['measurement_type'] == 'exp_+_var':
                            prediction = prediction[:env.action_space.shape[0]*2]

                        if 'postprocessing' in config.keys():
                            action = postprocessing(prediction, config, env.action_space.shape[0], weights[-1])
                        else:

                            if config['use_output_scaling_actor']:
                                if isinstance(config['output_scaling_actor'], list):
                                    prediction = prediction*weights[-len(config['output_scaling_actor'])]
                                else:
                                    prediction = prediction*weights[-1]
                            
                            if config['action_type'] == 'normal':
                                action = []
                                idx = 0
                                for _ in range(env.action_space.shape[0]):
                                    action.append(float(np.random.normal(prediction[idx], np.abs(prediction[idx+1]), 1)))
                                    idx += 2
                            elif config['action_type'] == 'raw':
                                action = prediction[:env.action_space.shape[0]]
                            
                            # action = np.clip(action, -1., 1.)
                
                elif isinstance(env.action_space, Discrete):
                    if config['mode'] == 'classical':
                        prediction = model(tf.reshape(state, (1,-1))).numpy()
                        action = np.argmax(prediction)

                    elif config['mode'] == 'quantum':
                        prediction = circuit(weights, state, config)

                        if config['measurement_type'] == 'exp':
                            prediction = prediction[:env.action_space.n]
                    
                        if config['use_output_scaling_actor']:
                            if isinstance(config['output_scaling_actor'], list):
                                prediction = prediction*weights[-len(config['output_scaling_actor'])]
                            else:
                                prediction = prediction*weights[-1]
                        if config['action_type'] == 'softmax':
                            def softmax(x):
                                """Compute softmax values for each sets of scores in x."""
                                e_x = np.exp(x - np.max(x))
                                return e_x / e_x.sum()
                            prediction = softmax(prediction)
                            action = np.random.choice([0,1], p=prediction)
                        else:
                            action = int(np.argmax(prediction))

                next_state, reward, done, info = env.step(action)
                state = next_state
                total_score += reward
                circuit_evaluations += 1

        return total_score, circuit_evaluations

    def step(self):

        def compute_ranks(x):
            """
            Returns ranks in [0, len(x))
            Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
            """
            assert x.ndim == 1
            ranks = np.empty(len(x), dtype=int)
            ranks[x.argsort()] = np.arange(len(x))
            return ranks

        def compute_centered_ranks(x):
            y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
            y /= (x.size - 1)
            y -= .5
            y *= 2
            return y


        self.t_step +=1
        self.epsilons = np.random.normal(0,1,(self.population_size, self.weight_space))
        self.fitness = []
        self.render_gif = False

        for idx in range(self.population_size):
  
            self.fitness.append(self.evaluate.remote(self.evaluate_num, self.agents[idx], self.env, self.config))

        self.fitness = np.array(ray.get(self.fitness))
        self.rewards = self.fitness[:,0]/self.evaluate_num
        self.circuit_evaluations += sum(self.fitness[:,1])

        elite_idx = np.argsort(self.rewards)[self.population_size-self.elite_num:]
        
        elite = []

        for idx in elite_idx:
            elite.append(deepcopy(self.agents[idx]))

        self.agents = []

        for e in elite:
            self.agents.append(deepcopy(e))

        self.fitness_elite = []

        for idx in range(self.elite_num):

            self.fitness_elite.append(self.evaluate.remote(self.evaluate_num_elite, self.agents[idx], self.env, self.config))

        self.fitness_elite = np.array(ray.get(self.fitness_elite))
        self.rewards_elite = self.fitness_elite[:,0]/self.evaluate_num_elite
        self.circuit_evaluations += sum(self.fitness_elite[:,1])

        for idx in range(self.population_size-self.elite_num):
            y = np.random.randint(0, self.elite_num)
            self.agents.append(deepcopy(elite[y] + self.epsilons[idx] * self.lr* self.sigma))       
        
        self.lr = self.lr * self.lr_decay_factor

        logging = {'episode_reward_mean': np.mean(self.rewards_elite),
                    'episode_reward_max': np.max(self.rewards_elite), 
                    'episode_reward_min': np.min(self.rewards_elite), 
                    'cur_lr_rate': self.lr,
                    'sigma': self.sigma,
                    'circuit_evaluations': self.circuit_evaluations,
                    'weights': self.agents[self.elite_num-1]
                }
        
        if self.mode == 'quantum':
            layer_size = 3*self.qubits 
            z = self.agents[self.elite_num-1]
            x = 0
            logging[f'Rot_X'] = []
            logging[f'Rot_Y'] = [] 
            logging[f'Rot_Z'] = [] 

            for idx in range(self.num_layers):
                logging[f'layer_{idx}'] = z[idx*layer_size:(idx+1)*layer_size]
                logging[f'Rot_X_layer_{idx}'] = [] 
                logging[f'Rot_Y_layer_{idx}'] = []
                logging[f'Rot_Z_layer_{idx}'] = []

                for i in range(self.qubits):
                    logging['Rot_X'].append(z[x])
                    logging['Rot_Y'].append(z[x+1])
                    logging['Rot_Z'].append(z[x+2])
                    logging[f'Rot_X_layer_{idx}'].append(z[x])
                    logging[f'Rot_Y_layer_{idx}'].append(z[x+1])
                    logging[f'Rot_Z_layer_{idx}'].append(z[x+2])
                    x += 3

        return  logging
                    

    

