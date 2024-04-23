from ray import tune 
import numpy as np 
import pennylane as qml 
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiDiscrete
import ray
from copy import deepcopy
from circuits.postprocessing import postprocessing

# Take care, this has not been maintained in a while.

class Evolution_Strategy(tune.Trainable):

    def setup(self, config):

        if isinstance(config['env'], str):        
            self.env = gym.make(config['env'], render_mode='rgb_array')
            print(self.env)
        else:
            self.env = config['env'](config['env_config']) 

        self.config = config
        self.mode = config['mode']
        self.lr = config['lr']
        self.lr_decay_factor = config['lr_decay_factor']
        self.population_size = config['population_size']
        self.evaluate_num = config['evaluate_num']
        self.sigma = config['sigma']
        self.num_layers = int(config['num_layers'])
        self.config['num_layers'] = int(config['num_layers'])
        self.qubits = self.env.observation_space.shape[0]
        self.config['num_qubits'] = self.qubits
        self.vqc_type = config['vqc_type']
        self.init_params = config['init_params']
        self.measurement_type = config['measurement_type']

        self.use_input_scaling = config['use_input_scaling']
        self.use_output_scaling_actor = config['use_output_scaling_actor']
        self.init_params_mode = config['init_params_mode']
        self.init_output_scaling_actor = config['init_output_scaling_actor']

        print(self.env.action_space)
        if isinstance(self.env.action_space, Box):
            self.action_space = self.env.action_space.shape[0]*2
        elif isinstance(self.env.action_space, Discrete):
            self.action_space = self.env.action_space.n
        elif isinstance(self.env.action_space, MultiDiscrete):
            self.action_space = np.sum(action.n for action in self.env.action_space)

        self.t_step = 0
        self.circuit_evaluations = 0
        self.average_reward = []
        self.agent = []
        self.init_population()
        self.weight_space = self.agent.shape[0]


    def init_population(self):

        if self.init_params_mode == 'double_center':    
            phi = np.random.uniform(-self.init_params+np.pi/2, self.init_params+np.pi/2, int(self.num_layers*self.qubits))
            psi = np.random.uniform(-self.init_params-np.pi/2, self.init_params-np.pi/2, int(self.num_layers*self.qubits))
            weights = np.concatenate([phi, psi])
            np.random.shuffle(weights)
        elif self.init_params_mode == 'plus-minus-uniform':
            weights = np.random.uniform(-self.init_params, self.init_params, int(self.config['num_variational_params']*self.num_layers*self.qubits))
        elif self.init_params_mode == 'plus-zero-normal':
            weights = np.random.normal(0, self.init_params, int(self.config['num_variational_params']*self.num_layers*self.qubits))

        if self.use_input_scaling:
            input_scaling = np.ones(self.num_layers*self.qubits*self.config['num_scaling_params'])
            weights = np.concatenate([weights, input_scaling])

        if self.use_output_scaling_actor:
            if isinstance(self.init_output_scaling_actor, list):
                if self.measurement_type == 'exp_@_exp':
                    weights = np.concatenate([weights, deepcopy(np.full(self.env.action_space.shape[0], self.init_output_scaling_actor))])
                elif isinstance(self.env.action_space, Discrete):
                    weights = np.concatenate([weights, deepcopy(np.full(self.env.action_space.n, self.init_output_scaling_actor))])
                elif isinstance(self.env.action_space, MultiDiscrete):
                    weights = np.concatenate([weights, deepcopy(np.full(np.sum(action.n for action in self.env.action_space), self.init_output_scaling_actor))])
                else:
                    weights = np.concatenate([weights, deepcopy(np.full(self.env.action_space.shape[0]*2, self.init_output_scaling_actor))])
            else:
                weights = np.concatenate([weights, deepcopy(np.array([self.init_output_scaling_actor]))])
        
        self.agent = weights

    @ray.remote(num_cpus=0.5)
    def evaluate(evaluate_num, weights, env, action_space, config):
        
        total_score = 0
        circuit_evaluations = 0
        dev = qml.device(config['backend_name'], wires=config['num_qubits'])
        circuit = create_vqc(config, dev)

        for _ in range(evaluate_num):

            done = False
            state = env.reset()
            while not done:

                prediction = circuit(state, weights, 'es')
                action = postprocessing(prediction, config, env.action_space, weights, 'es')
                
                next_state, reward, done, _ = env.step(action)
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

        fitness = [self.evaluate.remote(self.evaluate_num, self.agent + e * self.sigma, self.env, self.action_space, self.config) for e in self.epsilons]
        fitness.append(self.evaluate.remote(self.evaluate_num, self.agent, self.env, self.action_space, self.config))
        fitness = np.array(ray.get(fitness))
        rewards = fitness[:,0]/self.evaluate_num
        self.circuit_evaluations += sum(fitness[:,1])
        self.rewards = rewards
            
        main_reward = float(rewards[-1])/self.evaluate_num
        self.rewards = compute_centered_ranks(np.array(self.rewards))

        for i, e in enumerate(self.epsilons):
            noise = self.rewards[i]*e
            self.agent = self.agent + noise * self.lr/(self.population_size*self.sigma)

        self.lr = self.lr*self.lr_decay_factor


        logging =  {'episode_reward_mean': main_reward,
                    'episode_reward_max': np.max(rewards), 
                    'episode_reward_min': np.min(rewards), 
                    'cur_lr_rate': self.lr,
                    'sigma': self.sigma,
                    'circuit_evaluations': self.circuit_evaluations,
                    'weights': self.agent
                    } 

        return logging 
