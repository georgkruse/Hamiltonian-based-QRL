from abc import ABC
import pennylane as qml
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn import functional as F
from ray.rllib.utils.annotations import override
from ray.rllib.algorithms.pg.pg_torch_policy import PGTorchPolicy
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.algorithms.pg.pg import PGConfig
from ray.rllib.utils.schedules import PiecewiseSchedule
from ray.rllib.policy.policy import Policy


import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiBinary, MultiDiscrete, Dict

from circuits.postprocessing import *
from circuits.quantum_circuits import vqc_switch

class LearningRateSchedule:
    """Mixin for TorchPolicy that adds a learning rate schedule."""

    def __init__(self, lr, lr_schedule, config):
        self.config_ = config.model['custom_model_config']
        self._lr_schedule = None
        self.lr_schedule_data = lr_schedule 
        if lr_schedule is None:
            self.cur_lr = lr
        else:
            self._lr_schedule = PiecewiseSchedule(
                lr_schedule, outside_value=lr_schedule[-1][-1], framework=None
            )
            self.cur_lr_0 = self._optimizers[0].param_groups[0]['lr']
            self.cur_lr_1 = self._optimizers[0].param_groups[-1]['lr']
        if self.config_['use_temperature']:
            self.model.temperature_schedule = np.linspace(start=self.config_['temperature_schedule'][1][0],stop=self.config_['temperature_schedule'][1][1],
                                   num=self.config_['temperature_schedule'][0][1]-self.config_['temperature_schedule'][0][0])
            
            self.model.temperature = self.config_['temperature_schedule'][1][0]

    @override(Policy)
    def on_global_var_update(self, global_vars):
        super().on_global_var_update(global_vars)
        if self._lr_schedule:
            self.cur_lr = self._lr_schedule.value(global_vars["timestep"])
            if global_vars["timestep"] >= self.lr_schedule_data[1][0]:
                self.cur_lr_0 = self.cur_lr_0*self.lr_schedule_data[0][1]
                self.cur_lr_1 = self.cur_lr_1*self.lr_schedule_data[0][1]
                for opt in self._optimizers:
                    for p in opt.param_groups:
                        if len(p['params']) == 1:
                            p["lr"] = self.cur_lr_1
                        else:
                            p["lr"] = self.cur_lr_0
        if self.config_['use_temperature']:
            if self.model.temperature_schedule.shape[0] > global_vars["timestep"] + 1:
                self.model.temperature = self.model.temperature_schedule[global_vars["timestep"]]
            else:
                self.model.temperature = self.config_['temperature_schedule'][1][1]


class QRL_PGTorchPolicy(PGTorchPolicy, LearningRateSchedule, TorchPolicyV2):
    """PyTorch policy class used with PG."""

    def __init__(self, observation_space, action_space, config):
        # Enforce AlgorithmConfig for PG Policies.
        if isinstance(config, dict):
            config = PGConfig.from_dict(config)
        TorchPolicyV2.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config.model["max_seq_len"],
        )
        LearningRateSchedule.__init__(self, config.lr, config.lr_schedule, config)
        self._initialize_loss_from_dummy_batch()

    @override(TorchPolicyV2)
    def stats_fn(self, train_batch: SampleBatch):
        stats_dict = {
                "policy_loss": torch.mean(torch.stack(self.get_tower_stats("policy_loss"))),
                "cur_lr": self.cur_lr,
        }
        if self.config_['use_temperature']:
            stats_dict["temperature"] = self.model.temperature
            
        return convert_to_numpy(stats_dict)       
        
    @override(TorchPolicyV2)
    def optimizer(self): 
        '''
        In this function, the ray optimizer is overwritten. Use custom learning rate for variational parameters, 
        input scaling parameters and output scaling parameters
        '''
        if hasattr(self, "config"):
            
            params = []    

            if self.config['model']['custom_model_config']['mode'] == 'quantum':
                
                # Use custom learning rate for variational parameters, input scaling parameters and output scaling parameters
                params.append({'params': [self.model._parameters[f'weights_actor']], 'lr': self.config['lr']})
                params.append({'params': [self.model._parameters[f'input_scaling_actor']], 'lr': self.config['lr']})
                
                if 'lr_output_scaling' in self.config['model']['custom_model_config'].keys():
                    print('Using lr_output_scaling:', self.config['model']['custom_model_config']['lr_output_scaling'])
                    custom_lr = self.config['model']['custom_model_config']['lr_output_scaling']
                else:                            
                    print('NOT using lr_output_scaling:', self.config['lr'])
                    custom_lr = self.config['lr']

                if 'output_scaling_actor' in self.model._parameters.keys():
                    params.append({'params': self.model._parameters['output_scaling_actor'], 'lr': custom_lr})

                if 'weight_decay' in self.config['model']['custom_model_config'].keys():
                    weight_decay = self.config['model']['custom_model_config']['weight_decay']
                else:
                    weight_decay = 0

                if self.config['model']['custom_model_config']['custom_optimizer'] == 'Adam':
                    optimizers = [
                        torch.optim.Adam(params, amsgrad=True, weight_decay=weight_decay)
                    ]
                elif self.config['model']['custom_model_config']['custom_optimizer'] == 'SGD':
                    optimizers = [
                        torch.optim.SGD(params)
                    ]
                elif self.config['model']['custom_model_config']['custom_optimizer'] == 'RMSprop':
                    optimizers = [
                        torch.optim.RMSprop(params)
                    ]
                elif self.config['model']['custom_model_config']['custom_optimizer'] == 'LBFGS':
                    optimizers = [
                    torch.optim.LBFGS(params)
                    ]

                print('Using optimizer:', self.config['model']['custom_model_config']['custom_optimizer'])

            elif self.config['model']['custom_model_config']['mode'] == 'classical':
                optimizers = [torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])]
            else:
                print('Incomplete config file.')
                exit()
        else:
            optimizers = [torch.optim.Adam(self.model.parameters())]
        
        if getattr(self, "exploration", None):
            optimizers = self.exploration.get_exploration_optimizer(optimizers)
        return optimizers

class QRL_PG(Algorithm):
    
    @override(Algorithm)
    def get_default_policy_class(cls, config):
        if config["framework"] == "torch":
            return QRL_PGTorchPolicy

class QuantumPGModel(TorchModelV2, nn.Module, ABC):
    '''
    Quantum Model for Policy Gradient.
    '''

    def __init__(self, obs_space, action_space, num_actions, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_actions, model_config, name
        )
        nn.Module.__init__(self)
        self.counter = -3
        self.reset = True
        self.config = model_config['custom_model_config']
        self.mode = self.config['mode']
        self.num_params = self.config['num_variational_params']

        if isinstance(self.action_space, Box):
            self.num_outputs = self.action_space.shape[0]*2
        elif isinstance(self.action_space, Discrete):
            self.num_outputs = self.action_space.n
        elif isinstance(self.action_space, MultiBinary):
            self.num_outputs = self.action_space.n
        elif isinstance(self.action_space, MultiDiscrete): 
            self.num_outputs = np.sum(action.n for action in self.action_space)

        if isinstance(self.obs_space, gym.spaces.Dict):
            self.num_inputs = sum([box.shape[0] for box in self.obs_space.values()])
        else:
            self.num_inputs = self.obs_space.shape[0]

        self.init_variational_params = self.config['init_variational_params']
        self.measurement_type_actor = self.config['measurement_type_actor']
        self.softmax = torch.nn.Softmax(dim=1)
        self.num_actions = num_actions
        self.config['num_layers'] = int(self.config['num_layers'])
        self.num_layers = int(self.config['num_layers'])
        self.num_qubits = int(self.config['vqc_type'][1])
        self.config['num_qubits'] = self.num_qubits
        self.layerwise_training = self.config['layerwise_training']
        self.gradient_clipping = self.config['gradient_clipping']
        self.use_input_scaling = self.config['use_input_scaling']
        self.use_output_scaling_actor = self.config['use_output_scaling_actor']
        self.init_output_scaling_actor = self.config['init_output_scaling_actor']
        self.init_input_scaling_actor = self.config['init_input_scaling_actor']

        self.use_classical_layer = self.config['use_classical_layer']        
        self.layer_size = self.config['layer_size']

        self.weight_logging_interval = self.config['weight_logging_interval']
        self.weight_plotting = self.config['weight_plotting']
        self._value_out = None
       
        def init_weights(size):
            if self.config['init_variational_params_mode'] == 'plus-zero-uniform':
                return torch.FloatTensor(*size).uniform_(0, self.init_variational_params)
            elif self.config['init_variational_params_mode'] == 'plus-minus-uniform':
                return torch.FloatTensor(*size).uniform_(-self.init_variational_params, self.init_variational_params)
            elif self.config['init_variational_params_mode'] == 'plus-zero-normal':
                return torch.FloatTensor(*size).normal_(0., self.init_variational_params)
            elif self.config['init_variational_params_mode'] == 'constant':
                return torch.tensor(np.full((*size,), self.init_variational_params))

        if self.mode == 'quantum':     

            if self.config['encoding_type'] == 'graph_encoding':
                
                if self.config['graph_encoding_type'] in ['sge-sgv', 'sge-sgv-linear', 'sge-sgv-quadratic']:
                    size_vqc = 1
                    size_input_scaling = 1
                elif self.config['graph_encoding_type'] == 'mge-mgv':
                    size_vqc = self.num_qubits
                    size_input_scaling = sum(range(self.num_qubits+1))+self.num_qubits
                elif self.config['graph_encoding_type'] == 'mge-mgv-linear':
                    size_vqc = self.num_qubits
                    size_input_scaling = self.num_qubits
                elif self.config['graph_encoding_type'] == 'mge-mgv-quadratic':
                    size_vqc = self.num_qubits
                    size_input_scaling = sum(range(self.num_qubits+1))
                elif self.config['graph_encoding_type'] == 'mge-sgv':
                    size_vqc = 1
                    size_input_scaling = sum(range(self.num_qubits+1))+self.num_qubits
                elif self.config['graph_encoding_type'] == 'mge-sgv-linear':
                    size_vqc = 1
                    size_input_scaling = self.num_qubits + 1
                elif self.config['graph_encoding_type'] == 'mge-sgv-quadratic':
                    size_vqc = 1
                    size_input_scaling = sum(range(self.num_qubits+1)) + 1
                elif self.config['graph_encoding_type'] in ['angular', 'angular-hea']:
                    size_vqc = self.num_qubits*self.num_params
                    size_input_scaling = self.num_qubits*self.config['num_scaling_params']
                elif self.config['graph_encoding_type'] == 'hamiltonian-hea':
                    size_vqc = self.num_qubits*self.num_params
                    size_input_scaling = 0
                if self.config['block_sequence'] in ['enc_var_ent', 'enc_var', 'enc_ent_var']:
                    size_vqc += self.num_qubits*self.num_params
            else:
                size_vqc = self.num_qubits*self.num_params
                size_input_scaling = self.num_qubits*self.config['num_scaling_params']      
                
            self.register_parameter(name=f'weights_actor', param=torch.nn.Parameter(init_weights((self.num_layers, size_vqc)), requires_grad=True))

            if self.use_input_scaling:
                self.register_parameter(name=f'input_scaling_actor', param=torch.nn.Parameter(torch.tensor(np.full((self.num_layers, size_input_scaling), 1.)), requires_grad=True))
            
            if self.use_output_scaling_actor:
                if isinstance(self.init_output_scaling_actor, list):
                    self.register_parameter(name='output_scaling_actor', param=torch.nn.Parameter(torch.tensor(np.full(self.num_outputs, self.config['init_output_scaling_actor'][0])), requires_grad=True))
                else:
                    self.register_parameter(name='output_scaling_actor', param=torch.nn.Parameter(torch.tensor(self.config['init_output_scaling_actor']), requires_grad=True))
           

            if self.gradient_clipping:
                self.weights_actor.register_hook(lambda grad: torch.clip(grad, -1., 1.))
                                
            dev_actor = qml.device(self.config['backend_name'], wires=self.num_qubits)

            self.qnode_actor = qml.QNode(vqc_switch[self.config['vqc_type'][0]], dev_actor, interface=self.config['interface'], diff_method=self.config['diff_method']) #, argnum=0)

            if self.use_classical_layer:
                self.classical_layer_actor = nn.Linear(in_features=self.num_inputs, out_features=self.num_actions, dtype=torch.float32)
           
        elif self.mode == 'classical':
            if self.config['activation_function'] == 'relu':
                activation = nn.ReLU()
            elif self.config['activation_function'] == 'leaky_relu':
                activation = nn.LeakyReLU()
            if self.config['activation_function'] == 'tanh':
                activation = nn.Tanh()
            if len(self.layer_size) == 1:
                self.actor_network = nn.Sequential(nn.Linear(in_features=self.num_inputs, out_features=self.layer_size[0]),
                                                    activation,
                                                    nn.Linear(in_features=self.layer_size[0], out_features=self.num_outputs))
                
            elif len(self.layer_size) == 2:
                
                self.actor_network = nn.Sequential(nn.Linear(in_features=self.num_inputs, out_features=self.layer_size[0]),
                                                    activation,
                                                    nn.Linear(in_features=self.layer_size[0], out_features=self.layer_size[1]),
                                                    activation,
                                                    nn.Linear(in_features=self.layer_size[1], out_features=self.num_outputs))
                
            elif len(self.layer_size) == 3:
                self.actor_network = nn.Sequential(nn.Linear(in_features=self.num_inputs, out_features=self.layer_size[0]),
                                                    activation,
                                                    nn.Linear(in_features=self.layer_size[0], out_features=self.layer_size[1]),
                                                    activation,
                                                    nn.Linear(in_features=self.layer_size[1], out_features=self.layer_size[2]),
                                                    activation,
                                                    nn.Linear(in_features=self.layer_size[2], out_features=self.num_outputs))
                
                
    def forward(self, input_dict, state, seq_lens):
        
        state = input_dict['obs']
       
        # Set gradients in layers to True/False if layerwise training
        if self.layerwise_training:
            self.set_layerwise_training()

        if self.mode == 'quantum':

            # Check the encoding block type in order to adapt obs/state accordingly
            if 'double' in self.config['vqc_type'][0]:
                state = torch.concat([state, state], dim=1)
            elif 'triple' in self.config['vqc_type'][0]:
                state = torch.concat([state, state, state], dim=1)
            elif 'circular' in self.config['vqc_type'][0]:
                reps = round((self.num_qubits*self.num_layers)/state.shape[1] + 0.5)
                state = torch.concat([state for _ in range(reps)], dim=1)
            else:
                if not isinstance(self.obs_space, gym.spaces.Dict): 
                    state = torch.reshape(state, (-1, self.obs_space.shape[0]))


            # If vqc_type is relu or qcnn, two function calls are required
            if 'relu' in self.config['vqc_type'][0]:    
                activations_actor = self.qnode_activations(theta=state, weights=self._parameters, config=self.config, type='actor', activations=None)
                prob = self.qnode(theta=state, weights=self._parameters, config=self.config, type='actor', activations=activations_actor)
            elif 'qcnn' in self.config['vqc_type'][0]:    
                activations_actor = self.qnode(theta=state, weights=self._parameters, config=self.config, type='activations_actor', activations=None)
                activations_actor = torch.reshape(activations_actor, (-1, 4))
                prob = self.qnode(theta=state, weights=self._parameters, config=self.config, type='actor', activations=activations_actor)
            else:
                if self.config['measurement_type_actor'] == 'probs':
                    prob = []
                    for i in range(state['linear_0'].shape[0]):
                        tmp_state = {}
                        tmp_state['linear_0'] = np.reshape(state['linear_0'][i], (-1, *state['linear_0'].shape[1:]))
                        tmp_state['quadratic_0'] = np.reshape(state['quadratic_0'][i], (-1, *state['quadratic_0'].shape[1:]))

                        prob.append(self.qnode_actor(theta=tmp_state, weights=self._parameters, config=self.config, type='actor', activations=None, H=None))

                    prob = torch.stack(prob)
                elif self.config['measurement_type_actor'] == 'edge':
                    prob = []
                    for i in range(state['quadratic_0'].shape[1]):
                        if not state['quadratic_0'][0,0,0].to(torch.int).item() + state['quadratic_0'][0,0,1].to(torch.int).item() == 0:
                            self.config['edge_measurement'] = state['quadratic_0'][0,i,:2]
                        else:
                            self.config['edge_measurement'] = [torch.tensor(0.), torch.tensor(1.)]
                        prob.append(self.qnode_actor(theta=state, weights=self._parameters, config=self.config, type='actor', activations=None, H=None))
                    prob = torch.stack(prob).T
                    prob = torch.reshape(prob, (-1, state['quadratic_0'].shape[1]))

                else:
                    prob = self.qnode_actor(theta=state, weights=self._parameters, config=self.config, type='actor', activations=None, H=None)
            

            if isinstance(self.action_space, MultiDiscrete):
                
                # Do for now test if action space has 2 elements / is binary
                if self.action_space[0].n == 2:
                    prob = (torch.ones(self.num_qubits) + prob)/2
                    prob = torch.reshape(prob, (-1,self.num_qubits)).repeat_interleave(2, dim=1)*torch.tensor([1., -1.]).repeat(self.num_qubits)
                    prob = torch.tensor([0., 1.]).repeat(self.num_qubits) + prob
                # else:
                #     prob = torch.reshape(prob, (-1, self.action_space.shape[0], self.action_space.shape[1]))

            if self.use_output_scaling_actor:
                logits = torch.reshape(postprocessing(prob, self.config, self.num_outputs, self._parameters, 'actor'), (-1, self.num_outputs))
            else:
                logits =  torch.reshape(prob, (-1, self.num_outputs)) #prob[:,:self.num_outputs]


        elif self.mode == 'classical':

            if isinstance(self.action_space, MultiBinary):
                logits = []
                action = torch.tensor([0.]).repeat(state.shape[0], 1)
                for i in range(len(self.action_space)):
                    action_one_hot = F.one_hot(torch.tensor([i]), num_classes=len(self.action_space))
                    action_one_hot = action_one_hot[0].repeat(state.shape[0], 1)
                    input_actor = torch.concatenate([state, action_one_hot, action], dim=1)
                    probs = self.actor_network(input_actor)
                    action = torch.reshape(torch.argmax(probs, dim=1), (-1, 1))
                    action_mask = F.one_hot(torch.argmax(probs, dim=1), num_classes=2)
                    logits.append(probs*action_mask)
                logits = torch.hstack(logits)

            elif isinstance(self.action_space, MultiDiscrete):
                input_state = []
                for idx in range(int(len(state.keys())/2)):
                    linear = state[f'linear_{idx}'][:,:,1]
                    quadratic = state[f'quadratic_{idx}'][:,:,1]
                    input_state.append(linear)
                    # input_state.append(quadratic)

                input_state = torch.from_numpy(np.hstack(input_state)).type(torch.float32)
                logits = self.actor_network(input_state)
            
            else:
                logits = self.actor_network(state)
                
        if self.config['output_scaling_schedule']:
            self.counter += 1
            if self.counter % self.config["steps_for_output_scaling_update"] == 0:
                if self.counter <= self.config["max_steps_output_scaling_update"]:
                    if logits.shape[0] == 1.:
                        self._parameters['output_scaling_actor'] += self.config["output_scaling_update"]
                    else:
                        self.counter -= 1
                print('output_scaling_actor:', self._parameters['output_scaling_actor'])
                
        if self.config['use_temperature']:
            
            logits = logits/self.temperature

        if isinstance(self.obs_space, Dict):      
            if "annotations" in state.keys():
                if 'current_node' in state.keys():
                    if state['current_node'][0].to(torch.int).item() != -1:
                        if not state['quadratic_0'][0,0,0].to(torch.int).item() + state['quadratic_0'][0,0,1].to(torch.int).item() == 0:
                            new_logits_batch = []
                            
                            for batch_dim in range(state["quadratic_0"].shape[0]):
                                new_logits = []
                                weights_comp = []
                                for idx, (node1, node2, value) in enumerate(state['quadratic_0'][batch_dim]):
                                    node1 = int(node1)
                                    node2 = int(node2)
                                    if node1 == state['current_node'][batch_dim,0]:
                                        if state['annotations'][batch_dim,node2,1] == np.pi:
                                            # new_logits.append(logits[batch_dim,idx]*value)
                                            new_logits.append(logits[batch_dim,idx])
                                            weights_comp.append(value)
                                        elif state['annotations'][batch_dim,node2,1] == 0:
                                            new_logits.append(torch.tensor(-10_000))
                                            weights_comp.append(10_000)
                                    elif node2 == state['current_node'][batch_dim,0]:
                                        if state['annotations'][batch_dim,node1,1] == np.pi:
                                            # logits[idx] = logits[idx]*value
                                            # new_logits.append(logits[batch_dim,idx]*value)
                                            new_logits.append(logits[batch_dim,idx])
                                            weights_comp.append(value)

                                        elif state['annotations'][batch_dim,node1,1] == 0:
                                            # logits[idx] = -10_000
                                            new_logits.append(torch.tensor(-10_000))
                                            weights_comp.append(10_000)
                                    
                                new_logits_batch.append(torch.stack(new_logits))
                            logits = torch.stack(new_logits_batch, dim=0)
                        else:
                            logits = logits[:,:self.action_space.n]

                    elif state['current_node'][0].to(torch.int).item() == -1:
                        nodes = state["annotations"].shape[1]
                        batch_dim = state["annotations"].shape[0]
                        if batch_dim == 1:
                            for batch in range(batch_dim):
                                for node in range(nodes):
                                    if state["annotations"][batch][node][1] == 0:
                                        logits[batch][node] = -np.inf
                                    else:
                                        logits[batch][node] *= -1
                        else:
                            for batch in range(batch_dim):
                                for node in range(nodes):
                                    if state["annotations"][batch][node][1] == 0:
                                        logits[batch][node] = -10000
                                    else:
                                        logits[batch][node] *= -1
                else:
                    if not state['quadratic_0'][0,0,0].to(torch.int).item() + state['quadratic_0'][0,0,1].to(torch.int).item() == 0:
                        nodes = state["annotations"].shape[1]
                        batch_dim = state["annotations"].shape[0]
                        for batch in range(batch_dim):
                            for node in range(nodes):
                                if state["annotations"][batch][node][1] == 0:
                                    logits[batch][node] = -np.inf
        self._logits = logits
        return logits, []
    

