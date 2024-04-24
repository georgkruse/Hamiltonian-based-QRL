from abc import ABC
import numpy as np
import ray
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from gymnasium.spaces import Box, Discrete, MultiBinary, MultiDiscrete, Dict
import pennylane as qml
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from ray.rllib.utils.annotations import override
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import Postprocessing  
from ray.rllib.utils.torch_utils import sequence_mask, explained_variance
from ray.rllib.policy.policy import Policy

from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.schedules import PiecewiseSchedule

from circuits.postprocessing import *
from circuits.quantum_circuits import vqc_switch

from ray.rllib.policy.torch_mixins import (
    EntropyCoeffSchedule,
    KLCoeffMixin,
    ValueNetworkMixin,
)

class LearningRateSchedule:
    """Mixin for TorchPolicy that adds a learning rate schedule."""

    def __init__(self, lr, lr_schedule):
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

class QRL_PPOTorchPolicy(
    PPOTorchPolicy, 
    ValueNetworkMixin,
    LearningRateSchedule,
    EntropyCoeffSchedule,
    KLCoeffMixin,
    TorchPolicyV2,
):
    """PyTorch policy class used with PPO."""

    def __init__(self, observation_space, action_space, config):
        config = dict(ray.rllib.algorithms.ppo.ppo.PPOConfig().to_dict(), **config)
        TorchPolicyV2.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"],
        )
        print('Using Torch PPO policy')
        ValueNetworkMixin.__init__(self, config)
        LearningRateSchedule.__init__(self, config["lr"], config["lr_schedule"])
        EntropyCoeffSchedule.__init__(
            self, config["entropy_coeff"], config["entropy_coeff_schedule"]
        )
        KLCoeffMixin.__init__(self, config)

        # TODO: Don't require users to call this manually.
        self._initialize_loss_from_dummy_batch()


    @override(TorchPolicyV2)
    def stats_fn(self, train_batch: SampleBatch):
        return convert_to_numpy(
            {
                "cur_kl_coeff": self.kl_coeff,
                # "cur_lr_0": self.cur_lr_0,
                # "cur_lr_1": self.cur_lr_1,
                "total_loss": torch.mean(
                    torch.stack(self.get_tower_stats("total_loss"))
                ),
                "policy_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_policy_loss"))
                ),
                "vf_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_vf_loss"))
                ),
                "vf_explained_var": torch.mean(
                    torch.stack(self.get_tower_stats("vf_explained_var"))
                ),
                "kl": torch.mean(torch.stack(self.get_tower_stats("mean_kl_loss"))),
                "entropy": torch.mean(
                    torch.stack(self.get_tower_stats("mean_entropy"))
                ),
                "entropy_coeff": self.entropy_coeff,
            }
        )
    
    @override(TorchPolicyV2)
    def loss(self, model, dist_class, train_batch):
        """Compute loss for Proximal Policy Objective.

        Args:
            model: The Model to calculate the loss for.
            dist_class: The action distr. class.
            train_batch: The training data.

        Returns:
            The PPO loss tensor given the input batch.
        """
        
        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)

        # RNN case: Mask away 0-padded chunks at end of time axis.
        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major(),
            )
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid

        # non-RNN case: No masking.
        else:
            mask = None
            reduce_mean_valid = torch.mean

        prev_action_dist = dist_class(
            train_batch[SampleBatch.ACTION_DIST_INPUTS], model
        )

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
        )

        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if self.config["kl_coeff"] > 0.0:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = reduce_mean_valid(action_kl)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        surrogate_loss = torch.min(
            train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
            train_batch[Postprocessing.ADVANTAGES]
            * torch.clamp(
                logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]
            ),
        )

        # Compute a value function loss.
        if self.config["use_critic"]:
            value_fn_out = model.value_function()
            vf_loss = torch.pow(
                value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
            )
            vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
            mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
        # Ignore the value function.
        else:
            value_fn_out = torch.tensor(0.0).to(surrogate_loss.device)
            vf_loss_clipped = mean_vf_loss = torch.tensor(0.0).to(surrogate_loss.device)

        total_loss = reduce_mean_valid(
            -surrogate_loss
            + self.config["vf_loss_coeff"] * vf_loss_clipped
            - self.entropy_coeff * curr_entropy
        )

        # Add mean_kl_loss (already processed through `reduce_mean_valid`),
        # if necessary.
        if self.config["kl_coeff"] > 0.0:
            total_loss += self.kl_coeff * mean_kl_loss

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = reduce_mean_valid(-surrogate_loss)
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
        )
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss

        return total_loss

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
                params.append({'params': [self.model._parameters[f'weights_critic']], 'lr': self.config['lr']})
                params.append({'params': [self.model._parameters[f'weights_actor']], 'lr': self.config['lr']})
                
                params.append({'params': [self.model._parameters[f'input_scaling_actor']], 'lr': self.config['lr']})
                params.append({'params': [self.model._parameters[f'input_scaling_critic']], 'lr': self.config['lr']})
                
                if 'lr_output_scaling' in self.config['model']['custom_model_config'].keys():
                    print('Using lr_output_scaling:', self.config['model']['custom_model_config']['lr_output_scaling'])
                    custom_lr = self.config['model']['custom_model_config']['lr_output_scaling']
                else:                            
                    print('NOT using lr_output_scaling:', self.config['lr'])
                    custom_lr = self.config['lr']

                if 'output_scaling_actor' in self.model._parameters.keys():
                    params.append({'params': self.model._parameters['output_scaling_actor'], 'lr': custom_lr})

                if 'output_scaling_critic' in self.model._parameters.keys():
                    params.append({'params': self.model._parameters['output_scaling_critic'], 'lr': custom_lr})

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
    
class QRL_PPO(PPO):

    @override(PPO)
    def get_default_policy_class(self, config):

        if config["framework"] == "torch":

            return QRL_PPOTorchPolicy


class QuantumPPOModel(TorchModelV2, nn.Module, ABC):
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

        self.init_variational_params = self.config['init_variational_params']
        self.measurement_type_actor = self.config['measurement_type_actor']
        self.measurement_type_critic = self.config['measurement_type_critic']
        self.softmax = torch.nn.Softmax(dim=1)
        self.num_actions = num_actions
        self.config['num_layers'] = int(self.config['num_layers'])
        self.num_layers = int(self.config['num_layers'])
        self.num_qubits = int(self.config['vqc_type'][1])
        self.config['num_qubits'] = self.num_qubits
        self.layerwise_training = self.config['layerwise_training']
        self.gradient_clipping = self.config['gradient_clipping']
        self.use_input_scaling_actor = self.config['use_input_scaling_actor']
        self.use_input_scaling_critic = self.config['use_input_scaling_critic']
        self.use_output_scaling_actor = self.config['use_output_scaling_actor']
        self.use_output_scaling_critic = self.config['use_output_scaling_critic']
        self.init_output_scaling_actor = self.config['init_output_scaling_actor']
        self.init_output_scaling_critic = self.config['init_output_scaling_critic']

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

        if self.mode == 'quantum':     

            if self.config['encoding_type'] == 'graph_encoding':

                if self.config['graph_encoding_type'] in ['s-ppgl', 's-ppgl-linear', 's-ppgl-quadratic']:
                    size_vqc = 1
                    size_input_scaling = 1
                elif self.config['graph_encoding_type'] == 'm-ppgl':
                    size_vqc = self.num_qubits
                    size_input_scaling = sum(range(self.num_qubits+1))+self.num_qubits
                elif self.config['graph_encoding_type'] == 'm-ppgl-linear':
                    size_vqc = self.num_qubits
                    size_input_scaling = self.num_qubits
                elif self.config['graph_encoding_type'] == 'm-ppgl-quadratic':
                    size_vqc = self.num_qubits
                    size_input_scaling = sum(range(self.num_qubits+1))
                elif self.config['graph_encoding_type'] == 'h-ppgl':
                    size_vqc = 1
                    size_input_scaling = sum(range(self.num_qubits+1))+self.num_qubits
                elif self.config['graph_encoding_type'] == 'h-ppgl-linear':
                    size_vqc = 1
                    size_input_scaling = self.num_qubits + 1
                elif self.config['graph_encoding_type'] == 'h-ppgl-quadratic':
                    size_vqc = 1
                    size_input_scaling = sum(range(self.num_qubits+1)) + 1
                elif self.config['graph_encoding_type'] =='angular':
                    size_vqc = 0
                    size_input_scaling = self.num_qubits*self.config['num_scaling_params']
                elif self.config['graph_encoding_type'] =='angular-hwe':
                    size_vqc = self.num_qubits*self.num_params
                    size_input_scaling = self.num_qubits*self.config['num_scaling_params']
                elif self.config['graph_encoding_type'] == 'hamiltonian-hwe':
                    size_vqc = self.num_qubits*self.num_params
                    self.use_input_scaling_actor = False
                    self.use_input_scaling_critic = False

                if self.config['block_sequence'] in ['enc_var_ent', 'enc_var', 'enc_ent_var']:
                    size_vqc += self.num_qubits*self.num_params
            else:
                size_vqc = self.num_qubits*self.num_params
                size_input_scaling = self.num_qubits*self.config['num_scaling_params']
            
            if not (self.config['graph_encoding_type'] == 'angular' and self.config['block_sequence'] == 'enc'):

                self.register_parameter(name=f'weights_actor', param=torch.nn.Parameter(init_weights((self.num_layers, size_vqc)), requires_grad=True))
                self.register_parameter(name=f'weights_critic', param=torch.nn.Parameter(init_weights((self.num_layers, size_vqc)), requires_grad=True))

            if self.use_input_scaling_actor:
                self.register_parameter(name=f'input_scaling_actor', param=torch.nn.Parameter(torch.tensor(np.full((self.num_layers, size_input_scaling), 1.)), requires_grad=True))
            if self.use_input_scaling_critic:
                self.register_parameter(name=f'input_scaling_critic', param=torch.nn.Parameter(torch.tensor(np.full((self.num_layers, size_input_scaling), 1.)), requires_grad=True))

            if self.use_output_scaling_actor:
                if isinstance(self.init_output_scaling_actor, list):
                    self.register_parameter(name='output_scaling_actor', param=torch.nn.Parameter(torch.tensor(np.full(self.num_outputs, self.config['init_output_scaling_actor'][0])), requires_grad=True))
                else:
                    self.register_parameter(name='output_scaling_actor', param=torch.nn.Parameter(torch.tensor(self.config['init_output_scaling_actor']), requires_grad=True))
           
            if self.use_output_scaling_critic:
                if isinstance(self.init_output_scaling_critic, list):
                    self.register_parameter(name='output_scaling_critic', param=torch.nn.Parameter(torch.tensor(np.full(self.num_qubits, self.config['init_output_scaling_critic'][0])), requires_grad=True))
                else:
                    self.register_parameter(name='output_scaling_critic', param=torch.nn.Parameter(torch.tensor(self.config['init_output_scaling_critic']), requires_grad=True))
            
            if self.gradient_clipping:
                self.weights_actor.register_hook(lambda grad: torch.clip(grad, -1., 1.))
                self.weights_critic.register_hook(lambda grad: torch.clip(grad, -1., 1.))
                                
            dev_actor = qml.device(self.config['backend_name'], wires=self.num_qubits)
            dev_critic = qml.device(self.config['backend_name'], wires=self.num_qubits)

            self.qnode_actor = qml.QNode(vqc_switch[self.config['vqc_type'][0]], dev_actor, interface=self.config['interface'], diff_method=self.config['diff_method']) #, argnum=0)
            self.qnode_critic = qml.QNode(vqc_switch[self.config['vqc_type'][0]], dev_critic, interface=self.config['interface'], diff_method='adjoint') #, argnum=0)

            if self.use_classical_layer:
                self.classical_layer_actor = nn.Linear(in_features=self.obs_space.shape[0], out_features=self.num_actions, dtype=torch.float32)
                self.classical_layer_critic = nn.Linear(in_features=self.obs_space.shape[0], out_features=1, dtype=torch.float32)
           
        elif self.mode == 'classical':
            if self.config['activation_function'] == 'relu':
                activation = nn.ReLU()
            elif self.config['activation_function'] == 'leaky_relu':
                activation = nn.LeakyReLU()
            if self.config['activation_function'] == 'tanh':
                activation = nn.Tanh()
            if len(self.layer_size) == 1:
                self.actor_network = nn.Sequential(nn.Linear(in_features=self.obs_space.shape[0], out_features=self.layer_size[0]),
                                                    activation,
                                                    nn.Linear(in_features=self.layer_size[0], out_features=self.num_outputs))
                self.critic_network = nn.Sequential(nn.Linear(in_features=self.obs_space.shape[0], out_features=self.layer_size[0]),
                                                    activation,
                                                    nn.Linear(in_features=self.layer_size[0], out_features=1))
            elif len(self.layer_size) == 2:
                
                self.actor_network = nn.Sequential(nn.Linear(in_features=self.obs_space.shape[0], out_features=self.layer_size[0]),
                                                    activation,
                                                    nn.Linear(in_features=self.layer_size[0], out_features=self.layer_size[1]),
                                                    activation,
                                                    nn.Linear(in_features=self.layer_size[1], out_features=2))
                self.critic_network = nn.Sequential(nn.Linear(in_features=self.obs_space.shape[0], out_features=self.layer_size[0]),
                                                    activation,
                                                    nn.Linear(in_features=self.layer_size[0], out_features=self.layer_size[1]),
                                                    activation,
                                                    nn.Linear(in_features=self.layer_size[1], out_features=1))
            elif len(self.layer_size) == 3:
                self.actor_network = nn.Sequential(nn.Linear(in_features=self.obs_space.shape[0], out_features=self.layer_size[0]),
                                                    activation,
                                                    nn.Linear(in_features=self.layer_size[0], out_features=self.layer_size[1]),
                                                    activation,
                                                    nn.Linear(in_features=self.layer_size[1], out_features=self.layer_size[2]),
                                                    activation,
                                                    nn.Linear(in_features=self.layer_size[2], out_features=self.num_outputs))
                self.critic_network = nn.Sequential(nn.Linear(in_features=self.obs_space.shape[0], out_features=self.layer_size[0]),
                                                    activation,
                                                    nn.Linear(in_features=self.layer_size[1], out_features=self.layer_size[1]),
                                                    activation,
                                                    nn.Linear(in_features=self.layer_size[1], out_features=self.layer_size[2]),
                                                    activation,
                                                    nn.Linear(in_features=self.layer_size[2], out_features=1))
                

    def forward(self, input_dict, state, seq_lens):
        
        state = input_dict['obs']

        if self.mode == 'quantum':

            # Check the encoding block type in order to adapt obs/state accordingly
            if 'double' in self.config['vqc_type'][0]:
                state = torch.concat([state, state], dim=1)
            elif 'triple' in self.config['vqc_type'][0]:
                state = torch.concat([state, state, state], dim=1)
            elif 'circular' in self.config['vqc_type'][0]:
                reps = round((self.num_qubits*self.num_layers)/state.shape[1] + 0.5)
                state = torch.concat([state for _ in range(reps)], dim=1)
            # else:
            #     if not isinstance(self.obs_space.original_space, Dict): 
            #         state = torch.reshape(state, (-1, self.obs_space.shape[0]))


            # If vqc_type is relu or qcnn, two function calls are required
            if 'relu' in self.config['vqc_type'][0]:    
                activations_actor = self.qnode_activations(theta=state, weights=self._parameters, config=self.config, type='actor', activations=None)
                activations_critic = self.qnode_activations(theta=state, weights=self._parameters, config=self.config, type='critic', activations=None)
                prob = self.qnode(theta=state, weights=self._parameters, config=self.config, type='actor', activations=activations_actor)
                value = self.qnode(theta=state, weights=self._parameters, config=self.config, type='critic', activations=activations_critic)
            elif 'qcnn' in self.config['vqc_type'][0]:    
                activations_actor = self.qnode(theta=state, weights=self._parameters, config=self.config, type='activations_actor', activations=None)
                activations_actor = torch.reshape(activations_actor, (-1, 4))
                prob = self.qnode(theta=state, weights=self._parameters, config=self.config, type='actor', activations=activations_actor)
                value = self.qnode(theta=state, weights=self._parameters, config=self.config, type='critic', activations=None)
            else:
                prob = self.qnode_actor(theta=state, weights=self._parameters, config=self.config, type='actor', activations=None, H=None)
                
                if self.config['measurement_type_critic'] == 'hamiltonian':
                    value = []
                    for i in range(state['linear_0'].shape[0]):
                        tmp_state = {}
                        tmp_state['linear_0'] = np.reshape(state['linear_0'][i], (-1, *state['linear_0'].shape[1:]))
                        tmp_state['quadratic_0'] = np.reshape(state['quadratic_0'][i], (-1, *state['quadratic_0'].shape[1:]))

                        H_linear = qml.Hamiltonian(
                            [tensor[1] for tensor in state['linear_0'][i]],
                            [qml.PauliZ(int(key[0])) for key in state['linear_0'][i]]
                        )
                        H_quadratic = qml.Hamiltonian(
                            [tensor[2] for tensor in state['quadratic_0'][i]],
                            [qml.PauliZ(int(key[0]))@qml.PauliZ(int(key[1])) for key in state['quadratic_0'][i]]
                        )
                        H = H_quadratic - H_linear
                        value.append(self.qnode_critic(theta=tmp_state, weights=self._parameters, config=self.config, type='critic', activations=None, H=H))
                    value = torch.vstack(value)
                else:
                    value = self.qnode_critic(theta=state, weights=self._parameters, config=self.config, type='critic', activations=None, H=None)

            if isinstance(self.action_space, MultiDiscrete):
                
                prob = (prob + torch.ones(self.num_qubits))/2
                prob = torch.reshape(prob, (-1,self.num_qubits)).repeat_interleave(2, dim=1)*torch.tensor([1., -1.]).repeat(self.num_qubits)
                prob = prob + torch.tensor([0., 1.]).repeat(self.num_qubits)

            if self.use_output_scaling_critic:
                value = torch.reshape(postprocessing(value, self.config, self.num_outputs, self._parameters, 'critic'), (-1,))
            else:
                if len(value.shape) == 2:
                    value = value[:,0]
                else:
                    value = torch.reshape(value, (-1,))

            if self.use_output_scaling_actor:
                logits = torch.reshape(postprocessing(prob, self.config, self.num_outputs, self._parameters, 'actor'), (-1, self.num_outputs))
            else:
                logits = prob[:,:self.num_outputs]

            self._value_out = value

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
                logits = torch.reshape(logits, (-1, self.action_space.shape[0], 2))
                logits = torch.softmax(logits/0.01, dim=2)
                logits = torch.reshape(logits, (-1, 10))
            
            else:
                logits = self.actor_network(state)
                self._logits = logits
                self._value_out = torch.reshape(self.critic_network(state), (-1,))
        self.counter += 1

        return logits, []

    def value_function(self):
        return self._value_out

    def policy_function(self):
        return self._logits
    