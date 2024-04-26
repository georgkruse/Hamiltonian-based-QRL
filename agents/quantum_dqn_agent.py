from abc import ABC
import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from gymnasium.spaces import Box, Discrete, MultiBinary, MultiDiscrete, Dict
import pennylane as qml
import torch
import numpy as np
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from ray.rllib.utils.annotations import override
from ray.rllib.algorithms.simple_q.simple_q import SimpleQ, SimpleQConfig
from ray.rllib.algorithms.simple_q.simple_q_torch_policy import SimpleQTorchPolicy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import (
    TorchDistributionWrapper,
)
from ray.rllib.utils.torch_utils import l2_loss
from ray.rllib.utils.typing import TensorStructType, TensorType
from typing import Any, Dict, List, Tuple, Type, Union
from circuits.postprocessing import *
from circuits.quantum_circuits import vqc_switch
from torch.nn import functional as F
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils.exploration.epsilon_greedy import EpsilonGreedy
from typing import Optional
from ray.rllib.models.action_dist import ActionDistribution
import gymnasium as gym
from ray.rllib.utils.schedules import Schedule, PiecewiseSchedule
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.framework import try_import_tf, try_import_torch, get_variable
from ray.rllib.models.torch.torch_action_dist import TorchMultiActionDistribution
import tree  # pip install dm_tree
import random
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.policy.torch_mixins import LearningRateSchedule
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2

class CustomEpsilonGreedy(Exploration):
    """Epsilon-greedy Exploration class that produces exploration actions.

    When given a Model's output and a current epsilon value (based on some
    Schedule), it produces a random action (if rand(1) < eps) or
    uses the model-computed one (if rand(1) >= eps).
    """

    def __init__(
        self,
        action_space: gym.spaces.Space,
        *,
        framework: str,
        initial_epsilon: float = 1.0,
        final_epsilon: float = 0.05,
        warmup_timesteps: int = 0,
        epsilon_timesteps: int = int(1e5),
        epsilon_schedule: Optional[Schedule] = None,
        **kwargs,
    ):
        """Create an EpsilonGreedy exploration class.

        Args:
            action_space: The action space the exploration should occur in.
            framework: The framework specifier.
            initial_epsilon: The initial epsilon value to use.
            final_epsilon: The final epsilon value to use.
            warmup_timesteps: The timesteps over which to not change epsilon in the
                beginning.
            epsilon_timesteps: The timesteps (additional to `warmup_timesteps`)
                after which epsilon should always be `final_epsilon`.
                E.g.: warmup_timesteps=20k epsilon_timesteps=50k -> After 70k timesteps,
                epsilon will reach its final value.
            epsilon_schedule: An optional Schedule object
                to use (instead of constructing one from the given parameters).
        """
        assert framework is not None
        super().__init__(action_space=action_space, framework=framework, **kwargs)

        self.epsilon_schedule = from_config(
            Schedule, epsilon_schedule, framework=framework
        ) or PiecewiseSchedule(
            endpoints=[
                (0, initial_epsilon),
                (warmup_timesteps, initial_epsilon),
                (warmup_timesteps + epsilon_timesteps, final_epsilon),
            ],
            outside_value=final_epsilon,
            framework=self.framework,
        )

        # The current timestep value (tf-var or python int).
        self.last_timestep = get_variable(
            np.array(0, np.int64),
            framework=framework,
            tf_name="timestep",
            dtype=np.int64,
        )

        # Build the tf-info-op.
        if self.framework == "tf":
            self._tf_state_op = self.get_state()

    @override(Exploration)
    def get_exploration_action(
        self,
        *,
        action_distribution: ActionDistribution,
        timestep: Union[int, TensorType],
        explore: Optional[Union[bool, TensorType]] = True,
    ):

        if self.framework in ["tf2", "tf"]:
            return self._get_tf_exploration_action_op(
                action_distribution, explore, timestep
            )
        else:
            return self._get_torch_exploration_action(
                action_distribution, explore, timestep
            )

    def _get_torch_exploration_action(
        self,
        action_distribution: ActionDistribution,
        explore: bool,
        timestep: Union[int, TensorType],
    ) -> "torch.Tensor":
        """Torch method to produce an epsilon exploration action.

        Args:
            action_distribution: The instantiated
                ActionDistribution object to work with when creating
                exploration actions.

        Returns:
            The exploration-action.
        """
        q_values = action_distribution.inputs
        self.last_timestep = timestep
        exploit_action = action_distribution.deterministic_sample()
        batch_size = q_values.size()[0]
        action_logp = torch.zeros(batch_size, dtype=torch.float)

        # Explore.
        if explore:
            # Get the current epsilon.
            epsilon = self.epsilon_schedule(self.last_timestep)
            if isinstance(action_distribution, TorchMultiActionDistribution):
                exploit_action = tree.flatten(exploit_action)
                for i in range(batch_size):
                    if random.random() < epsilon:
                        # TODO: (bcahlit) Mask out actions
                        random_action = tree.flatten(self.action_space.sample())
                        for j in range(len(exploit_action)):
                            exploit_action[j][i] = torch.tensor(random_action[j])
                exploit_action = tree.unflatten_as(
                    action_distribution.action_space_struct, exploit_action
                )

                return exploit_action, action_logp

            else:
                # Mask out actions, whose Q-values are -inf, so that we don't
                # even consider them for exploration.
                random_valid_action_logits = torch.where(
                    q_values <= FLOAT_MIN,
                    torch.ones_like(q_values) * 0.0,
                    torch.ones_like(q_values),
                )
                # A random action.
                random_actions = torch.squeeze(
                    torch.multinomial(random_valid_action_logits, 1), axis=1
                )

                # Pick either random or greedy.
                action = torch.where(
                    torch.empty((batch_size,)).uniform_().to(self.device) < epsilon,
                    random_actions,
                    exploit_action,
                )

                return action, action_logp
        # Return the deterministic "sample" (argmax) over the logits.
        else:
            return exploit_action, action_logp


class QRL_DQNTorchPolicy(SimpleQTorchPolicy, LearningRateSchedule, TorchPolicyV2):
    """Pytorch Policy Class used with DQN"""
    
    @override(SimpleQTorchPolicy)
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
    
    @override(SimpleQTorchPolicy)
    def loss(
            self,
            model: ModelV2,
            dist_class: Type[TorchDistributionWrapper],
            train_batch: SampleBatch,
        ) -> Union[TensorType, List[TensorType]]:
        """Compute loss for SimpleQ.

        Args:
            model: The Model to calculate the loss for.
            dist_class: The action distr. class.
            train_batch: The training data.

        Returns:
            The SimpleQ loss tensor given the input batch.
        """
        target_model = self.target_models[model]

        # q network evaluation
        q_t = self._compute_q_values(
            model, train_batch[SampleBatch.CUR_OBS], is_training=True
        )

        # target q network evalution
        q_tp1 = self._compute_q_values(
            target_model,
            train_batch[SampleBatch.NEXT_OBS],
            is_training=True,
        )

        # q scores for actions which we know were selected in the given state.
        one_hot_selection = F.one_hot(
            train_batch[SampleBatch.ACTIONS].long(), self.action_space.n
        )
        q_t_selected = torch.sum(q_t * one_hot_selection, 1)

        # compute estimate of best possible value starting from state at t + 1
        dones = train_batch[SampleBatch.TERMINATEDS].float()
        q_tp1_best_one_hot_selection = F.one_hot(
            torch.argmax(q_tp1, 1), self.action_space.n
        )
        q_tp1_best = torch.sum(q_tp1 * q_tp1_best_one_hot_selection, 1)
        q_tp1_best_masked = (1.0 - dones) * q_tp1_best

        # compute RHS of bellman equation
        q_t_selected_target = (
            train_batch[SampleBatch.REWARDS] + self.config["gamma"] * q_tp1_best_masked
        )

        # Compute the error (Square/Huber).
        td_error = q_t_selected - q_t_selected_target.detach()
        loss = torch.mean(l2_loss(td_error))

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["loss"] = loss
        # TD-error tensor in final stats
        # will be concatenated and retrieved for each individual batch item.
        model.tower_stats["td_error"] = td_error

        return loss

class QRL_DQN(SimpleQ):  
    @override(SimpleQ)
    def get_default_policy_class(cls, config):
        if config["framework"] == "torch":
            return QRL_DQNTorchPolicy
        
class QuantumDQN_Model(TorchModelV2,nn.Module,ABC):
    def __init__(self,obs_space,action_space,num_actions,model_config,name):
        TorchModelV2.__init__(
            self, obs_space,action_space,num_actions,model_config,name
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
            self.num_inputs = 5 #sum([box.shape[0] for box in self.obs_space.values()])
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
        self.num_scaling_params = self.config['num_scaling_params']
        self.action_masking = self.config['action_masking']

        self.use_classical_layer = self.config['use_classical_layer']        
        self.layer_size = self.config['layer_size']

        self.weight_logging_interval = self.config['weight_logging_interval']
        self.weight_plotting = self.config['weight_plotting']
        self._value_out = None

        def init_weights(size):
            if self.config['init_variational_params_mode'] == 'plus-zero-uniform':
                return torch.FloatTensor(*size,).uniform_(0, self.init_variational_params)
            elif self.config['init_variational_params_mode'] == 'plus-minus-uniform':
                return torch.FloatTensor(*size,).uniform_(-self.init_variational_params, self.init_variational_params)
            elif self.config['init_variational_params_mode'] == 'plus-zero-normal':
                return torch.FloatTensor(*size,).normal_(0., self.init_variational_params)
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

            if self.config['measurement_type_actor'] == 'edge':
                prob = []
                for i in range(state['quadratic_0'].shape[1]):
                    if not state['quadratic_0'][0,0,0].to(torch.int).item() + state['quadratic_0'][0,0,1].to(torch.int).item() == 0:
                        self.config['edge_measurement'] = state['quadratic_0'][0,i,:2]
                    else:
                        self.config['edge_measurement'] = [torch.tensor(0.), torch.tensor(1.)]

                    prob.append(self.qnode_actor(theta=state, weights=self._parameters, config=self.config, type='actor', activations=None, H=None))

                q_values = torch.reshape(torch.stack(prob).T, (-1, state['quadratic_0'].shape[1]))
            else:
                q_values = self.qnode_actor(theta=state, weights=self._parameters, config=self.config, type='actor', activations=None, H=None)

            if self.use_output_scaling_actor:
                logits = torch.reshape(postprocessing(q_values, self.config, self.num_outputs, self._parameters, 'actor'), (-1, self.num_outputs))
            else:
                logits = q_values

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
                linear = state['linear_0'][:,:,1]
                quadratic = state['quadratic_0'][:,:,1]
                input_state = torch.from_numpy(np.concatenate([linear, quadratic], axis=1))
                logits = self.actor_network(input_state)
            
            else:
                logits = self.actor_network(state)
                self._logits = logits
        self.counter += 1

        if self.config['output_scaling_schedule']:
            self.counter += 1
            if self.counter % 2000 == 0:
                if self.counter <= 75000:
                    if logits.shape[0] == 1.:
                        self._parameters['output_scaling_actor'] += 0.5
                    else:
                        self.counter -= 1
                print('output_scaling_actor:', self._parameters['output_scaling_actor'])

        elif self.config["problem_scaling"]:
            if not state['quadratic_0'][0,0,0].to(torch.int).item() + state['quadratic_0'][0,0,1].to(torch.int).item() == 0:
                nodes = state["scale_qvalues"].shape[1]
                batch_dim = state["scale_qvalues"].shape[0]
                for batch in range(batch_dim):
                    for node in range(nodes):
                        logits[batch][node] *= state["scale_qvalues"][batch][node][1]


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
                        if batch_dim == 1:
                            for batch in range(batch_dim):
                                for node in range(nodes):
                                    if state["annotations"][batch][node][1] == 0:
                                        logits[batch][node] = -np.inf
                        else:
                            for batch in range(batch_dim):
                                for node in range(nodes):
                                    if state["annotations"][batch][node][1] == 0:
                                        logits[batch][node] = -10000
            
        
        return logits, []

