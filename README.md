# Hamiltonian-based-QRL

This repository contains the code for the paper **Hamiltonian-based Quantum Reinforcement Learning for Neural Combinatorial Optimization**. 

To replicate the results from the publications, please refer to the config files in the `configs` folder and run them via the `main.py` file. 

**DISCLAIMER** The knapsack example, qaoa as well as the variance calculations are still to be fixed.

### If you have any questions regarding the code, feel free to create an issue or to drop a message!

To create the appropriate conda environment run (for more info refer to [conda cheat sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf))

```
conda env create --file environment.yml
```

Most experiments can be carried out by simply executing the main.py with the --path flag set to the path of your config file. Examplary config files can be found in the configs folder.

```
python main.py --path configs/qpg/qpg_maxcut.yml
```

This will execute a ray tune trainable run and execute them in parallel on your system. Be sure to first configure the number of cpus in the config file before execution.

The paper **Hamiltonian-based Quantum Reinforcement Learning** is currently under review and will be published soon. 

# General Information

Every config file contains four main blocks. In the first block you need to specify the name and type of the algorithm. You need to set

- **ray_local_mode**: If you encounter an error, set this to True in order to be able to **DEBUG**
- **checkpoint_freq** and **checkpoint_at_end**: this integer depends on the number of **training_iterations**
- **ray_logging_path**: Here the data/logs will be saved
- **total_number_cpus**: the maximal amount of cpus you want to use for your training run
- **ray_num_trial_samples**: How many seeds you want to start with the same conifg (e.g. 5 for 5 parallel trainings)
- **run_sections**: Set to **qrl_training** and/or **plotting** 

```yaml
type:                             QRL                   # choose a type: QRL, GA, ES
alg:                              QPG                   # name of the algorithm [QPPO, QPG, QDQN]
seed:                             42                    # seed for ray alogorithms/pytorch
constant_seed:                    False                 # setting the same seed accross all runs
checkpoint_at_end:                False                 # create checkpoint at the end of training
checkpoint_freq:                  100                   # set checkpoint frequency, depends on training iterations
ray_local_mode:                   True                 # set local_mode of ray to True for debugging
ray_logging_path:                 logs/w-maxcut/nodes5/qpg            # logging directory
total_num_cpus:                   10                    # total number of cpus
total_num_gpus:                   0                     # total number of gpus
ray_num_trial_samples:            1                     # number of training seeds per combination
training_iterations:              250                   # number of training iterations
run_sections:                                           # specifiy the code you want to run
                                - qrl_training
                                - plotting

```

In the second block, the environment config needs to be set. For e.g. MaxCut only 5 or 10 nodes can be selected. You need to consider:

- Depending on the number of **nodes** the **vqc_type** variable needs to be change also!
    - If **nodes** = 5 also **vqc_type** == **[vqc_generator, 5]**
- Note that the change of the action_space depends on the **measurement_type_actor** variable!
    - If **action_space** == **discrete_edges** also **measurement_type_actor** == **edges**
    - If **action_space** == **discrete_nodes** also **measurement_type_actor** == **exp**

All other variables my be changed and for additional information take a look at the environmen file.
```yaml
############################## Env Config #############################################
env:                              weightedMaxCut        # specifiy env string                              
env_config:
  nodes:                          5                     # numnber of graph nodes either set to 5 or 10
  mode:                           dynamic               # only dynamic implemented
  reward_mode:                    normal                # either normal or ratio
  annotation_value:               0                     # annotation value (either 0 or 2pi)
  constant:                       False                 # set the training graph constant for testing
  action_space:                   discrete_edges        # either 'discrete_nodes' or 'discrete_edges' measurements (measurement needs to be set to 'skolik' for discrete_edges)       
  reward_at_end:                  False                 # return reward only at end if set True
  callback:                       MaxCut_Callback       # callback to log additional information during training     
  path:                           /home/users/kruse/Hamiltonian-based-QRL # path to dataset needs to be specified (run pwd / current directory)
```

In the algorithm config block, you need to select torch as a framework. In order to efficiently train different vqc's whith different hyperparameter configurations, you can use the following notation when changing the default yaml.

E.g you want to train the same vqc with different learning rates, set the **lr** to

```yaml
lr: 
    - grid_search           # Do a grid search with number of seeds == ray_num_trial_samples
    - float                 # Define the type [float, int, string, list]
    - [0.01, 0.001, 0.0001] # List of parameters to select
```

If **ray_num_trial_samples** is set to 5 and **total_num_cpus** also to 15, than each hyperparameter configuration will be trained with 5 seeds in parallel (watch your RAM for larger qubit numbers).

You can now also edit the **graph_encoding_type** variable to 

```yaml
graph_encoding_type: 
    - grid_search           # Do a grid search with number of seeds == ray_num_trial_samples
    - string                # Define the type [float, int, string, list]
    - [sge-sgv, mge-sgv]    # List of parameters to select
```

This will now start for each of the **graph_encoding_type** a hyperparameter run with all the specified learning rates for **ray_num_trial_samples**, so a total of 30 trials. If **total_num_cpus** is set to 10, than it will sequentially execute the 30 trials with 10 trials in parallel.

## Important 

If you want to run the **sge-sgv** type of ansatzes, always set the **block_sequence** to **enc** and **encoding_type** to **graph_encoding**. If you want to run the **sge-sgv+hea** ansatz, set **block_sequence** to **enc_var_ent**.

```yaml
############################## Alg Config #############################################
algorithm_config:                                       # config for QRL training
  reuse_actors:                   True  
  framework:                      torch                 # ray framework, only torch supported.
  ###########################################################################
  lr:                             0.025                 # select lr for nn, variational params and input scaling params
  lr_output_scaling:              0.1                   # select lr for output scaling params
  num_layers:                     5                     # select number of layers of vqc (layer nn defined below)
  ###########################################################################
  mode:                           quantum               # select mode [classical, quantum]
  interface:                      torch                 # select pennylane interface, default: torch
  diff_method:                    adjoint               # select pennylane diff_method [adjoing, backprop, ...] 
  backend_name:                   lightning.qubit       # select pennylane backend [lightning.qubit, default.qubit, ...]
  custom_optimizer:               Adam                  # select the classical optimizer [Adam, RMSprop, LBFGS, ...] 
  ###########################################################################
  vqc_type:                       [vqc_generator, 5]    # select vqc_generator or other circuit generator function + number of qubits (take a look at vqc_switch)
  use_hadamard:                   True                  # Create equal superposition in the beginning [True, False]
  block_sequence:                 enc                   # select the block sequence, enc_var_ent == classical hea ansatz, graph_encoding only needs enc 
  encoding_type:                  graph_encoding        # data encoding type [angular_classical (RY_RZ), arctan_sigmoid, graph_encoding ... ]
  graph_encoding_type:            sge-sgv                # # if encoding_type=graph_encoding, than select [sge-sgv, mge-sgv, mge-mgv, hamiltonian-hea, angular-hea, angular, ...]
  use_input_scaling:              True                  # use input scaling [True, False] 
  init_input_scaling_actor:       [1.]                  # if list, then each gate gets one params, if single float, all have same param [[1.], 1., ...]
  num_scaling_params:             2                     # select the number of params, so e.g. 2 for angular_classical -> RY_RZ
  quadratic_gate:                 ZZ                    # ZZ, XX, YY
  linear_gate:                    RZ                    # RZ, RX, RY
  annotations_gate:               RX                    # RZ, RX, RY
  measurement_gate:               PauliZ                # PauliZ, PauliX, PauliY
  variational_type:               RZ_RY                 # select the gate sequence [RZ_RY, RY_RZ]
  num_variational_params:         2                     # select the number of params, so e.g. 2 for RZ_RY
  init_variational_params:        0.1                   # select initialization of the variational parameters
  init_variational_params_mode:   plus-zero-uniform     # plus-zero-uniform, plus-plus-normal, plus-zero-normal
  entangling_type:                chain                 # type of entanglement [chain, full, ...]
  entangling_gate:                CZ                    # type of entanglement gate [CNOT, CZ, CH, ...]
  measurement_type_actor:         edge                # type of measurement (check the python files for examples) (exp for discrete) exp_@_exp+exp
  use_output_scaling_actor:       True                  # use output scaling [True, False]
  init_output_scaling_actor:      [1.]                  # if list, then each qubit gets one param, if single float, all have same param [[1.], 1., ...]
  postprocessing_actor:           constant              # select postprocessing (check the file postprocessing.py)
  output_scaling_schedule:        False
  use_temperature:                False
  temperature_schedule:           [[0, 100_000], [1.00, 0.05]]   # number of increment steps, total steps
  ###########################################################################
  noise:                                                # use noise during training
    coherent:                     [False, 0.]           # bool + float for magnitude of used coherent noise
    depolarizing:                 [False, 0.001]        # bool + float for magnitude of used depolarizing noise
  layerwise_training:             False                 # layerwise training (DEPRECATED)
  gradient_clipping:              False                 # gradient clipping (DEPRECATED)
  use_classical_layer:            False                 # additional postprocessing (DEPRECATED)
  layer_size:                     [64, 64]              # classical NN, max 3 layers with in as number of neurons in the according layer
  activation_function:            relu                  # activation function of classical NN
  weight_logging_interval:        5000                  # weight logging + plotting interval (DEPRECATED)
  weight_plotting:                False                 # weight logging + plotting (DEPRECATED)
  ###########################################################################
  # More ray params
  explore:                        True


############################## Eval Config #############################################
evaluation:
  set_seed:               True 
  seed:                   42
  ###########################################################################
  plotting:
    mode:                         auto 
    y_axis:                       episode_reward_mean 
    path:                         logs/uc/paper_2/10units/2024-04-08--17-00-08_QRL_QPG
  
```

# [Examples]()

Examples as well as more documentation will be added upon request!

### Additional Information

In this repository additional implementations of e.g. Q-PPO as well as well as OpenAI gym wrappers can be found, which were mainly used in the paper. [Variational Quantum Circuit Design for Quantum Reinforcement Learning on Continuous Environments](https://www.scitepress.org/PublicationsDetail.aspx?ID=gnvuXCuulvU=&t=1) which can also be found on [arxiv](https://arxiv.org/abs/2312.13798).

```
@conference{icaart24,
author={Georg Kruse. and Theodora{-}Augustina Drăgan. and Robert Wille. and Jeanette Miriam Lorenz.},
title={Variational Quantum Circuit Design for Quantum Reinforcement Learning on Continuous Environments},
booktitle={Proceedings of the 16th International Conference on Agents and Artificial Intelligence - Volume 3: ICAART},
year={2024},
pages={393-400},
publisher={SciTePress},
organization={INSTICC},
doi={10.5220/0012353100003636},
isbn={978-989-758-680-4},
issn={2184-433X},
}
```



If you have any questions regarding the code or the paper, feel free to reach out. We are happy to help!
