import matplotlib.pyplot as plt
import numpy as np
import yaml
from ray.rllib.policy.policy import Policy
import pandas as pd

from ray.rllib.policy.policy import Policy
from games.ev_wrapper import EV_Game_QUBO_Bandit_Wrapper
from games.uc.uc_game import UC_game
from games.jssp.jssp_game import JSSP_game
from games.fp.fp_game import FP_game
from games.maxcut_wrapper import MAXCUTWEIGHTEDDYNAMIC_Wrapper
import yaml
import numpy as np
import itertools

# path = 'logs/maxcut_weighted/dynamic/5nodes/2024-03-08--14-43-22_QRL_QPG/'
path = 'logs/maxcut_weighted/dynamic/5nodes/2024-03-08--13-37-16_QRL_QPG/'

# checkpoint_path = path + 'QRL_PG_MAXCUTWEIGHTEDDYNAMIC_d4de6_00001_1,/checkpoint_000001'
checkpoint_path = path + 'QRL_PG_MAXCUTWEIGHTEDDYNAMIC_98acb_00000_0,/checkpoint_000500'

agent = Policy.from_checkpoint(checkpoint_path)['default_policy']  

with open(path + '/alg_config.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


env = MAXCUTWEIGHTEDDYNAMIC_Wrapper(config['env_config'])

# env = UC_game(config['env_config'])
# env = JSSP_game(config['env_config'])
# env = FP_game(config['env_config'])
theta, _ = env.reset()

# Do some reshaping due to batch dimension (not always required, depends on vqc structure)
theta['linear_0'] = np.reshape(theta['linear_0'], (-1, *theta['linear_0'].shape))
theta['quadratic_0'] = np.reshape(theta['quadratic_0'], (-1, *theta['quadratic_0'].shape))
theta['a_0'] = np.reshape(theta['a_0'], (-1, *theta['a_0'].shape))
optimal_rewards = []

for eval in range(0,40*48, 48):

    epoch_demand = []
    epoch_disp = []
    epoch_commitment = []
    rewards = []
    actions = []
    wind_creation = []
    over_delivery = []
    done = False
    theta, _ = env.reset()
    timesteps = 0

    while not done: 

        # Do some reshaping due to batch dimension (not always required, depends on vqc structure)
        theta['linear_0'] = np.reshape(theta['linear_0'], (-1, *theta['linear_0'].shape))
        theta['quadratic_0'] = np.reshape(theta['quadratic_0'], (-1, *theta['quadratic_0'].shape))
        theta['a_0'] = np.reshape(theta['a_0'], (-1, *theta['a_0'].shape))

        # action, _, _ = agent.compute_single_action(theta)
        actions = agent.compute_actions(theta, explore=False)
        # print(theta['linear_0'])
        # print(theta['quadratic_0'])
        # print(actions[0][0])

        theta, reward, done, _, _ = env.step(actions[0][0])
        timesteps += 1
    optimal_rewards.append(reward/env.env.optimal_cost_)
    print(reward, env.env.optimal_cost_, reward/env.env.optimal_cost_)
print(np.mean(optimal_rewards))
#     fig, ax1 = plt.subplots()
#     ax2 = ax1.twinx()
#     # ax2.set_yscale('log')

#     ax1.set_title(f'total day cost: {sum(rewards)}')
#     ax2.plot(np.arange(len(rewards)), rewards, label='reward agent', c='black', linestyle='--')
#     ax2.plot(np.arange(len(rewards)), optimal_rewards, label='reward optimal', linestyle='--')

#     # ax2.plot(np.arange(len(over_delivery)), over_delivery, label='over delivery', linestyle='--')
#     ax1.set_xlabel('timestep')
#     ax2.set_ylabel('rewards')

#     ax1.set_ylabel('Power [MW]')
#     ax1.plot(np.arange(env.env.episode_length), env.env.time_series[:env.env.episode_length], label='demand')
#     ax1.plot(np.arange(env.env.episode_length), env.env.history, label='produced')
#     ax1.plot(np.arange(env.env.episode_length), np.sum(np.stack(env.env.optimal_actions)*env.env.generator_outputs, axis=1), label='optimal')

#     # ax1.plot(np.arange(len(wind_creation)), wind_creation, label='wind energy')
#     # ax1.plot(np.arange(len(epoch_disp)), epoch_disp, label='produced')
#     # ax1.plot(np.arange(len(epoch_commitment)), epoch_commitment, label='commitment', c='red')
#     # ax1.plot(np.arange(len(env.env.episode_forecast)), env.env.episode_forecast, label='forcast', c='green')
#     ax1.legend(loc='lower left')
#     ax2.legend(loc='lower right')
#     ax1.minorticks_on()
#     ax1.grid(which='both', alpha=0.4)
#     plt.savefig(f'demand_full_episode_static_5_1000_{eval}.png')
#     plt.close()
# print(full_cost)