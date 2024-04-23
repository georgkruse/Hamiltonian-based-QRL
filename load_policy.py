from ray.rllib.policy.policy import Policy
from games.ev_wrapper import EV_Game_QUBO_Contextual_Bandit_Wrapper, EV_Game_QUBO_Sequential
from games.mvc_wrapper import MVC_Wrapper
from games.maxcut_wrapper import MAXCUTCONTEXTUALBANDIT_Wrapper
import yaml
import numpy as np
import itertools

checkpoint_path = "/home/users/coelho/quantum-computing/QRL/logs/qpg/maxcut_contextual_bandit/first_results/2024-02-15--15-48-32_QRL_QPG/QRL_PG_MAXCUTCONTEXTUALBANDIT_49df0_00000_0_graph_encoding_type=eqc_2024-02-15_15-48-32/checkpoint_000000"
agent = Policy.from_checkpoint(checkpoint_path)['default_policy']  

with open('/home/users/coelho/quantum-computing/QRL/logs/qpg/maxcut_contextual_bandit/first_results/2024-02-15--15-48-32_QRL_QPG/alg_config.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

env = MAXCUTCONTEXTUALBANDIT_Wrapper(config['env_config'])

theta, _ = env.reset(use_specific_timestep = True, timestep = 4)
# Do some reshaping due to batch dimension (not always required, depends on vqc structure)
theta['linear_0'] = np.reshape(theta['linear_0'], (-1, *theta['linear_0'].shape))
theta['quadratic_0'] = np.reshape(theta['quadratic_0'], (-1, *theta['quadratic_0'].shape))

actions = agent.compute_actions(theta)

logits = actions[2]["action_dist_inputs"].reshape(config["env_config"]["nodes"],2)

probabilities = np.zeros((config["env_config"]["nodes"],2))

def softmax(logits):
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)

for i in range(len(logits)):
    probabilities[i] = softmax(logits[i])

probabilities = probabilities[:,1]

# Generate all possible combinations of actions
actions = list(itertools.product([0, 1], repeat=len(probabilities)))

# Calculate the probability for each combination of actions
action_probabilities = []
for action in actions:
    prob = 1.0
    for i, val in enumerate(action):
        prob *= probabilities[i] if val == 1 else (1 - probabilities[i])
    action_probabilities.append((action, prob))

# Sort the action probabilities by probability in descending order
action_probabilities.sort(key=lambda x: x[1], reverse=True)

# Select the top 10 most probable actions
top_10_actions = action_probabilities[:10]

for action, prob in top_10_actions:
    print("Action:", action, "Probability:", prob)