from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import RolloutWorker, Episode
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from typing import Dict


class MaxCut_Callback(DefaultCallbacks):
    def __init__(self):
        self.timestep = 0
    
    def on_episode_start(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            env_index: int,
            **kwargs,
    ):
        
        self.timestep = worker.env.timestep
        
    def on_episode_end(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            env_index: int,
            **kwargs,
    ):
        optimal_cost = worker.env.optimal_cost(timestep = self.timestep)
        approximation_ratio = float(episode.total_reward)/optimal_cost
        episode.custom_metrics["approximation_ratio"] = approximation_ratio
        # print(approximation_ratio)

class TSP_Callback(DefaultCallbacks):
    def __init__(self):
        self.timestep = 0
    
    def on_episode_start(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            env_index: int,
            **kwargs,
    ):
        
        self.timestep = worker.env.timestep
        
    def on_episode_end(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            env_index: int,
            **kwargs,
    ):
        approximation_ratio = worker.env.ratio
        episode.custom_metrics["approximation_ratio"] = approximation_ratio
        # print(approximation_ratio)

class KnapsackCallback(DefaultCallbacks):
    def __init__(self):
        self.timestep = 0
    def on_episode_start(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            env_index: int,
            **kwargs,
    ):
        
        self.timestep = worker.env.env.timestep

    def on_episode_end(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            env_index: int,
            **kwargs,
    ):
        
        optimal_action = worker.env.env.solution_str[self.timestep]
        optimal_action = list(optimal_action)
        optimal_action = [int(a) for a in optimal_action]
        optimal_reward = worker.env.env.calculate_reward(optimal_action,timestep = self.timestep)
        approximation_ratio = float(episode.total_reward)/optimal_reward
        episode.custom_metrics["approximation_ratio"] = approximation_ratio
        # episode.custom_metrics["episode_reward"] = float(episode.total_reward)