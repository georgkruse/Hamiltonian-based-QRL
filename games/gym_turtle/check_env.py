import gymnasium as gym
from stable_baselines3.common.env_checker import check_env

from envs.env_compact import CompactEnv


env = CompactEnv()
print("checkEnv(env):\n\n", check_env(env))