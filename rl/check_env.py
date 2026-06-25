import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3.common.env_checker import check_env
from envs.scheduling_env import SchedulingEnv

env = SchedulingEnv()

check_env(env)

print("Environment passed SB3 checks!")