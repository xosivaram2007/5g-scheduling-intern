import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from envs.scheduling_env import SchedulingEnv

os.makedirs("results", exist_ok=True)

env = SchedulingEnv()

env = Monitor(
    env,
    filename="results/training_log"
)

model = DQN(
    "MlpPolicy",
    env,
    verbose=1
)

print("Starting Training...")

model.learn(total_timesteps=10000)

model.save("scheduler_dqn")

print("Training Complete!")