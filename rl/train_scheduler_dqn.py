import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import DQN
from envs.scheduling_env import SchedulingEnv

env = SchedulingEnv()

model = DQN(
    "MlpPolicy",
    env,
    verbose=1
)

print("Starting Training...")

model.learn(total_timesteps=5000)
 
model.save("scheduler_dqn")

print("Training Complete!")