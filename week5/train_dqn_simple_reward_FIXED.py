import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

# The simplified-reward environment is the ablated component itself
from envs.scheduling_env_simple_reward import SchedulingEnv

# Create environment with 30 UEs (matches Full DQN config in train_dqn_30.py)
env = SchedulingEnv(
    num_ues=30
)

env = Monitor(
    env,
    filename="training_log_simple_reward_30ue"
)

model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    gamma=0.99,
    verbose=1
)

# Train for 50,000 steps (matches Full DQN training length)
model.learn(
    total_timesteps=50000
)

model.save(
    "models/dqn_simple_reward_30ue"
)

print("\nTraining complete!")
print(
    "Model saved as: models/dqn_simple_reward_30ue.zip"
)