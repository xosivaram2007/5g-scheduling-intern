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

from envs.scheduling_env import SchedulingEnv


# Create environment with 30 UEs (matches Full DQN config in train_dqn_30.py)
env = SchedulingEnv(
    num_ues=30
)

# Add monitor for logging rewards
env = Monitor(
    env,
    filename="training_log_no_replay_30ue"
)

# Create DQN model WITHOUT replay buffer (buffer_size=1 is the ablated component
# — this is the ONLY thing that should differ from train_dqn_30.py)
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    gamma=0.99,
    buffer_size=1,
    verbose=1
)

# Train for 50,000 steps (matches Full DQN training length)
model.learn(
    total_timesteps=50000
)

# Save model
model.save(
    "models/dqn_no_replay_30ue"
)

print("\nTraining complete!")
print("Model saved as: models/dqn_no_replay_30ue.zip")