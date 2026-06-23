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


# Create environment with 15 UEs
env = SchedulingEnv(
    num_ues=15
)

# Add monitor for logging rewards
env = Monitor(
    env,
    filename="training_log_15ue"
)

# Create DQN model
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    gamma=0.99,
    verbose=1
)

# Train for 50,000 steps
model.learn(
    total_timesteps=50000
)

# Save model
model.save(
    "models/dqn_15ue"
)

print("\nTraining complete!")
print("Model saved as: models/dqn_15ue.zip")