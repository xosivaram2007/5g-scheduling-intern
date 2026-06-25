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
from stable_baselines3.common.callbacks import CheckpointCallback

from envs.scheduling_env import SchedulingEnv

# Create checkpoints folder
os.makedirs(
    "models/checkpoints",
    exist_ok=True
)

# Create environment with 30 UEs
env = SchedulingEnv(
    num_ues=30
)

# Add monitor for logging rewards
env = Monitor(
    env,
    filename="training_log_30ue_checkpoints"
)

# Create DQN model
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    gamma=0.99,
    verbose=1
)

# Save model every 5,000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path="models/checkpoints",
    name_prefix="dqn_30ue"
)

# Train for 50,000 steps
model.learn(
    total_timesteps=50000,
    callback=checkpoint_callback
)

# Save final model
model.save(
    "models/dqn_30ue_50k"
)

print("\nTraining complete!")
print("Final model saved as: models/dqn_30ue_50k.zip")
print("Checkpoint models saved in: models/checkpoints/")