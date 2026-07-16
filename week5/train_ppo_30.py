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

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from envs.scheduling_env import SchedulingEnv

# Create environment with 30 UEs (mirrors week5/train_dqn_30.py, but for PPO —
# this is the run that was missing: evaluate_ppo.py never trained/evaluated
# PPO at num_ues=30, it silently used the SchedulingEnv default of 5)
env = SchedulingEnv(
    num_ues=30
)

# Add monitor for logging rewards
env = Monitor(
    env,
    filename="training_log_ppo_30ue"
)

# Create PPO model — same hyperparameters as week4/train_ppo.py so the
# only thing that changes relative to the original PPO run is num_ues
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    gamma=0.99,
    verbose=1
)

# Train for 50,000 steps (same budget as DQN 30-UE run, for a fair comparison)
model.learn(
    total_timesteps=50000
)

# Save model
model.save(
    "models/ppo_30ue"
)

print("\nTraining complete!")
print("Model saved as: models/ppo_30ue.zip")
