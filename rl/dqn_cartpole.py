import gymnasium as gym
from stable_baselines3 import DQN

env = gym.make("CartPole-v1")

model = DQN(
    "MlpPolicy",
    env,
    verbose=1
)

print("Training Started...")

model.learn(total_timesteps=10000)

model.save("dqn_cartpole")

print("Training Complete!")
