import gymnasium as gym
from stable_baselines3 import DQN

env = gym.make("CartPole-v1", render_mode="human")

model = DQN.load("dqn_cartpole")

obs, info = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs)

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
