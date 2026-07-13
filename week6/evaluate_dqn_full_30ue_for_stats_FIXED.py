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

import numpy as np
import pandas as pd
from stable_baselines3 import DQN

from envs.scheduling_env import SchedulingEnv

# Full DQN model, 30 UEs — matches the headline "Full DQN (30 UE)" configuration
env = SchedulingEnv(num_ues=30)

model = DQN.load("models/dqn_30ue")

num_episodes = 100

throughputs = []
latencies = []
fairnesses = []
episode_results = []

for ep in range(num_episodes):

    # Seed BEFORE reset so episode `ep` generates the same UE conditions
    # (buffer/SINR/QoS) here as in the PF script below — this is what makes
    # a paired Wilcoxon test between the two files valid.
    np.random.seed(ep)

    obs, _ = env.reset()
    done = False
    episode_throughput = 0
    episode_latency = 0
    allocations = np.zeros(env.num_ues)

    while not done:

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        episode_throughput += info["throughput"]
        episode_latency += info["latency"]
        allocations += info["allocations"]

    fairness = (
        np.sum(allocations) ** 2
        / (len(allocations) * np.sum(allocations ** 2) + 1e-8)
    )

    throughputs.append(episode_throughput)
    latencies.append(episode_latency)
    fairnesses.append(fairness)

    episode_results.append({
        "Episode": ep + 1,
        "Throughput": episode_throughput,
        "Latency": episode_latency,
        "Fairness": fairness
    })

df = pd.DataFrame(episode_results)
df.to_csv("results/dqn_full_30ue_evaluation_results.csv", index=False)

print("\nFull DQN (30 UE) Evaluation Results")
print("-" * 40)
print(f"Throughput: {np.mean(throughputs):.2f} +/- {np.std(throughputs):.2f}")
print(f"Latency:    {np.mean(latencies):.2f} +/- {np.std(latencies):.2f}")
print(f"Fairness:   {np.mean(fairnesses):.4f} +/- {np.std(fairnesses):.4f}")
print("\nSaved to: results/dqn_full_30ue_evaluation_results.csv")