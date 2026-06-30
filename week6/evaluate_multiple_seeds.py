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

from envs.scheduling_env_simple_reward import SchedulingEnv

# Create environment
env = SchedulingEnv(
    num_ues=15
)

# Load trained model
model = DQN.load(
    "models/dqn_simple_reward"
)

num_episodes = 100

throughputs = []
latencies = []
fairnesses = []

episode_results = []

for ep in range(num_episodes):

    obs, _ = env.reset()

    done = False

    episode_throughput = 0
    episode_latency = 0

    allocations = np.zeros(
        env.num_ues
    )

    while not done:

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, terminated, truncated, info = env.step(
            action
        )

        done = (
            terminated
            or truncated
        )

        episode_throughput += info["throughput"]
        episode_latency += info["latency"]

        allocations += info["allocations"]

    fairness = (
        np.sum(allocations) ** 2
        /
        (
            len(allocations)
            * np.sum(allocations ** 2)
            + 1e-8
        )
    )

    throughputs.append(
        episode_throughput
    )

    latencies.append(
        episode_latency
    )

    fairnesses.append(
        fairness
    )

    episode_results.append({
        "Episode": ep + 1,
        "Throughput": episode_throughput,
        "Latency": episode_latency,
        "Fairness": fairness
    })

# Save episode-wise results
df = pd.DataFrame(episode_results)

df.to_csv(
    "results/dqn_evaluation_results.csv",
    index=False
)

print("\nSimplified Reward DQN Results")
print("-" * 35)

print(
    f"Throughput: "
    f"{np.mean(throughputs):.2f} "
    f"+/- {np.std(throughputs):.2f}"
)

print(
    f"Latency: "
    f"{np.mean(latencies):.2f} "
    f"+/- {np.std(latencies):.2f}"
)

print(
    f"Fairness: "
    f"{np.mean(fairnesses):.4f} "
    f"+/- {np.std(fairnesses):.4f}"
)

print("\nEvaluation results saved to:")
print("results/dqn_evaluation_results.csv")