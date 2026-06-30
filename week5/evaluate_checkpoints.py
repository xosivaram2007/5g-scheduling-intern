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
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from envs.scheduling_env import SchedulingEnv

# Create results folder
os.makedirs(
    "results",
    exist_ok=True
)

# Environment
env = SchedulingEnv(
    num_ues=30
)

# Checkpoint steps
steps_list = [
    5000,
    10000,
    15000,
    20000,
    25000,
    30000,
    35000,
    40000,
    45000,
    50000
]

results = []

for steps in steps_list:

    model_path = (
        f"models/checkpoints/"
        f"dqn_30ue_{steps}_steps.zip"
    )

    print(f"\nEvaluating {steps} steps...")

    model = DQN.load(model_path)

    throughputs = []
    latencies = []
    fairnesses = []

    num_episodes = 20

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

            done = terminated or truncated

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

    results.append(
        [
            steps,
            np.mean(throughputs),
            np.mean(latencies),
            np.mean(fairnesses)
        ]
    )

# Save CSV
df = pd.DataFrame(
    results,
    columns=[
        "steps",
        "throughput",
        "latency",
        "fairness"
    ]
)

csv_path = "results/checkpoint_results.csv"

df.to_csv(
    csv_path,
    index=False
)

print(f"\nSaved: {csv_path}")

# Throughput plot
plt.figure(figsize=(8,5))
plt.plot(
    df["steps"],
    df["throughput"],
    marker="o"
)
plt.xlabel("Training Steps")
plt.ylabel("Throughput")
plt.title("Throughput vs Training Steps")
plt.grid(True)
plt.savefig(
    "results/throughput_convergence.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# Latency plot
plt.figure(figsize=(8,5))
plt.plot(
    df["steps"],
    df["latency"],
    marker="o"
)
plt.xlabel("Training Steps")
plt.ylabel("Latency")
plt.title("Latency vs Training Steps")
plt.grid(True)
plt.savefig(
    "results/latency_convergence.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# Fairness plot
plt.figure(figsize=(8,5))
plt.plot(
    df["steps"],
    df["fairness"],
    marker="o"
)
plt.xlabel("Training Steps")
plt.ylabel("Fairness")
plt.title("Fairness vs Training Steps")
plt.grid(True)
plt.savefig(
    "results/fairness_convergence.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("\nDone!")
print("Generated:")
print(" - results/checkpoint_results.csv")
print(" - results/throughput_convergence.png")
print(" - results/latency_convergence.png")
print(" - results/fairness_convergence.png")