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

from envs.scheduling_env import SchedulingEnv

# Round Robin, 30 UEs — Table I's RR row (1545.13 ± 169.00 / 3101.76 ± 787.92
# / 0.7623 ± 0.0477) does not match any file in the repo, and the only RR
# scripts that exist either default to num_ues=5 (evaluate_rr.py) or print
# a 30-UE result without ever saving it to a CSV (week5/evaluate_rr_scenarios.py).
# This script closes that gap the same way the DQN/PF/PPO 30-UE scripts do.
env = SchedulingEnv(num_ues=30)

num_episodes = 100

throughputs = []
latencies = []
fairnesses = []
episode_results = []

for ep in range(num_episodes):

    # Same seeding scheme as the other three 30-UE FIXED scripts, so this
    # can be paired against DQN/PF/PPO in a Wilcoxon test if needed.
    np.random.seed(ep)

    obs, _ = env.reset()
    done = False
    episode_throughput = 0
    episode_latency = 0
    allocations = np.zeros(env.num_ues)
    current_ue = 0

    while not done:

        action = current_ue
        current_ue = (current_ue + 1) % env.num_ues

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
df.to_csv("results/rr_30ue_evaluation_results.csv", index=False)

# Reported with ddof=1 (pandas default / sample std) to match the convention
# already used to produce Table I's DQN and PF rows via
# statistical_summary_30ue_FIXED.py — NOT np.std's ddof=0 default, which is
# what the original evaluate_rr.py / evaluate_ppo.py scripts used. Mixing
# the two within one table is the kind of thing that shouldn't ship twice.
print("\nRound Robin (30 UE) Evaluation Results")
print("-" * 40)
print(f"Throughput: {df['Throughput'].mean():.2f} +/- {df['Throughput'].std():.2f}")
print(f"Latency:    {df['Latency'].mean():.2f} +/- {df['Latency'].std():.2f}")
print(f"Fairness:   {df['Fairness'].mean():.4f} +/- {df['Fairness'].std():.4f}")
print("\nSaved to: results/rr_30ue_evaluation_results.csv")