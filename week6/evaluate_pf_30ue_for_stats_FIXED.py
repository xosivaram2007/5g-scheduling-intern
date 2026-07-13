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

# Proportional Fair, 30 UEs — matches the Full DQN config above so the two
# are directly comparable.
env = SchedulingEnv(num_ues=30)

num_episodes = 100

throughputs = []
latencies = []
fairnesses = []
episode_results = []

for ep in range(num_episodes):

    # Same seeding scheme as evaluate_dqn_full_30ue_for_stats_FIXED.py —
    # episode `ep` here sees the identical UE conditions as episode `ep`
    # in the DQN file, so pairing them in a Wilcoxon test is now valid.
    np.random.seed(ep)

    obs, _ = env.reset()
    done = False
    episode_throughput = 0
    episode_latency = 0
    allocations = np.zeros(env.num_ues)
    avg_throughput = np.ones(env.num_ues)

    while not done:

        pf_metric = []
        for ue in range(env.num_ues):
            current_tp = obs[ue][1]  # SINR proxy
            metric = current_tp / avg_throughput[ue]
            pf_metric.append(metric)

        action = int(np.argmax(pf_metric))

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        episode_throughput += info["throughput"]
        episode_latency += info["latency"]
        allocations += info["allocations"]

        avg_throughput[action] = (
            0.9 * avg_throughput[action]
            + 0.1 * info["throughput"]
        )

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
df.to_csv("results/pf_30ue_evaluation_results.csv", index=False)

print("\nProportional Fair (30 UE) Evaluation Results")
print("-" * 40)
print(f"Throughput: {np.mean(throughputs):.2f} +/- {np.std(throughputs):.2f}")
print(f"Latency:    {np.mean(latencies):.2f} +/- {np.std(latencies):.2f}")
print(f"Fairness:   {np.mean(fairnesses):.4f} +/- {np.std(fairnesses):.4f}")
print("\nSaved to: results/pf_30ue_evaluation_results.csv")