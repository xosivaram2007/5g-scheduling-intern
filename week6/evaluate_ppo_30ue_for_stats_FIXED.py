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
from stable_baselines3 import PPO

from envs.scheduling_env import SchedulingEnv

# PPO model, 30 UEs — this is the run Table I currently claims exists but
# doesn't: there is no models/ppo_30ue.zip and no results/ppo_30ue_evaluation_results.csv
# anywhere in the repo prior to this script. Run week5/train_ppo_30.py first.
env = SchedulingEnv(num_ues=30)

model = PPO.load("models/ppo_30ue")

num_episodes = 100

throughputs = []
latencies = []
fairnesses = []
episode_results = []

for ep in range(num_episodes):

    # Same seeding scheme as evaluate_dqn_full_30ue_for_stats_FIXED.py and
    # evaluate_pf_30ue_for_stats_FIXED.py — episode `ep` here sees the
    # identical UE conditions as episode `ep` in those files, so pairing
    # this against either of them in a Wilcoxon test is valid.
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
df.to_csv("results/ppo_30ue_evaluation_results.csv", index=False)

print("\nPPO (30 UE) Evaluation Results")
print("-" * 40)
print(f"Throughput: {np.mean(throughputs):.2f} +/- {np.std(throughputs):.2f}")
print(f"Latency:    {np.mean(latencies):.2f} +/- {np.std(latencies):.2f}")
print(f"Fairness:   {np.mean(fairnesses):.4f} +/- {np.std(fairnesses):.4f}")
print("\nSaved to: results/ppo_30ue_evaluation_results.csv")
