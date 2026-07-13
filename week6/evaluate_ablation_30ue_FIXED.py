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
from stable_baselines3 import DQN
from envs.scheduling_env import SchedulingEnv

# One entry per model to re-evaluate, all on the SAME 30-UE environment
# used for the Full DQN model, so this is now a fair, single-variable ablation.
MODELS_TO_EVALUATE = [
    ("Full DQN",         "models/dqn_30ue"),
    ("No Replay Buffer", "models/dqn_no_replay_30ue"),
    ("No Target Network","models/dqn_no_target_30ue"),
    ("Simplified Reward","models/dqn_simple_reward_30ue"),
]

num_episodes = 100

for label, model_path in MODELS_TO_EVALUATE:

    env = SchedulingEnv(num_ues=30)

    model = DQN.load(model_path)

    throughputs = []
    latencies = []
    fairnesses = []

    for ep in range(num_episodes):

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

    print(f"\n{label} (30 UE) Evaluation Results")
    print("-" * 40)
    print(f"Throughput: {np.mean(throughputs):.2f} +/- {np.std(throughputs):.2f}")
    print(f"Latency:    {np.mean(latencies):.2f} +/- {np.std(latencies):.2f}")
    print(f"Fairness:   {np.mean(fairnesses):.4f} +/- {np.std(fairnesses):.4f}")