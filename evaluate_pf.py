import numpy as np
from envs.scheduling_env import SchedulingEnv

env = SchedulingEnv()

num_episodes = 100

throughputs = []
latencies = []
fairnesses = []

for ep in range(num_episodes):

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

        action = np.argmax(pf_metric)

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
        /
        (
            len(allocations)
            * np.sum(allocations ** 2)
            + 1e-8
        )
    )

    throughputs.append(episode_throughput)
    latencies.append(episode_latency)
    fairnesses.append(fairness)

print("\nPF Evaluation Results")
print("-" * 30)

print(
    f"Throughput: {np.mean(throughputs):.2f} +/- {np.std(throughputs):.2f}"
)

print(
    f"Latency: {np.mean(latencies):.2f} +/- {np.std(latencies):.2f}"
)

print(
    f"Fairness: {np.mean(fairnesses):.4f} +/- {np.std(fairnesses):.4f}"
)
