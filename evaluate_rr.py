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

    current_ue = 0

    while not done:

        # Round Robin action
        action = current_ue
        current_ue = (current_ue + 1) % env.num_ues

        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        episode_throughput += info["throughput"]
        episode_latency += info["latency"]

        allocations += info["allocations"]

    # Jain's Fairness Index
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

print("\nRR Evaluation Results")
print("-" * 30)

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
