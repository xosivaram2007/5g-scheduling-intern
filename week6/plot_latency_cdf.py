import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load results
dqn = pd.read_csv("results/dqn_full_30ue_evaluation_results.csv")
pf = pd.read_csv("results/pf_30ue_evaluation_results.csv")

# Sort latency values
dqn_latency = np.sort(dqn["Latency"])
pf_latency = np.sort(pf["Latency"])

# Compute cumulative probability
dqn_cdf = np.arange(1, len(dqn_latency) + 1) / len(dqn_latency)
pf_cdf = np.arange(1, len(pf_latency) + 1) / len(pf_latency)

# Plot
plt.figure(figsize=(7,5), dpi=300)

plt.plot(dqn_latency, dqn_cdf, linewidth=2, label="DQN")
plt.plot(pf_latency, pf_cdf, linewidth=2, label="PF")

plt.xlabel("Latency")
plt.ylabel("Cumulative Probability")
plt.title("Latency Cumulative Distribution Function")
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "results/latency_cdf.png",
    dpi=300
)

plt.show()