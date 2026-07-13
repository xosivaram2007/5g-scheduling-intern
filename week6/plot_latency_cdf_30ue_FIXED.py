import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load results — matched 30-UE, seeded/paired files (see the two
# evaluate_*_30ue_for_stats_FIXED.py scripts)
dqn = pd.read_csv("results/dqn_full_30ue_evaluation_results.csv")
pf = pd.read_csv("results/pf_30ue_evaluation_results.csv")

# Sort latency values
dqn_latency = np.sort(dqn["Latency"])
pf_latency = np.sort(pf["Latency"])

# Compute cumulative probability
dqn_cdf = np.arange(1, len(dqn_latency) + 1) / len(dqn_latency)
pf_cdf = np.arange(1, len(pf_latency) + 1) / len(pf_latency)

# Plot
plt.figure(figsize=(7, 5), dpi=300)

plt.plot(dqn_latency, dqn_cdf, linewidth=2, label="DQN (Full, 30 UE)")
plt.plot(pf_latency, pf_cdf, linewidth=2, label="PF (30 UE)")

plt.xlabel("Latency")
plt.ylabel("Cumulative Probability")
plt.title("Latency Cumulative Distribution Function (30 UE)")
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "results/latency_cdf_30ue.png",
    dpi=300
)

plt.show()