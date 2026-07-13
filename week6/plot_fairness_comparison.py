import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("results/final_results_table_30ue.csv")

# Keep only the main algorithms
methods = [
    "Round Robin",
    "Proportional Fair",
    "PPO",
    "DQN (30 UE)"
]

df = df[df["Method"].isin(methods)]

# Plot
plt.figure(figsize=(7,5), dpi=300)

plt.bar(
    df["Method"],
    df["Fairness"]
)

plt.xlabel("Scheduling Method")
plt.ylabel("Jain's Fairness Index")
plt.title("Jain's Fairness Comparison")

# Add value labels
for i, value in enumerate(df["Fairness"]):
    plt.text(
        i,
        value + 0.01,
        f"{value:.3f}",
        ha="center",
        fontsize=9
    )

plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig(
    "results/fairness_comparison.png",
    dpi=300
)

plt.show()