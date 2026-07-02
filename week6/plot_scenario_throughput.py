import pandas as pd
import matplotlib.pyplot as plt

# Load scenario results
df = pd.read_csv("results/week5_scenarios.csv")

# Plot
plt.figure(figsize=(7, 5), dpi=300)

plt.bar(
    df["Scenario"],
    df["Mean Throughput"]
)

plt.xlabel("Network Scenario")
plt.ylabel("Mean Throughput")
plt.title("Throughput under Different Network Load Scenarios")

# Add value labels
for i, value in enumerate(df["Mean Throughput"]):
    plt.text(
        i,
        value + 150,
        f"{value:.0f}",
        ha="center",
        fontsize=9
    )

plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig(
    "results/scenario_throughput.png",
    dpi=300
)

plt.show()