import pandas as pd
import matplotlib.pyplot as plt

# Load ablation results
df = pd.read_csv("results/week5_ablation_study.csv")

# Plot
plt.figure(figsize=(8, 5), dpi=300)

plt.bar(
    df["Configuration"],
    df["Throughput"]
)

plt.xlabel("DQN Variant")
plt.ylabel("Throughput")
plt.title("Ablation Study: Effect of DQN Components")

plt.xticks(rotation=15)

# Value labels
for i, value in enumerate(df["Throughput"]):
    plt.text(
        i,
        value + 100,
        f"{value:.0f}",
        ha="center",
        fontsize=9
    )

plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig(
    "results/ablation_bar.png",
    dpi=300
)

plt.show()