import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/week5_ablation_study.csv")

plt.figure(figsize=(8,5), dpi=300)

plt.bar(
    df["Variant"],
    df["Throughput Mean"],
    yerr=df["Throughput Std"],
    capsize=5
)

plt.xlabel("DQN Variant")
plt.ylabel("Mean Throughput")
plt.title("Ablation Study of DQN Components")

plt.xticks(rotation=15)

for i, value in enumerate(df["Throughput Mean"]):
    plt.text(i, value + 250, f"{value:.0f}", ha="center", fontsize=9)

plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("results/ablation_bar.png", dpi=300, bbox_inches="tight")
plt.show()