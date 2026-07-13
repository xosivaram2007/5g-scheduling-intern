import pandas as pd
import matplotlib.pyplot as plt

# Load evaluation results
dqn = pd.read_csv("results/dqn_full_30ue_evaluation_results.csv")
pf = pd.read_csv("results/pf_30ue_evaluation_results.csv")

# Load the aggregate 4-method summary table (RR, PF, PPO, DQN) — used for
# the Figure 1 "aggregate throughput comparison" chart, since only DQN and
# PF have per-episode result files. RR and PPO are only available as
# summary means, so this chart uses means for all four methods for a
# consistent, fair comparison.
summary = pd.read_csv("results/final_results_table_30ue.csv")


# -------------------------------
# Figure 1: Aggregate Throughput Comparison — RR, PF, PPO, DQN
# -------------------------------
# This chart previously plotted only DQN vs PF on three metrics
# (Throughput, Latency, Fairness), while the paper caption described an
# "aggregate throughput comparison across RR, PF, PPO, and DQN." That
# mismatch is fixed here: this chart now plots throughput only, across
# all four methods named in the caption.

methods = ["Round Robin (RR)", "Proportional Fair (PF)", "PPO", "DQN (Ours)"]
throughput_means = [
    summary.loc[summary["Method"] == m, "Throughput Mean"].values[0]
    for m in methods
]

plt.figure(figsize=(8, 5))
bars = plt.bar(methods, throughput_means, color=["#8c8c8c", "#4a90d9", "#f5a623", "#2e7d32"])

for bar, value in zip(bars, throughput_means):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + max(throughput_means) * 0.01,
        f"{value:.2f}",
        ha="center",
        fontsize=9
    )

plt.ylabel("Mean Throughput")
plt.title("Aggregate Throughput Comparison: RR, PF, PPO, and DQN")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("results/algorithm_comparison.png", dpi=300)
plt.close()


# -------------------------------
# Throughput Boxplot (DQN vs PF — per-episode data available)
# -------------------------------

plt.figure(figsize=(6, 5))

plt.boxplot(
    [dqn["Throughput"], pf["Throughput"]],
    labels=["DQN", "PF"]
)

plt.title("Throughput Comparison (DQN vs PF, per-episode)")
plt.ylabel("Throughput")

plt.tight_layout()
plt.savefig("results/throughput_boxplot.png")
plt.close()

# -------------------------------
# Latency Boxplot
# -------------------------------

plt.figure(figsize=(6, 5))

plt.boxplot(
    [dqn["Latency"], pf["Latency"]],
    labels=["DQN", "PF"]
)

plt.title("Latency Comparison (DQN vs PF, per-episode)")
plt.ylabel("Latency")

plt.tight_layout()
plt.savefig("results/latency_boxplot.png")
plt.close()

# -------------------------------
# Fairness Boxplot
# -------------------------------

plt.figure(figsize=(6, 5))

plt.boxplot(
    [dqn["Fairness"], pf["Fairness"]],
    labels=["DQN", "PF"]
)

plt.title("Fairness Comparison (DQN vs PF, per-episode)")
plt.ylabel("Fairness")

plt.tight_layout()
plt.savefig("results/fairness_boxplot.png")
plt.close()

print("Plots generated successfully!")
print("Saved in results/ folder.")