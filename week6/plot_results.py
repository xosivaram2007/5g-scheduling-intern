import pandas as pd
import matplotlib.pyplot as plt

# Load evaluation results
dqn = pd.read_csv("results/dqn_full_30ue_evaluation_results.csv")
pf = pd.read_csv("results/pf_30ue_evaluation_results.csv")


# -------------------------------
# Bar Chart (Mean Comparison)
# -------------------------------

metrics = ["Throughput", "Latency", "Fairness"]

dqn_means = [
    dqn["Throughput"].mean(),
    dqn["Latency"].mean(),
    dqn["Fairness"].mean()
]

pf_means = [
    pf["Throughput"].mean(),
    pf["Latency"].mean(),
    pf["Fairness"].mean()
]

x = range(len(metrics))
width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(
    [i - width/2 for i in x],
    dqn_means,
    width,
    label="DQN"
)

plt.bar(
    [i + width/2 for i in x],
    pf_means,
    width,
    label="PF"
)

plt.xticks(x, metrics)
plt.ylabel("Average Value")
plt.title("DQN vs PF Performance Comparison")
plt.legend()

plt.tight_layout()
plt.savefig("results/algorithm_comparison.png")
plt.close()

# -------------------------------
# Throughput Boxplot
# -------------------------------

plt.figure(figsize=(6, 5))

plt.boxplot(
    [dqn["Throughput"], pf["Throughput"]],
    labels=["DQN", "PF"]
)

plt.title("Throughput Comparison")
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

plt.title("Latency Comparison")
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

plt.title("Fairness Comparison")
plt.ylabel("Fairness")

plt.tight_layout()
plt.savefig("results/fairness_boxplot.png")
plt.close()

print("Plots generated successfully!")
print("Saved in results/ folder.")