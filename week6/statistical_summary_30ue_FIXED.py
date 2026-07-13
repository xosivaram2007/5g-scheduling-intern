import pandas as pd

# Read evaluation results — matched 30-UE Full DQN file (not the
# simple-reward/15-UE variant that the original script pointed at)
df = pd.read_csv("results/dqn_full_30ue_evaluation_results.csv")

# Calculate statistics
summary = pd.DataFrame({
    "Metric": ["Throughput", "Latency", "Fairness"],
    "Mean": [
        df["Throughput"].mean(),
        df["Latency"].mean(),
        df["Fairness"].mean()
    ],
    "Std": [
        df["Throughput"].std(),
        df["Latency"].std(),
        df["Fairness"].std()
    ]
})

# Save summary
summary.to_csv(
    "results/statistical_summary_30ue.csv",
    index=False
)

print("\nStatistical Summary — Full DQN (30 UE)")
print("-" * 40)
print(summary)