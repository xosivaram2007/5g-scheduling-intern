import pandas as pd

# Read evaluation results
df = pd.read_csv("results/dqn_evaluation_results.csv")

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
    "results/statistical_summary.csv",
    index=False
)

print("\nStatistical Summary")
print("-" * 30)
print(summary)