import pandas as pd
from scipy.stats import wilcoxon

# Load evaluation results
dqn = pd.read_csv("results/dqn_evaluation_results.csv")
pf = pd.read_csv("results/pf_evaluation_results.csv")

metrics = ["Throughput", "Latency", "Fairness"]

print("\nWilcoxon Signed-Rank Test")
print("-" * 40)

for metric in metrics:

    statistic, p_value = wilcoxon(
        dqn[metric],
        pf[metric]
    )

    print(f"\nMetric: {metric}")
    print(f"Statistic : {statistic:.4f}")
    print(f"P-value   : {p_value:.6f}")

    if p_value < 0.05:
        print("Result    : Statistically Significant ✅")
    else:
        print("Result    : Not Statistically Significant ❌")