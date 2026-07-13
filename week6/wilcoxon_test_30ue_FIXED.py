import pandas as pd
from scipy.stats import wilcoxon

# Load evaluation results — both generated with num_ues=30 and identical
# per-episode seeding (see evaluate_dqn_full_30ue_for_stats_FIXED.py and
# evaluate_pf_30ue_for_stats_FIXED.py), so episode i in one file corresponds
# to the same UE conditions as episode i in the other. That correspondence
# is required for a paired Wilcoxon signed-rank test to be valid.
dqn = pd.read_csv("results/dqn_full_30ue_evaluation_results.csv")
pf = pd.read_csv("results/pf_30ue_evaluation_results.csv")

metrics = ["Throughput", "Latency", "Fairness"]

print("\nWilcoxon Signed-Rank Test - Full DQN (30 UE) vs PF (30 UE)")
print("-" * 60)

for metric in metrics:

    statistic, p_value = wilcoxon(
        dqn[metric],
        pf[metric]
    )

    print(f"\nMetric: {metric}")
    print(f"Statistic : {statistic:.4f}")
    print(f"P-value   : {p_value:.6f}")

    if p_value < 0.05:
        print("Result    : Statistically Significant (YES)")
    else:
        print("Result    : Not Statistically Significant (NO)")