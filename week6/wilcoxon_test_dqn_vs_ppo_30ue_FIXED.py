import pandas as pd
from scipy.stats import wilcoxon

# Load evaluation results — both generated with num_ues=30 and identical
# per-episode seeding (see evaluate_dqn_full_30ue_for_stats_FIXED.py and
# evaluate_ppo_30ue_for_stats_FIXED.py), so episode i in one file corresponds
# to the same UE conditions as episode i in the other. That correspondence
# is required for a paired Wilcoxon signed-rank test to be valid.
#
# NOTE: this is the DQN-vs-PPO test the paper's Table I / Section V-A
# currently reports (throughput p = 0.518, latency p < 0.001) — prior to
# this script, no such comparison existed anywhere in the repo, and no
# results/ppo_30ue_evaluation_results.csv existed either. Run
# week5/train_ppo_30.py then week6/evaluate_ppo_30ue_for_stats_FIXED.py
# before running this.
dqn = pd.read_csv("results/dqn_full_30ue_evaluation_results.csv")
ppo = pd.read_csv("results/ppo_30ue_evaluation_results.csv")

metrics = ["Throughput", "Latency", "Fairness"]

print("\nWilcoxon Signed-Rank Test - Full DQN (30 UE) vs PPO (30 UE)")
print("-" * 60)

for metric in metrics:

    dqn_vals = dqn[metric]
    ppo_vals = ppo[metric]

    # Fairness will be a constant (1/30) in every episode for both methods
    # if the collapse is real — wilcoxon() errors out on all-zero differences
    # rather than silently returning p=1, so guard for that case explicitly
    # instead of letting the script crash.
    if (dqn_vals - ppo_vals).abs().sum() == 0:
        print(f"\nMetric: {metric}")
        print("Result    : Degenerate — identical in every paired episode, "
              "no variance to test")
        continue

    statistic, p_value = wilcoxon(
        dqn_vals,
        ppo_vals
    )

    print(f"\nMetric: {metric}")
    print(f"Statistic : {statistic:.4f}")
    print(f"P-value   : {p_value:.6f}")

    if p_value < 0.05:
        print("Result    : Statistically Significant (YES)")
    else:
        print("Result    : Not Statistically Significant (NO)")
