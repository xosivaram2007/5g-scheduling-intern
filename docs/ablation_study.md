# 🧪 Ablation Study

## Overview

To evaluate the contribution of key Deep Q-Network (DQN) components, an ablation study was performed by systematically removing or modifying individual architectural elements while keeping the remaining training setup unchanged. The objective was to quantify the impact of each component on scheduling performance in the proposed 5G resource allocation framework.

The following DQN variants were evaluated:

- 🔄 **DQN without Replay Buffer**
- 🎯 **DQN without Target Network**
- 📝 **DQN with Simplified Reward Function**

All experiments were conducted using the same **30 User Equipment (UE)** scheduling environment to ensure a fair comparison.

---

# 📊 Performance Comparison

| Model | Throughput | Latency (ms) | Fairness |
|----------------------|-----------:|-----------:|---------:|
| **DQN (Full Model)** | **16820.00** | **59630.00** | **0.0333** |
| DQN No Replay Buffer | 16310.00 | 41250.00 | 0.0333 |
| DQN No Target Network | 17670.00 | 47640.00 | 0.0333 |
| DQN Simplified Reward | **18050.00** | 63130.00 | 0.0333 |

*(Values taken from `results/final_results_table_30ue.csv`, all 30-UE runs.)*

---

# 🔄 Replay Buffer Analysis

## Comparison

| Configuration | Throughput | Latency (ms) |
|---------------|-----------:|-----------:|
| Full DQN | **16820.00** | 59630.00 |
| No Replay Buffer | 16310.00 | **41250.00** |

## Impact

```text
Throughput: (16310 - 16820) / 16820 × 100 = -3.03%
Latency:    (41250 - 59630) / 59630 × 100 = -30.82%
```

## Findings

Removing the replay buffer reduced throughput by approximately **3.03%**, consistent with the expected role of experience replay in stabilizing learning by decorrelating consecutive samples. Notably, it also cut latency by **30.82%** — the largest latency reduction of any variant tested. This is a real trade-off, not a strict win/loss: the replay buffer buys a small throughput gain at a substantial latency cost, which is worth discussing explicitly rather than presenting replay buffer removal as purely harmful.

---

# 🎯 Target Network Analysis

## Comparison

| Configuration | Throughput | Latency (ms) |
|---------------|-----------:|-----------:|
| Full DQN | 16820.00 | 59630.00 |
| No Target Network | **17670.00** | **47640.00** |

## Impact

```text
Throughput: (17670 - 16820) / 16820 × 100 = +5.05%
Latency:    (47640 - 59630) / 59630 × 100 = -20.11%
```

## Findings

Contrary to the standard expectation that a target network stabilizes Q-value estimation and improves performance, removing it **improved both throughput (+5.05%) and latency (-20.11%)** in this environment. This is a counter-intuitive result and should be flagged as such rather than smoothed over — it mirrors the Simplified Reward finding your mentor already called the strongest contribution in the paper. A plausible explanation worth investigating: in this scheduling environment the target network's slower-moving updates may be introducing lag that hurts adaptation to fast-changing UE conditions more than it helps stability. This should be framed as an open finding, not asserted as fact, unless you can support it with additional evidence (e.g., training curve comparisons).

---

# 📝 Reward Function Analysis

## Comparison

| Configuration | Throughput | Latency (ms) |
|---------------|-----------:|-----------:|
| Full DQN | 16820.00 | 59630.00 |
| Simplified Reward | **18050.00** | 63130.00 |

## Impact

```text
Throughput: (18050 - 16820) / 16820 × 100 = +7.31%
Latency:    (63130 - 59630) / 59630 × 100 = +5.87%
```

## Findings

The simplified reward function produced the **largest throughput improvement of any variant tested (+7.31%)**, at the cost of a modest 5.87% increase in latency. Combined with the fairness result already highlighted in the paper (Simplified Reward achieving 0.2000 fairness vs. 0.0333 for the full reward), this reinforces that the full four-term reward function may be over-constraining the policy relative to a simpler formulation.

---

# 📌 Key Takeaways

| Component removed | Throughput vs. Full DQN | Latency vs. Full DQN |
|---|---:|---:|
| Replay Buffer | −3.03% | −30.82% (better) |
| Target Network | +5.05% (better) | −20.11% (better) |
| Reward Shaping (Simplified) | +7.31% (better) | +5.87% (worse) |

---

# Conclusions

✅ The Replay Buffer contributes a small throughput gain, but removing it substantially reduces latency — a genuine trade-off rather than a clear net positive.

⚠️ Removing the Target Network **improved** both throughput and latency in this environment — the opposite of the standard DQN expectation. This is a notable, counter-intuitive finding worth discussing rather than a confirmation that the target network is "critical."

✅ Simplifying the reward function gave the largest throughput gain of any ablation, at a modest latency cost — consistent with the fairness advantage already reported for this variant elsewhere in the paper.

✅ No single component change altered fairness (all variants converge to the 1/N floor of 0.0333), so the differentiating metrics across variants are throughput and latency, not fairness.

---

# Future Work

Although the proposed scheduler demonstrates strong throughput performance, several improvements can further enhance its applicability to practical 5G and beyond networks:

- Dynamic packet arrival models
- Variable traffic load scenarios
- Adaptive QoS-aware scheduling
- Fairness-aware reward engineering
- Multi-resource block allocation
- Multi-objective reinforcement learning
- Evaluation under realistic wireless channel conditions
- Investigating *why* target-network removal improved performance here (training curve / Q-value divergence analysis), since it runs counter to standard DQN theory

These extensions will enable a more comprehensive assessment of reinforcement learning-based scheduling for next-generation cellular networks.