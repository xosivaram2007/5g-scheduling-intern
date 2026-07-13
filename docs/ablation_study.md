# 🧪 Ablation Study

## Overview

To understand the contribution of individual Deep Q-Network (DQN) components, an ablation study was conducted by systematically removing key architectural elements and comparing performance against the full DQN implementation.

The following variants were evaluated:

- 🔄 DQN without Replay Buffer
- 🎯 DQN without Target Network
- 📝 DQN with Simplified Reward Function

All experiments were evaluated using the same 30 UE scheduling environment.

---

## 📊 Performance Comparison

| Model | Throughput | Latency | Fairness |
|---------|-----------:|---------:|---------:|
| **DQN (Full Model)** | **17520.00** | **61230.00** | **0.0333** |
| DQN No Replay | 16070.00 | 39830.00 | 0.0333 |
| DQN No Target | 14390.00 | 54400.00 | 0.0333 |
| DQN Simple Reward | 17610.00 | 63190.00 | 0.0333 |

---

# 🔄 Replay Buffer Analysis

## Comparison

| Configuration | Throughput |
|---------------|-----------:|
| Full DQN | **17520.00** |
| No Replay Buffer | 16070.00 |

## Impact

Throughput Gain:

```text
((17520 - 16070) / 16070) × 100
= 9.02%
```

## Findings

The replay buffer improves learning by allowing the agent to reuse past experiences during training, reducing correlation between consecutive samples and improving convergence.

Removing the replay buffer reduced throughput by approximately **9.02%**, demonstrating that experience replay contributes significantly to learning efficiency in the scheduling environment.

---

# 🎯 Target Network Analysis

## Comparison

| Configuration | Throughput |
|---------------|-----------:|
| Full DQN | **17520.00** |
| No Target Network | 14390.00 |

## Impact

```text
((17520 - 14390) / 14390) × 100
= 21.75%
```

## Findings

The target network stabilizes Q-value updates by providing a slowly changing target during training.

Removing the target network resulted in a throughput reduction of approximately **21.75%**, making it the most influential DQN component evaluated in this study. This highlights the importance of stable target estimation for effective reinforcement learning.

---

# 📝 Reward Function Analysis

## Comparison

| Configuration | Throughput |
|---------------|-----------:|
| Full DQN | 17520.00 |
| Simplified Reward | **17610.00** |

## Impact

```text
((17610 - 17520) / 17520) × 100
= 0.51%
```

## Findings

The simplified reward function achieved nearly identical performance to the original reward formulation, with a marginal throughput increase of approximately **0.51%**.

This indicates that the current scheduling environment is primarily driven by throughput optimization, and additional reward shaping has only a limited impact on overall performance.

---

# 📌 Key Takeaways

| Component | Throughput Contribution |
|------------|-----------------------:|
| Replay Buffer | +9.02% |
| Target Network | +21.75% |
| Reward Shaping | ~0.51% (negligible difference) |

## Conclusions

✅ Replay Buffer significantly improves learning efficiency.

✅ Target Network is the most critical component for stable and effective DQN training.

✅ Simplifying the reward function has minimal impact on throughput under the current environment.

✅ The Target Network and Replay Buffer remain the primary contributors to DQN performance in the proposed scheduler.

---

## Future Work

The current environment uses a simplified traffic model. Future enhancements may include:

- Dynamic packet arrivals
- Variable traffic loads
- Multi-resource scheduling
- Fairness-aware reward engineering
- Multi-objective reinforcement learning

These additions would enable a more realistic evaluation of intelligent scheduling strategies for next-generation 5G and beyond wireless networks.