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

| Model | Throughput | Latency | Fairness |
|----------------------|-----------:|-----------:|---------:|
| **DQN (Full Model)** | **17520.00** | **61230.00** | **0.0333** |
| DQN No Replay Buffer | 16070.00 | 39830.00 | 0.0333 |
| DQN No Target Network | 14390.00 | 54400.00 | 0.0333 |
| DQN Simplified Reward | **17610.00** | 63190.00 | 0.0333 |

---

# 🔄 Replay Buffer Analysis

## Comparison

| Configuration | Throughput |
|---------------|-----------:|
| Full DQN | **17520.00** |
| No Replay Buffer | 16070.00 |

## Impact

Throughput Improvement:

```text
((17520 - 16070) / 16070) × 100
= 9.02%
```

## Findings

Experience replay enables the agent to learn from a diverse collection of past transitions rather than relying solely on consecutive experiences.

Removing the replay buffer reduced throughput by approximately **9.02%**, indicating that replay memory plays an important role in improving learning efficiency and convergence stability.

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

The target network stabilizes Q-value estimation by periodically updating the target used during Bellman learning.

Removing the target network caused the largest degradation in throughput, reducing performance by approximately **21.75%**. This confirms that the target network is the most influential architectural component in the proposed DQN scheduler.

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

The simplified reward function achieved performance very close to the original reward formulation, producing a marginal throughput increase of approximately **0.51%**.

Since the difference is extremely small, the results suggest that the scheduling policy is primarily influenced by the learning architecture rather than the specific reward shaping used in this environment.

---

# 📌 Key Takeaways

| Component | Throughput Contribution |
|------------|-----------------------:|
| Replay Buffer | +9.02% |
| Target Network | +21.75% |
| Reward Function | ~0.51% (negligible difference) |

---

# Conclusions

✅ Replay Buffer significantly improves learning efficiency by enabling experience reuse.

✅ Target Network is the most critical DQN component for maintaining stable and effective learning.

✅ Simplifying the reward function produces nearly identical performance under the current simulation setup.

✅ The experimental results demonstrate that architectural components (Replay Buffer and Target Network) have a substantially greater impact on scheduler performance than reward shaping.

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

These extensions will enable a more comprehensive assessment of reinforcement learning-based scheduling for next-generation cellular networks.