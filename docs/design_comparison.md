# Design Comparison: Our Scheduler vs Research Paper

**Paper:** Deep Reinforcement Learning for Resource Allocation in 5G Communications (APSIPA 2019)

| Component | Paper Design                                         | Our Design                                 |
| --------- | ---------------------------------------------------- | ------------------------------------------ |
| State     | CSI, SINR, RRH status, user demand, previous actions | Buffer occupancy, SINR, QoS class          |
| Action    | RRH selection, power allocation, resource allocation | UE scheduling / PRB assignment             |
| Reward    | Throughput − power consumption                       | Throughput − latency penalty               |
| Algorithm | DQN (Q-learning based)                               | DQN and PPO                                |
| KPIs      | Throughput, energy efficiency, QoS                   | Throughput, latency, Jain's fairness index |

## Observations

1. Both approaches use Deep Reinforcement Learning for 5G resource management.
2. The paper focuses on energy-efficient CRAN operation, while our work focuses on scheduling fairness and latency reduction.
3. Both systems utilize channel quality information such as SINR.
4. Our environment is simpler and more suitable for Gymnasium-based experimentation.
5. Future work could incorporate energy-aware rewards and power control mechanisms.
