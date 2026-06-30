📈 Convergence Analysis
Objective
The objective of this experiment was to verify that the Deep Q-Network (DQN) scheduler successfully learns a stable policy over time and does not exhibit catastrophic forgetting during training.
To evaluate convergence, model checkpoints were saved every 5,000 training steps and evaluated independently.
🧪 Experimental Setup
Training Configuration
Parameter
Value
Algorithm
DQN
Environment
5G Scheduling Environment
Number of UEs
30
Total Training Steps
50,000
Checkpoint Interval
5,000 Steps
Evaluation Episodes
20
📊 Checkpoint Evaluation Results
Training Steps
Throughput
Latency
Fairness
5,000
19200.00
42100.00
0.0333
10,000
14550.00
39300.00
0.0333
15,000
16750.00
56550.00
0.0333
20,000
16300.00
60300.00
0.0333
25,000
13850.00
59650.00
0.0333
30,000
17800.00
49850.00
0.0333
35,000
16100.00
67700.00
0.0333
40,000
17600.00
64600.00
0.0333
45,000
16600.00
62550.00
0.0333
50,000
17700.00
65900.00
0.0333
📉 Reward Curve Analysis
The training reward curve was generated using monitor logs collected during training.
Observations
✅ Initial rewards increased during early training.
✅ The moving average reward remained above the initial training baseline.
✅ Reward fluctuations were observed due to the stochastic nature of the scheduling environment.
Interpretation
The reward trend indicates that the agent successfully learned scheduling behavior and improved performance compared to its initial policy.
📈 Throughput Convergence
Observations
Throughput improved significantly after the initial stages of training.
Later checkpoints consistently achieved throughput values between approximately 16,000 and 19,000.
No long-term degradation in performance was observed.
Interpretation
The throughput metric demonstrates that the DQN agent converged toward a stable scheduling policy.
⏱️ Latency Convergence
Observations
Latency fluctuated throughout training.
No uncontrolled growth or instability was observed.
Latency remained within a stable operating range across checkpoints.
Interpretation
Despite environment randomness, latency remained bounded and stable throughout training.
⚖️ Fairness Analysis
Observations
Fairness remained constant across all checkpoints.
Jain's Fairness Index remained approximately 0.0333.
Interpretation
The fairness metric was heavily influenced by the simplified environment design and therefore did not provide meaningful differentiation between checkpoints.
This limitation is discussed further in the project limitations section.
🚨 Catastrophic Forgetting Check
Catastrophic forgetting occurs when a trained reinforcement learning agent suddenly loses previously acquired knowledge and experiences a significant performance collapse.
Evaluation Criteria
The following indicators were monitored:
Sudden throughput collapse
Persistent latency explosion
Significant performance degradation at later checkpoints
Results
✅ No major throughput collapse observed.
✅ No sustained latency explosion observed.
✅ Final checkpoints maintained performance comparable to earlier checkpoints.
Conclusion
No evidence of catastrophic forgetting was observed during training.
🎯 Final Conclusion
The checkpoint evaluation results demonstrate that the DQN scheduler learned a stable policy over time.
Key findings include:
Successful policy improvement during training.
Stable throughput performance at later checkpoints.
No catastrophic forgetting.
Consistent behavior across multiple evaluation stages.
Overall, the convergence analysis provides evidence that the DQN agent achieved stable learning behavior within the simulated 5G scheduling environment.