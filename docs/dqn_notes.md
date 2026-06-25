# Deep Q-Network (DQN) Notes

**Week 3 Day 2 – Deep Reinforcement Learning Foundations**

---

## Paper

**Title:** Human-Level Control Through Deep Reinforcement Learning

**Authors:** Volodymyr Mnih et al.

**Year:** 2015

**Organization:** DeepMind

---

## Introduction

Traditional Q-Learning uses a Q-table to store values for every state-action pair.

For large or continuous state spaces, Q-tables become impractical because they require excessive memory and cannot generalize to unseen states.

Deep Q-Networks (DQN) address this limitation by replacing the Q-table with a neural network that approximates Q-values.

---

## Core Idea

Instead of storing:

```text
Q(State, Action)
```

DQN learns:

```text
Q(State, Action; θ)
```

where θ represents the neural network parameters.

The network predicts the expected future reward for each possible action.

---

## DQN Architecture

```text
Current State
      │
      ▼
+-------------------+
|    Q-Network      |
| (Neural Network)  |
+-------------------+
      │
      ▼
 Q-values for Actions
      │
      ▼
 Select Best Action
```

---

## Key Components

### 1. Q-Network

The Q-Network is a neural network that estimates Q-values for all available actions.

#### Input

State information:

- Buffer Size
- SINR
- QoS Class

#### Output

```text
Action 0 → Q = 10.2
Action 1 → Q = 15.8
Action 2 → Q = 8.4
Action 3 → Q = 12.1
Action 4 → Q = 17.5
```

Selected Action:

```text
argmax(Q)
```

In this example, Action 4 would be selected.

#### Advantages

- Handles large state spaces
- Eliminates the need for massive Q-tables
- Generalizes to unseen states

---

### 2. Experience Replay Buffer

Instead of learning immediately after every interaction, experiences are stored in memory.

#### Stored Experience

```text
(State, Action, Reward, Next State)
```

Example:

```text
(S1, A1, R1, S2)
(S2, A0, R2, S3)
(S3, A4, R3, S4)
```

#### Training Process

1. Store experiences in replay memory.
2. Randomly sample a mini-batch.
3. Train the Q-Network using the sampled experiences.

#### Advantages

- Breaks correlation between consecutive samples
- Improves training stability
- Increases sample efficiency

---

### 3. Target Network

Using a single network for both prediction and target generation can cause unstable learning.

DQN introduces a second network called the Target Network.

#### Structure

```text
Main Q-Network
       │
       ▼
Target Network
```

#### Update Rule

```text
Target Network ← Main Q-Network
```

The update occurs periodically rather than after every step.

#### Advantages

- Stabilizes learning
- Reduces oscillations
- Improves convergence

---

### 4. Epsilon-Greedy Policy

The epsilon-greedy strategy balances exploration and exploitation.

#### Exploration

Choose a random action.

Purpose:

- Discover new actions
- Gather more information about the environment

#### Exploitation

Choose the action with the highest predicted Q-value.

Purpose:

- Maximize reward
- Use previously learned knowledge

#### Example

```text
ε = 1.0
100% Random Actions
```

```text
ε = 0.1
10% Random Actions
90% Best Actions
```

---

## DQN Workflow

```text
1. Observe current state
2. Select action using epsilon-greedy policy
3. Execute action
4. Receive reward and next state
5. Store transition in replay buffer
6. Sample random mini-batch
7. Update Q-Network
8. Periodically update Target Network
9. Repeat
```

---

## Application to 5G Resource Scheduling

### State

- Buffer Size
- SINR
- QoS Class

### Action

Select a User Equipment (UE) for scheduling.

### Reward

```text
Reward = Throughput + QoS Benefit
```

### Objective

Maximize:

- Throughput
- Fairness
- Quality of Service (QoS)

while minimizing latency.

---

## References

1. Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015).
   "Human-Level Control Through Deep Reinforcement Learning."
   Nature, 518(7540), 529–533.

2. Gymnasium Documentation:
   https://gymnasium.farama.org/

3. Stable-Baselines3 Documentation:
   https://stable-baselines3.readthedocs.io/