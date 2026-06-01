# 5G Resource Scheduling Environment Specification

## Objective

The objective of the scheduler is to allocate available Resource Blocks (RBs) to User Equipments (UEs) in order to maximize network throughput, minimize latency, and maintain fairness among users.

---

## Environment Description

The environment represents a simplified 5G cellular network consisting of multiple users competing for limited radio resources.

At every time slot, the scheduler observes the network state and decides how resources should be allocated.

---

## State Space

The state contains information about all active UEs.

For each UE, the following parameters are considered:

* Buffer Size (amount of pending data)
* SINR (Signal-to-Interference-plus-Noise Ratio)
* QoS Class (priority level)

Example state:

UE0 = [10, 15, 1]

UE1 = [5, 20, 2]

UE2 = [8, 12, 0]

UE3 = [4, 18, 1]

UE4 = [7, 10, 2]

Where:

* Buffer Size = packets waiting for transmission
* SINR = channel quality indicator
* QoS Class = service priority level

---

## Action Space

The action represents the scheduling decision.

The scheduler selects a UE to receive a resource block.

Example:

Action = 2

Meaning:

Allocate the current resource block to UE2.

---

## Reward Function

The reward encourages high throughput and low latency.

Reward = Throughput − Latency Penalty

Example:

Reward = 100 − 20 = 80

Higher rewards indicate better scheduling decisions.

---

## Episode Termination

An episode ends after 1000 time slots.

---

## Performance Metrics (KPIs)

### Throughput

Measures the amount of successfully transmitted data.

Unit: Mbps

### Latency

Measures packet transmission delay.

Unit: milliseconds (ms)

### Jain's Fairness Index

Measures fairness of resource allocation among users.

Formula:

J = (Σxi)² / (n × Σxi²)

Range:

* 0 = Unfair allocation
* 1 = Perfectly fair allocation

---

## Agent

The RL agent acts as the scheduler.

Its goal is to learn an optimal scheduling policy that maximizes long-term rewards.

---

## Future Implementation

The environment will be implemented using Gymnasium and will provide:

* reset()
* step()
* observation_space
* action_space

The environment will later be used for training Deep Reinforcement Learning algorithms such as DQN and PPO.
