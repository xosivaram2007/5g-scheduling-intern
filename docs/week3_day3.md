# Week 3 Day 3 – Stable-Baselines3 DQN on CartPole

## Objective

Learn how to use Stable-Baselines3 (SB3) to train a Deep Q-Network (DQN) agent on the CartPole environment.

## Tasks Completed

1. Verified Gymnasium installation.
2. Verified Stable-Baselines3 installation.
3. Explored the CartPole observation space and action space.
4. Created a DQN agent using SB3.
5. Trained the DQN agent for 10,000 timesteps.
6. Saved the trained model as `dqn_cartpole.zip`.

## Key Concepts Learned

### Observation Space

CartPole provides four state values:

* Cart Position
* Cart Velocity
* Pole Angle
* Pole Angular Velocity

### Action Space

The agent can perform two actions:

* 0 → Move Left
* 1 → Move Right

### Deep Q-Network (DQN)

DQN combines Q-Learning with Deep Neural Networks. Stable-Baselines3 automatically handles:

* Replay Buffer
* Target Network
* Epsilon-Greedy Exploration
* Neural Network Training

## Results

The training process completed successfully and generated a trained model file.

The episode reward showed improvement during training, confirming that the SB3 training pipeline was functioning correctly.

## Outcome

A complete reinforcement learning workflow was successfully executed using Stable-Baselines3 and Gymnasium. This serves as preparation for training a DQN agent on the custom 5G scheduling environment in Week 3 Day 4.
