import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SchedulingEnv(gym.Env):

    def __init__(
        self,
        num_ues=5,
        max_buffer=100,
        max_steps=1000
    ):
        super().__init__()

        self.num_ues = num_ues
        self.max_buffer = max_buffer
        self.max_steps = max_steps

        # Action: choose which UE to schedule
        self.action_space = spaces.Discrete(self.num_ues)

        # Observation: [buffer_size, SINR, QoS] for each UE
        self.observation_space = spaces.Box(
            low=0,
            high=max_buffer,
            shape=(self.num_ues, 3),
            dtype=np.float32
        )

        self.current_step = 0
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0

        self.state = np.zeros(
            (self.num_ues, 3),
            dtype=np.float32
        )

        for ue in range(self.num_ues):

            buffer_size = np.random.randint(
                1,
                self.max_buffer + 1
            )

            sinr = np.random.randint(
                5,
                31
            )

            qos = np.random.randint(
                0,
                3
            )

            self.state[ue] = [
                buffer_size,
                sinr,
                qos
            ]

        return self.state, {}

    def step(self, action):

        # Extract UE information
        buffer_size = self.state[action][0]
        sinr = self.state[action][1]
        qos = self.state[action][2]

        # Compute throughput
        throughput = min(buffer_size, sinr)

        # Reward function
        reward = throughput + (qos * 2)

        # Update buffer
        self.state[action][0] = max(
            0,
            buffer_size - throughput
        )

        # Approximate latency
        latency = max(
            0,
            buffer_size - throughput
        )

        # Track allocations for fairness
        allocations = np.zeros(
            self.num_ues,
            dtype=np.float32
        )

        allocations[action] = throughput

        # Info dictionary for evaluation
        info = {
            "throughput": float(throughput),
            "latency": float(latency),
            "allocations": allocations
        }

        self.current_step += 1

        terminated = False

        truncated = (
            self.current_step >= self.max_steps
        )

        return (
            self.state,
            float(reward),
            terminated,
            truncated,
            info
        )
