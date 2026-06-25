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

        self.action_space = spaces.Discrete(
            self.num_ues
        )

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

        buffer_size = self.state[action][0]
        sinr = self.state[action][1]

        throughput = min(
            buffer_size,
            sinr
        )

        # Simplified reward
        reward = throughput

        self.state[action][0] = max(
            0,
            buffer_size - throughput
        )

        latency = max(
            0,
            buffer_size - throughput
        )

        allocations = np.zeros(
            self.num_ues,
            dtype=np.float32
        )

        allocations[action] = throughput

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