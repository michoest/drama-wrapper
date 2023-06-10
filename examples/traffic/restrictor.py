import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gymnasium as gym
from gymnasium.spaces import Space
from src.restrictors import Restrictor, RestrictorActionSpace


class QNetwork(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, action_space.n),
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class TrafficRestrictor(Restrictor):
    def __init__(
        self,
        observation_space,
        action_space,
        start_e,
        end_e,
        exploration_fraction,
        total_timesteps,
        cuda,
        learning_rate,
    ) -> None:
        super().__init__(observation_space, action_space)

        self.start_e, self.end_e = start_e, end_e
        self.exploration_fraction = exploration_fraction
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.cuda else "cpu"
        )

        self.q_network = QNetwork(observation_space, action_space).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.target_network = QNetwork(observation_space, action_space).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.global_step = 0

    def act(self, observation: Space) -> RestrictorActionSpace:
        epsilon = linear_schedule(
            self.start_e,
            self.end_e,
            self.exploration_fraction * self.total_timesteps,
            self.global_step,
        )
        if random.random() < epsilon:
            action = self.action_space.sample()
        else:
            q_values = self.q_network(torch.Tensor(observation).to(self.device))
            action = torch.argmax(q_values, dim=1).cpu().numpy()

        self.global_step += 1

        return action
