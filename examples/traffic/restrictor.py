import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gymnasium as gym
from gymnasium.spaces import Space, Discrete, Box
from drama.restrictors import Restrictor, RestrictorActionSpace, DiscreteVectorActionSpace
from drama.restrictions import DiscreteVectorRestriction
from drama.utils import flatdim

from examples.utils import ReplayBuffer


class QNetwork(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, flatdim(action_space)),
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class TrafficRestrictor(Restrictor):
    def __init__(
        self,
        number_of_edges,
        number_of_routes,
        valid_route_restrictions,
        total_timesteps,
        # route_list
    ) -> None:
        observation_space = Box(0, np.inf, (number_of_edges, ))
        action_space = DiscreteVectorActionSpace(Discrete(number_of_routes))
        self.network_action_space = Discrete(len(valid_route_restrictions))

        super().__init__(observation_space, action_space)

        self.valid_route_restrictions = valid_route_restrictions
        self.start_e, self.end_e = 1.0, 0.01
        self.exploration_fraction = 0.5
        self.total_timesteps = total_timesteps
        self.learning_rate = 2.5e-4
        self.batch_size = 128
        self.gamma = 0.99
        self.target_network_frequency = 500
        self.tau = 1
        self.training_start = 10_000
        self.training_frequency = 10
        self.cuda = True

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.cuda else "cpu"
        )

        self.q_network = QNetwork(observation_space, self.network_action_space).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.target_network = QNetwork(observation_space, self.network_action_space).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.replay_buffer = ReplayBuffer(state_dim=flatdim(observation_space), action_dim=1)

        self.global_step = 0

    def act(self, observation: Space) -> RestrictorActionSpace:
        epsilon = linear_schedule(
            self.start_e,
            self.end_e,
            self.exploration_fraction * self.total_timesteps,
            self.global_step,
        )
        if random.random() < epsilon:
            action = self.network_action_space.sample()
        else:
            q_values = self.q_network(torch.Tensor(observation).to(self.device))
            action = torch.argmax(q_values).cpu().numpy()

        return action

    def learn(self, obs, action, next_obs, reward, done):
        self.record(obs, action, next_obs, reward, done)

        if self.global_step >= self.training_start:
            if (self.global_step - self.training_start) % self.training_frequency == 0:
                self.update_q_network()

            if (self.global_step - self.training_start) % self.target_network_frequency == 0:
                self.update_target_network()

        self.global_step += 1

    def record(self, obs, action, next_obs, reward, done):
        self.replay_buffer.add(obs, action, next_obs, reward, done)

    def update_q_network(self):
        states, actions, next_states, rewards, not_dones = self.replay_buffer.sample(self.batch_size)

        actions = actions.type(torch.int64)
        rewards = rewards.flatten()
        not_dones = not_dones.flatten()

        # print(f'{states=}, {actions=}, {next_states=}, {rewards=}, {not_dones=}')

        # Get values from target network
        with torch.no_grad():
            target_max, _ = self.target_network(next_states).max(dim=1)

            # print(f'{target_max=}')

            td_target = rewards.flatten() + self.gamma * target_max * not_dones

            # print(f'{td_target=}')
        
        # Get values from q network
        # print(f'{self.q_network(states)=}')
        old_val = self.q_network(states).gather(1, actions).squeeze()

        # print(f'{old_val=}')

        # Compute loss
        loss = F.mse_loss(td_target, old_val)

        # Optimize model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        for target_network_param, q_network_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_network_param.data.copy_(
                self.tau * q_network_param.data + (1.0 - self.tau) * target_network_param.data
            )

    # def train(self):
    #     states, actions, next_states, rewards, not_dones = self.replay_buffer.sample(self.batch_size)

    #     with torch.no_grad():
    #         target_max, _ = self.target_network(next_states).max(dim=1)
    #         td_target = rewards.flatten() + self.gamma * target_max * (1 - not_dones.flatten())
    #     old_val = self.q_network(states).gather(1, actions).squeeze()
    #     loss = F.mse_loss(td_target, old_val)

    #     # if global_step % 100 == 0:
    #     #     writer.add_scalar("losses/td_loss", loss, global_step)
    #     #     writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
    #     #     print("SPS:", int(global_step / (time.time() - start_time)))
    #     #     writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    #     # # optimize the model
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    #     # update target network
    #     if self.global_step % self.target_network_frequency == 0:
    #         for target_network_param, q_network_param in zip(self.target_network.parameters(), self.q_network.parameters()):
    #             target_network_param.data.copy_(
    #                 self.tau * q_network_param.data + (1.0 - self.tau) * self.target_network_param.data
    #             )

    def postprocess_restriction(self, restriction):
        allowed_actions = self.valid_route_restrictions[restriction]

        return DiscreteVectorRestriction(self.action_space.base_space, allowed_actions=allowed_actions)
