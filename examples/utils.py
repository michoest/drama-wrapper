import numpy as np
import torch
from typing import Union
import pandas as pd


def play(
    env,
    policies,
    *,
    max_iter=1_000,
    render_mode="human",
    verbose=False,
    record_trajectory=False,
) -> Union[pd.DataFrame, None]:
    env.unwrapped.render_mode = render_mode

    env.reset()

    if render_mode is not None:
        env.render()

    if record_trajectory:
        trajectory = []

    for i, agent in zip(range(max_iter), env.agent_iter()):
        observation, reward, termination, truncation, info = env.last()
        if verbose:
            print(
                f"{agent=}, {observation=}, {reward=}, {termination=}, {truncation=}, \
                {info=}"
            )

        action = (
            policies[agent](observation) if not termination and not truncation else None
        )
        if verbose:
            print(f"{action=}")

        if record_trajectory:
            trajectory.append(
                (agent, observation, reward, termination, truncation, info, action)
            )

        env.step(action)

    if render_mode is not None:
        env.render()

    return (
        pd.DataFrame(
            trajectory,
            columns=[
                "agent",
                "observation",
                "reward",
                "termination",
                "truncation",
                "info",
                "action",
            ],
        )
        if record_trajectory
        else None
    )


def restriction_aware_random_policy(observation):
    observation, restriction = observation["observation"], observation["restriction"]
    return restriction.sample()


class ReplayBuffer(object):
    # Replay buffer from https://github.com/sfujim/TD3/blob/master/utils.py (https://arxiv.org/abs/1802.09477)
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
