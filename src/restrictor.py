from typing import Any

import gymnasium as gym
import numpy as np
from pettingzoo import AECEnv

from src.restriction import Restriction


class RestrictorActionSpace(gym.Space):
    def __init__(
        self, base_space: gym.Space, seed: int | np.random.Generator | None = None
    ):
        super().__init__(None, None, seed)
        self.base_space = base_space

    def contains(self, x: Restriction) -> bool:
        return x.base_space == self.base_space

    def sample(self, mask: Any | None = None) -> Restriction:
        raise NotImplementedError

    def is_compatible_with(self, action_space: gym.Space):
        return self.base_space == action_space


class Restrictor:
    def __init__(self) -> None:
        pass

    def preprocess_observation(self, env: AECEnv) -> gym.Space:
        return env.state()

    def act(self, observation: gym.Space) -> RestrictorActionSpace:
        raise NotImplementedError
