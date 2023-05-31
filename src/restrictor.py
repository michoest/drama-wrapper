from typing import Any

import gymnasium

import numpy as np

from src.restrictions import Restriction


class Restrictor:
    def __init__(self) -> None:
        pass

    def preprocess_observation(self, env):
        return env.state()

    def act(self, observation):
        raise NotImplementedError


class RestrictorActionSpace(gymnasium.Space):
    def __init__(
        self, base_space: gymnasium.Space, seed: int | np.random.Generator | None = None
    ):
        super().__init__(None, None, seed)
        self.base_space = base_space

    def contains(self, x: Restriction) -> bool:
        return x.base_space == self.base_space

    def sample(self, mask: Any | None = None) -> Any:
        return self.base_space

    def is_compatible_with(self, action_space):
        return self.base_space == action_space
