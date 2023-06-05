from abc import ABC
from typing import Any, Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from pettingzoo import AECEnv

from src.restrictions import Restriction, IntervalUnionRestriction


class RestrictorActionSpace(ABC, gym.Space):
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
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(base_space={self.base_space})"


class Restrictor(ABC):
    def __init__(self, observation_space, action_space) -> None:
        self.observation_space = observation_space
        self.action_space = action_space

    def preprocess_observation(
        self, env: AECEnv
    ) -> Union[np.ndarray, torch.TensorType]:
        return env.state()

    def act(self, observation: gym.Space) -> RestrictorActionSpace:
        raise NotImplementedError


class IntervalUnionActionSpace(RestrictorActionSpace):
    def __init__(self, base_space: Box):
        super().__init__(base_space)

    @property
    def is_np_flattenable(self) -> bool:
        return True

    def sample(self, mask: Any | None = None) -> IntervalUnionRestriction:
        assert isinstance(self.base_space, Box)

        interval_union = IntervalUnionRestriction(self.base_space)
        num_intervals = self.np_random.geometric(0.25)

        for _ in range(num_intervals):
            interval_start = self.np_random.uniform(
                self.base_space.low[0], self.base_space.high[0]
            )
            interval_union.remove(
                interval_start,
                self.np_random.uniform(interval_start, self.base_space.high[0]),
            )
        return interval_union
