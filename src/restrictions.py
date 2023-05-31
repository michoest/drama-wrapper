from typing import Optional, Set

from abc import ABC
from random import random

import numpy as np

import gymnasium


class Restriction(ABC, gymnasium.Space):
    def __init__(
        self,
        base_space: gymnasium.Space,
        *,
        seed: int | np.random.Generator | None = None,
    ):
        super().__init__(base_space.shape, base_space.dtype, seed)
        self.base_space = base_space

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class DiscreteRestriction(Restriction):
    def __init__(
        self,
        base_space: gymnasium.spaces.Discrete,
        *,
        seed: int | np.random.Generator | None = None,
    ):
        super().__init__(base_space, seed=seed)


class ContinuousRestriction(Restriction):
    def __init__(
        self,
        base_space: gymnasium.spaces.Box,
        *,
        seed: int | np.random.Generator | None = None,
    ):
        super().__init__(base_space, seed=seed)


class DiscreteSetRestriction(DiscreteRestriction):
    def __init__(
        self,
        base_space: gymnasium.spaces.Discrete,
        *,
        allowed_actions: Optional[Set[int]] = None,
        seed: int | np.random.Generator | None = None,
    ):
        super().__init__(base_space, seed=seed)

        self.allowed_actions = (
            allowed_actions
            if allowed_actions is not None
            else set(range(base_space.start, base_space.start + base_space.n))
        )

    @property
    def is_np_flattenable(self) -> bool:
        return True

    def sample(self) -> int:
        return random.choice(tuple(self.allowed_actions))

    def contains(self, x: int) -> bool:
        return x in self.allowed_actions

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.allowed_actions})"


class DiscreteVectorRestriction(DiscreteRestriction):
    def __init__(
        self,
        base_space: gymnasium.spaces.Discrete,
        *,
        allowed_actions: Optional[np.ndarray[bool]] = None,
        seed: int | np.random.Generator | None = None,
    ):
        super().__init__(base_space, seed=seed)

        self.allowed_actions = (
            allowed_actions
            if allowed_actions is not None
            else set(range(base_space.start, base_space.start + base_space.n))
        )

    @property
    def is_np_flattenable(self) -> bool:
        return True

    def sample(self) -> int:
        return self.start + random.choice(
            tuple(index for index, value in enumerate(self.allowed_actions) if value)
        )

    def contains(self, x: int) -> bool:
        return self.allowed_actions[x - self.start]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.allowed_actions})"


class IntervalUnionRestriction(ContinuousRestriction):
    pass
