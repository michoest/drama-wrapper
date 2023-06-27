# Typing
import math
from typing import Any

# Standard modules
from abc import ABC

# External modules
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo import AECEnv

# Internal modules
from drama.restrictions import (
    Restriction,
    IntervalUnionRestriction,
    DiscreteVectorRestriction,
    DiscreteSetRestriction,
    BucketSpaceRestriction,
)


class RestrictorActionSpace(ABC, gym.Space):
    """Action space representing sets of restrictions for a given base space."""
    def __init__(
        self, base_space: gym.Space, seed: int | np.random.Generator | None = None
    ):
        """Constructor of :class:`RestrictorActionSpace`.

        Args:
            base_space: The base action space which is restricted
            seed: Optionally, you can use this argument to seed the RNG that is used to sample from the space
        """
        super().__init__(None, None, seed)
        self.base_space = base_space

    def contains(self, x: Restriction) -> bool:
        """Check if a restriction was created with the same base space.

        Args:
            x: The restriction

        Returns:
            `True` if the restriction is compatible with the restrictor action space.
            `False` otherwise
        """
        return x.base_space == self.base_space

    def sample(self, mask: Any | None = None) -> Restriction:
        """Randomly sample a restriction from the restrictor action space.

        Args:
            mask: The mask used for sampling

        Returns:
            A sampled restriction
        """
        raise NotImplementedError

    def is_compatible_with(self, action_space: gym.Space):
        """Check if a action space is compatible with the restrictor action space.

        Args:
            action_space: The action space which is checked for compatibility

        Returns:
            `True` if the action space is compatible with the restrictor action space.
            `False` otherwise
        """
        return self.base_space == action_space

    def __repr__(self) -> str:
        """String representation of the restrictor action space."""
        return f"{self.__class__.__name__}(base_space={self.base_space})"


class Restrictor(ABC):
    """An agent whose actions are restrictions."""
    def __init__(self, observation_space, action_space) -> None:
        """Constructor of :class:`Restrictor`.

        Args:
            observation_space: The observation space of the restrictor
            action_space: The action space of the restrictor
        """
        self.observation_space = observation_space
        self.action_space = action_space

    def preprocess_observation(self, env: AECEnv) -> Any:
        """Pre-processing function applied by the :class:`RestrictionWrapper`
        before the observation is forwarded to act().

        Args:
            env: The environment at the point in time

        Returns:
            The restrictor observation
        """
        return env.state()

    def act(self, observation: gym.Space) -> Restriction:
        """Compute the restriction for a observation.

        Args:
            observation: The observation used to compute the restriction

        Returns:
            The computed restriction
        """
        raise NotImplementedError


class DiscreteSetActionSpace(RestrictorActionSpace):
    """Action space representing valid restrictions for a discrete action space as a set of allowed actions."""
    def __init__(self, base_space: Discrete):
        """Constructor of :class:`DiscreteSetActionSpace`.

        Args:
            base_space: The :class:`gymnasium.spaces.Discrete` action space which is restricted
        """
        super().__init__(base_space)

    @property
    def is_np_flattenable(self) -> bool:
        """Checks whether this space can be flattened to a :class:`gymnasium.spaces.Box`.

        Returns:
            `True`
        """
        return True

    def sample(self, mask: Any | None = None) -> DiscreteSetRestriction:
        """Randomly sample an instance of :class:`DiscreteSetRestriction` from the :class:`DiscreteSetActionSpace`.

        Args:
            mask: The mask used for sampling (currently no effect)

        Returns:
            A sampled :class:`DiscreteSetRestriction`
        """
        assert isinstance(self.base_space, Discrete)

        discrete_set = DiscreteSetRestriction(
            self.base_space,
            allowed_actions=set(
                np.arange(
                    self.base_space.start, self.base_space.start + self.base_space.n
                )[np.random.choice([True, False], size=self.base_space.n)]
            ),
        )

        return discrete_set


class DiscreteVectorActionSpace(RestrictorActionSpace):
    """Action space representing valid restrictions for a discrete action space as a binary vector."""
    def __init__(self, base_space: Discrete):
        """Constructor of :class:`DiscreteVectorActionSpace`.

        Args:
            base_space: The :class:`gymnasium.spaces.Discrete` action space which is restricted
        """
        super().__init__(base_space)

    @property
    def is_np_flattenable(self) -> bool:
        """Checks whether this space can be flattened to a :class:`gymnasium.spaces.Box`.

        Returns:
            `True`
        """
        return True

    def sample(self, mask: Any | None = None) -> DiscreteVectorRestriction:
        """Randomly sample an instance of :class:`DiscreteVectorRestriction` from the :class:`DiscreteVectorActionSpace`.

        Args:
            mask: The mask used for sampling (currently no effect)

        Returns:
            A sampled :class:`DiscreteVectorRestriction`
        """
        assert isinstance(self.base_space, Discrete)

        discrete_vector = DiscreteVectorRestriction(
            self.base_space,
            allowed_actions=np.random.choice([True, False], self.base_space.n),
        )

        return discrete_vector


class IntervalUnionActionSpace(RestrictorActionSpace):
    """Action space representing valid restrictions for a :class:`gymnasium.spaces.Box` action space
    as a union of intervals."""
    def __init__(self, base_space: Box):
        """Constructor of :class:`IntervalUnionActionSpace`.

        Args:
            base_space: The :class:`gymnasium.spaces.Box` action space which is restricted
        """
        super().__init__(base_space)

    @property
    def is_np_flattenable(self) -> bool:
        """Checks whether this space can be flattened to a :class:`gymnasium.spaces.Box`.

        Returns:
            `True`
        """
        return True

    def sample(self, mask: Any | None = None) -> IntervalUnionRestriction:
        """Randomly sample an instance of :class:`IntervalUnionRestriction` from the :class:`IntervalUnionActionSpace`.

        Args:
            mask: The mask used for sampling (currently no effect)

        Returns:
            A sampled :class:`IntervalUnionRestriction`
        """
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


class BucketSpaceActionSpace(RestrictorActionSpace):
    """Action space representing valid restrictions for a :class:`gymnasium.spaces.Box` action space
    as a binary indicator vector for evenly split buckets."""
    def __init__(self, base_space: Box, bucket_width=1.0, epsilon=0.01):
        """Constructor of :class:`BucketSpaceActionSpace`.

        Args:
            base_space: The :class:`gymnasium.spaces.Box` action space which is restricted
            bucket_width: The width of each bucket
            epsilon: The radius in which buckets are set valid/invalid around a specific point
        """
        super().__init__(base_space)
        assert isinstance(self.base_space, Box)
        self.bucket_width = bucket_width
        self.epsilon = epsilon
        self.number_of_buckets = math.ceil(
            (self.base_space.high.item() - self.base_space.low.item())
            / self.bucket_width
        )

    @property
    def is_np_flattenable(self) -> bool:
        """Checks whether this space can be flattened to a :class:`gymnasium.spaces.Box`.

        Returns:
            `True`
        """
        return True

    def sample(self, mask: Any | None = None) -> BucketSpaceRestriction:
        """Randomly sample an instance of :class:`BucketSpaceRestriction` from the :class:`BucketSpaceActionSpace`.

        Args:
            mask: The mask used for sampling (currently no effect)

        Returns:
            A sampled :class:`BucketSpaceRestriction`
        """
        assert isinstance(self.base_space, Box)

        return BucketSpaceRestriction(
            self.base_space,
            self.bucket_width,
            self.epsilon,
            available_buckets=np.random.choice([True, False], self.number_of_buckets),
        )


class PredicateActionSpace(RestrictorActionSpace):
    pass
