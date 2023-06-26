# Typing
from typing import Any

# Standard modules
from collections import OrderedDict
from functools import singledispatch

# External modules
import gymnasium.spaces.utils
import numpy as np
from numpy.typing import NDArray
from gymnasium import Space
from gymnasium.spaces import Dict
from gymnasium.spaces.utils import FlatType, T

# Internal modules
from drama.restrictions import (
    DiscreteSetRestriction,
    DiscreteVectorRestriction,
    IntervalUnionRestriction,
    BucketSpaceRestriction,
    Restriction,
)
from drama.restrictors import (
    DiscreteSetActionSpace,
    DiscreteVectorActionSpace,
    IntervalUnionActionSpace,
    BucketSpaceActionSpace,
)


class IntervalsOutOfBoundException(Exception):
    def __init__(self, *args) -> None:
        super().__init__(*args)


class RestrictionViolationException(Exception):
    def __init__(self, *args) -> None:
        super().__init__(*args)


def random_replacement_violation_fn(env, action, restriction: Restriction):
    env.step(restriction.sample())


def projection_violation_fn(env, action, restriction: IntervalUnionRestriction):
    env.step(np.array([restriction.nearest_element(action.item())], dtype=np.float32))


# flatten functions for restriction classes
@singledispatch
def flatten(space: Space, x: T, **kwargs) -> FlatType:
    return gymnasium.spaces.utils.flatten(space, x)


@flatten.register(Dict)
def _flatten_dict(space: Dict, x: T, **kwargs):
    if space.is_np_flattenable:
        return np.concatenate(
            [np.array(flatten(s, x[key], **kwargs)) for key, s in space.spaces.items()]
        )
    return OrderedDict(
        (key, flatten(s, x[key], **kwargs)) for key, s in space.spaces.items()
    )


@flatten.register(DiscreteSetActionSpace)
def _flatten_discrete_set_action_space(
    space: DiscreteSetActionSpace, x: DiscreteSetRestriction
) -> NDArray:
    return np.asarray(list(x.allowed_actions), dtype=space.dtype)


@flatten.register(DiscreteVectorActionSpace)
def _flatten_discrete_vector_action_space(
    space: DiscreteVectorActionSpace, x: DiscreteVectorRestriction
):
    return np.asarray(x.allowed_actions, dtype=space.dtype)


@flatten.register(IntervalUnionActionSpace)
def _flatten_interval_union_restriction(
    space: IntervalUnionActionSpace,
    x: IntervalUnionRestriction,
    pad: bool = True,
    clamp: bool = True,
    max_len: int = 7,
    pad_value: float = 0.0,
    raise_error: bool = True,
):
    intervals = np.array(x.intervals(), dtype=np.float32)
    if raise_error and intervals.shape[0] > max_len:
        raise IntervalsOutOfBoundException
    if clamp:
        intervals = intervals[:max_len]
    if pad:
        padding = np.full((max_len - intervals.shape[0], 2), pad_value)
        return (
            np.concatenate([intervals, padding], axis=0, dtype=np.float32).flatten()
            if len(intervals) > 0
            else padding.flatten()
        )
    return intervals.flatten()


@flatten.register(BucketSpaceActionSpace)
def _flatten_bucket_space_action_space(
    space: BucketSpaceActionSpace, x: BucketSpaceRestriction
):
    return np.concatenate(
        [
            np.array(
                [
                    space.base_space.low.item(),
                    space.base_space.high.item(),
                    space.number_of_buckets,
                ]
            ),
            x.buckets,
        ]
    )


# flatdim functions for restriction classes
@singledispatch
def flatdim(space: Space[Any]) -> int:
    return gymnasium.spaces.utils.flatdim(space)


@flatdim.register(DiscreteSetActionSpace)
def _flatdim_discrete_set_action_space(space: DiscreteSetActionSpace):
    raise ValueError(
        "Cannot get flattened size as the DiscreteSetActionSpace has a dynamic size."
    )


@flatdim.register(DiscreteVectorActionSpace)
def _flatdim_discrete_vector_action_space(space: DiscreteVectorActionSpace) -> int:
    return space.base_space.n


@flatdim.register(IntervalUnionActionSpace)
def _flatdim_graph(space: IntervalUnionActionSpace):
    raise ValueError(
        "Cannot get flattened size as the IntervalUnionActionSpace has a dynamic size."
    )


@flatdim.register(BucketSpaceActionSpace)
def _flatdim_bucket_space_action_space(space: BucketSpaceActionSpace) -> int:
    return space.number_of_buckets + 3


# unflatten functions for restriction classes
@singledispatch
def unflatten(space: Space[T], x: FlatType) -> T:
    return gymnasium.spaces.utils.unflatten(space, x)


@unflatten.register(DiscreteSetActionSpace)
def _unflatten_discrete_set_action_space(
    space: DiscreteSetActionSpace, x: FlatType
) -> DiscreteSetRestriction:
    return DiscreteSetRestriction(space.base_space, allowed_actions=set(x))


@unflatten.register(DiscreteVectorActionSpace)
def _unflatten_discrete_vector_action_space(
    space: DiscreteVectorActionSpace, x: FlatType
) -> DiscreteVectorRestriction:
    return DiscreteVectorRestriction(space.base_space, allowed_actions=np.array(x))


@unflatten.register(IntervalUnionActionSpace)
def _unflatten_interval_union_action_space(
    space: IntervalUnionActionSpace, x: FlatType
) -> T:
    unpadded_intervals = np.array(
        [[x[i], x[i + 1]] for i in range(0, len(x), 2) if x[i] != x[i + 1]]
    ).flatten()
    interval_union_space = IntervalUnionRestriction(space.base_space)
    for i in list(range(0, len(unpadded_intervals) + 2, 2)):
        if i == 0:
            if unpadded_intervals[i] != space.base_space.low.item():
                interval_union_space.remove(
                    space.base_space.low[0], unpadded_intervals[i - 1]
                )
        elif i == len(unpadded_intervals):
            if unpadded_intervals[i - 1] != space.base_space.high.item():
                interval_union_space.remove(
                    unpadded_intervals[i - 1], space.base_space.high.item()
                )
        else:
            interval_union_space.remove(
                unpadded_intervals[i - 1], unpadded_intervals[i]
            )
    return interval_union_space


@unflatten.register(BucketSpaceActionSpace)
def _unflatten_bucket_space_action_space(
    space: BucketSpaceActionSpace, x: FlatType
) -> BucketSpaceRestriction:
    return BucketSpaceRestriction(
        space.base_space, space.bucket_width, space.epsilon, x[3:]
    )
