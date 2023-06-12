from typing import Any

from collections import OrderedDict
from functools import singledispatch

import gymnasium.spaces.utils
import numpy as np
from gymnasium import Space
from gymnasium.spaces import Dict
from gymnasium.spaces.utils import FlatType

from src.restrictions import IntervalUnionRestriction, DiscreteVectorRestriction

from gymnasium.spaces.utils import T

from src.restrictors import IntervalUnionActionSpace, DiscreteVectorActionSpace


class IntervalsOutOfBoundException(Exception):
    def __init__(self, *args) -> None:
        super().__init__(*args)


class RestrictionViolationException(Exception):
    def __init__(self, *args) -> None:
        super().__init__(*args)


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
    return OrderedDict((key, flatten(s, x[key], **kwargs)) for key, s in space.spaces.items())


@flatten.register(IntervalUnionActionSpace)
def _flatten_interval_union_restriction(space: IntervalUnionActionSpace, x: IntervalUnionRestriction,
                                        pad: bool = True, clamp: bool = True, max_len: int = 7,
                                        pad_value: float = 0.0, raise_error: bool = True):
    intervals = np.array(x.intervals(), dtype=np.float32)
    if raise_error and intervals.shape[0] > max_len:
        raise IntervalsOutOfBoundException
    if clamp:
        intervals = intervals[:max_len]
    if pad:
        padding = np.full((max_len - intervals.shape[0], 2), pad_value)
        return np.concatenate([intervals, padding], axis=0, dtype=np.float32).flatten() \
            if len(intervals) > 0 else padding.flatten()
    return intervals.flatten()


@flatten.register(DiscreteVectorActionSpace)
def _flatten_discrete_vector_action_space(space: DiscreteVectorActionSpace, x: DiscreteVectorRestriction):
    return np.asarray(x.allowed_actions, dtype=space.dtype).flatten()


# flatdim functions for restriction classes
@singledispatch
def flatdim(space: Space[Any]) -> int:
    return gymnasium.spaces.utils.flatdim(space)


@flatdim.register(DiscreteVectorActionSpace)
def _flatdim_discrete_vector_action_space(space: DiscreteVectorActionSpace) -> int:
    # print('_flatdim_discrete_vector_action_space')
    return space.base_space.n


# unflatten functions for restriction classes
@singledispatch
def unflatten(space: Space[T], x: FlatType) -> T:
    return gymnasium.spaces.utils.unflatten(space, x)


@unflatten.register(DiscreteVectorActionSpace)
def _unflatten_discrete_vector_action_space(space: DiscreteVectorActionSpace, x: FlatType) -> T:
    return DiscreteVectorRestriction(space.base_space, allowed_actions=x)
