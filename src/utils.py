from collections import OrderedDict
from functools import singledispatch

import gymnasium.spaces.utils
import numpy as np
from gymnasium import Space
from gymnasium.spaces import Dict
from gymnasium.spaces.utils import FlatType

from src.restrictions import IntervalUnionRestriction

from gymnasium.spaces.utils import T

from src.restrictors import IntervalUnionActionSpace


class IntervalsOutOfBoundException(Exception):
    def __init__(self, *args) -> None:
        super().__init__(*args)


class RestrictionViolationException(Exception):
    def __init__(self, *args) -> None:
        super().__init__(*args)


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
