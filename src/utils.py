from functools import singledispatch

import gymnasium.spaces.utils
import numpy as np
from gymnasium import Space
from gymnasium.spaces.utils import FlatType

from src.restrictions import IntervalUnionRestriction, Restriction

from gymnasium.spaces.utils import T

from src.restrictors import IntervalUnionActionSpace


class IntervalsOutOfBoundException(Exception):
    def __init__(self, *args) -> None:
        super().__init__(*args)


@singledispatch
def flatten(space: Space, x: T, **kwargs) -> FlatType:
    return gymnasium.spaces.utils.flatten(space, x)


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
        return np.concatenate([intervals, np.full((max_len - intervals.shape[0], 2), pad_value)],
                              axis=0, dtype=np.float32)
    return intervals
