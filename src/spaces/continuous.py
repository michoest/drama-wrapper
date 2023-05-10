from typing import SupportsFloat, Any, Union, Optional

import gymnasium as gym
import numpy as np


class ContinuousActionSpace(gym.spaces.Box):
    """ Node in the AVL tree which represents a valid interval """

    def __init__(self, low: SupportsFloat, high: SupportsFloat,
                 dtype: Union[type[np.floating[Any]], type[np.integer[Any]]] = np.float32,
                 seed: Optional[Union[int, np.random.Generator]] = None) -> None:
        super().__init__(low, high, shape=None, dtype=dtype, seed=seed)

    def reset(self):
        """ Resets the action space to its initial state """

        raise NotImplementedError
