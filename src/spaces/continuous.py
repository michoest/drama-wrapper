from typing import SupportsFloat, Any, Union, Optional

import gymnasium as gym
import numpy as np


class ContinuousActionSpace(gym.spaces.Box):
  def __init__(self, low: SupportsFloat, high: SupportsFloat, dtype: Union[type[np.floating[Any]], type[np.integer[Any]]] = np.float32, seed: Optional[Union[int, np.random.Generator]] = None) -> None:
    super().__init__(low, high, shape=None, dtype=dtype, seed=seed)