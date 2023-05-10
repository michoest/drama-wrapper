import math
from decimal import Decimal
from random import uniform
from typing import SupportsFloat, Union, Optional, Any

import numpy as np

from src.spaces.continuous import ContinuousActionSpace


class BucketSpace(ContinuousActionSpace):
    """ Interval Action Space as predefined buckets """

    def __init__(self, low: SupportsFloat, high: SupportsFloat, bucket_width=1.0, epsilon=0.01,
                 dtype: Union[type[np.floating[Any]], type[np.integer[Any]]] = np.float32,
                 seed: Optional[Union[int, np.random.Generator]] = None) -> None:
        super().__init__(low, high, dtype, seed)

        self.a, self.b = Decimal(f'{self.low[0]}'), Decimal(f'{self.high[0]}')
        self.bucket_width, self.epsilon = Decimal(f'{bucket_width}'), Decimal(f'{epsilon}')
        self.number_of_buckets = math.ceil((self.b - self.a) / self.bucket_width)
        self.buckets = np.ones((self.number_of_buckets,), dtype=bool)

    def contains(self, x):
        """ Determines if a number is part of the action space

        Args:
            x: Number

        Returns:
            Boolean indicating if it is part of the action space
        """
        return False if x < self.a or x >= self.b else self.buckets[self._bucket(x)]

    def sample(self, mask: None = None):
        """ Sample a random action from a uniform distribution over the action space

        Args:
            mask: A mask for sampling values from the Box space, currently unsupported.

        Returns:
            Sampled action as a float
        """
        if not self.intervals:
            return None
        else:
            x = Decimal(f'{uniform(0.0, float(self.b - self.a))}')

            for i, (a, b) in enumerate(self.intervals):
                if x > Decimal(b) - Decimal(a):
                    x -= Decimal(b) - Decimal(a)
                else:
                    return Decimal(a) + x

        return self.intervals[-1][1]

    def clone(self):
        """ Returns a copy of the action space

        Returns:
            space: Action space copy
        """
        space = BucketSpace(self.a, self.b, bucket_width=float(self.bucket_width), epsilon=float(self.epsilon))
        space.buckets = np.copy(self.buckets)
        return space

    def clone_and_remove(self, x):
        """ Returns a copy of the action space in which buckets containing a specific value are removed

        Args:
            x: Buckets containing this value should be removed from the action space

        Returns:
            space: Action space copy
        """
        space = self.clone()
        space.remove(x)
        return space

    def remove(self, x, with_epsilon=True):
        """ Removes buckets containing a specific value from the action space

        Args:
            x: Value with which buckets are to be removed
            with_epsilon: Whether a subset of epsilon around x should be removed
        """
        x = Decimal(f'{x}')

        if with_epsilon:
            self._set(x, False)
        else:
            self.buckets[self._bucket(x)] = False

    def add(self, x, with_epsilon=True):
        """ Add buckets containing a specific value to the action space

        Args:
            x: Value with which buckets are to be added
            with_epsilon: Whether a subset of epsilon around x should be added
        """
        x = Decimal(f'{x}')

        if with_epsilon:
            self._set(x)
        else:
            self.buckets[self._bucket(x)] = True

    @property
    def intervals(self):
        """ Returns all intervals of the action space ordered

        Returns:
            List of tuples containing the ordered intervals. For example:

            [(0.1,0.5), (0.7,0.9)]
        """
        a, intervals = None, []
        for i in range(self.number_of_buckets):
            if a is None:
                if self.buckets[i]:
                    a = self.a + i * self.bucket_width
            elif not self.buckets[i]:
                intervals.append((float(a), float(self.a + i * self.bucket_width)))
                a = None
            elif i == self.number_of_buckets - 1:
                intervals.append((float(a), float(self.b)))

        return intervals

    def _bucket(self, x):
        """ Finds the bucket which contains a specific value

        Args:
            x: Value for which the bucket has to be found

        Returns:
            Integer (ID) of the bucket
        """
        return math.floor((x - self.a) / self.bucket_width)

    def _set(self, x, value=True):
        lower_bucket = self._bucket(x - self.epsilon) if x - self.epsilon >= self.a else None
        upper_bucket = self._bucket(x + self.epsilon) if x + self.epsilon <= self.b else None

        if lower_bucket is None:
            if upper_bucket is None:
                self.buckets = np.ones((self.number_of_buckets,), dtype=bool) if value else np.zeros(
                    (self.number_of_buckets,), dtype=bool)
            else:
                self.buckets[:upper_bucket + 1] = value
        else:
            if upper_bucket is None:
                self.buckets[lower_bucket:] = value
            else:
                self.buckets[lower_bucket:upper_bucket + 1] = value

    def reset(self):
        """ Resets the action space to the unrestricted state
        """
        self.buckets = np.ones((self.number_of_buckets,), dtype=bool)

    def __str__(self):
        intervals = ' '.join(f'[{float(a)}, {float(b)})' for a, b in self.intervals) if self.intervals else '()'
        return f'<BucketSpace {intervals}>'

    def __repr__(self):
        return self.__str__()

    def __bool__(self):
        return bool(np.any(self.buckets))

    def __contains__(self, item):
        return self.contains(item)

    def __hash__(self):
        return hash((self.a, self.b, self.bucket_width, tuple(self.intervals)))

    def __eq__(self, other):
        return (self.a, self.b, self.bucket_width, tuple(self.intervals)) == (
            other.a, other.b, other.bucket_width, tuple(other.intervals))
