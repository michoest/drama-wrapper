# Enable module import from ../src
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

# External modules
import unittest
from gymnasium.spaces import Discrete
import numpy as np

# Internal modules
from src.utils import flatdim, flatten, unflatten
from src.restrictors import (
    DiscreteSetActionSpace,
    DiscreteVectorActionSpace,
    IntervalUnionActionSpace,
    BucketSpaceActionSpace,
)
from src.restrictions import DiscreteSetRestriction, DiscreteVectorRestriction


class DiscreteSetActionSpaceTest(unittest.TestCase):
    b1, b2 = Discrete(5), Discrete(3, start=2)

    s1 = DiscreteSetActionSpace(b1)
    s2 = DiscreteSetActionSpace(b2)

    r1 = DiscreteSetRestriction(b1, allowed_actions={1, 2, 3})
    r2 = DiscreteSetRestriction(b2, allowed_actions={2, 4})

    def test_flatten(self):
        self.assertTrue(np.array_equal(flatten(self.s1, self.r1), np.array([1, 2, 3])))
        self.assertTrue(np.array_equal(flatten(self.s2, self.r2), np.array([2, 4])))

    def test_flatdim(self):
        with self.assertRaises(ValueError):
            flatdim(self.s1)
        with self.assertRaises(ValueError):
            flatdim(self.s2)

    def test_unflatten(self):
        self.assertTrue(unflatten(self.s1, np.array([1, 2, 3])) == self.r1)
        self.assertTrue(unflatten(self.s2, np.array([2, 4])) == self.r2)


class DiscreteVectorActionSpaceTest(unittest.TestCase):
    pass


class IntervalUnionActionSpaceTest(unittest.TestCase):
    pass


class BucketSpaceActionSpaceTest(unittest.TestCase):
    pass


class PredicateActionSpaceTest(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
