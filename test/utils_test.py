# Enable module import from ../src
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

# External modules
import unittest
from gymnasium.spaces import Discrete
import numpy as np

# Internal modules
from drama.utils import flatdim, flatten, unflatten
from drama.restrictors import (
    DiscreteSetActionSpace,
    DiscreteVectorActionSpace,
    IntervalUnionActionSpace,
    BucketSpaceActionSpace,
)
from drama.restrictions import DiscreteSetRestriction, DiscreteVectorRestriction


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
    b1, b2 = Discrete(5), Discrete(3, start=2)

    s1 = DiscreteVectorActionSpace(b1)
    s2 = DiscreteVectorActionSpace(b2)

    r1 = DiscreteVectorRestriction(b1, allowed_actions=np.array([0, 1, 1, 1, 0]))
    r2 = DiscreteVectorRestriction(b2, allowed_actions=np.array([0, 0, 1]))

    def test_flatten(self):
        self.assertTrue(np.array_equal(flatten(self.s1, self.r1), np.array([0, 1, 1, 1, 0])))
        self.assertTrue(np.array_equal(flatten(self.s2, self.r2), np.array([0, 0, 1])))

    def test_flatdim(self):
        assert flatdim(self.s1) == 5
        assert flatdim(self.s2) == 3

    def test_unflatten(self):
        self.assertEqual(unflatten(self.s1, np.array([0, 1, 1, 1, 0])), self.r1)
        self.assertEqual(unflatten(self.s2, np.array([0, 0, 1])), self.r2)

    def test_no_binary_allowed_actions_initialization(self):
        with self.assertRaises(AssertionError):
            DiscreteVectorRestriction(self.b1, allowed_actions=np.array([0, 2, 1, 1, 0]))

    def test_partial_allowed_actions_initialization(self):
        with self.assertRaises(AssertionError):
            DiscreteVectorRestriction(self.b1, allowed_actions=np.array([0, 1, 1, 0]))


class IntervalUnionActionSpaceTest(unittest.TestCase):
    pass


class BucketSpaceActionSpaceTest(unittest.TestCase):
    pass


class PredicateActionSpaceTest(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
