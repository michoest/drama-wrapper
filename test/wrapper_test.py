# Enable module import from ../src
import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

# Standard modules


# External modules
import unittest
from pettingzoo import AECEnv
from pettingzoo.classic import rps_v2
from gymnasium.spaces import Discrete

# Internal modules
from drama.wrapper import RestrictionWrapper
from drama.restrictors import Restrictor, DiscreteSetActionSpace

# from src.restrictions import


class RestrictionWrapperTest(unittest.TestCase):
    def test_restrictor_init(self):
        env = rps_v2.env()
        restrictors = Restrictor(Discrete(1), DiscreteSetActionSpace(Discrete(3)))
        RestrictionWrapper(env, restrictors)

        restrictors = {
            "restrictor_0": Restrictor(
                Discrete(1), DiscreteSetActionSpace(Discrete(3))
            ),
            "restrictor_1": Restrictor(
                Discrete(1), DiscreteSetActionSpace(Discrete(3))
            ),
        }
        with self.assertRaises(AssertionError):
            RestrictionWrapper(env, restrictors)

        RestrictionWrapper(
            env,
            restrictors,
            agent_restrictor_mapping={
                "player_0": "restrictor_0",
                "player_1": "restrictor_1",
            },
        )


if __name__ == "__main__":
    unittest.main()
