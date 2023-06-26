import copy

import numpy as np
from gymnasium.spaces import Box
from pettingzoo import AECEnv

from drama.restrictors import Restrictor, RestrictorActionSpace
from drama.restrictions import IntervalUnionRestriction


class CournotRestrictor(Restrictor):
    def __init__(self, observation_space, action_space) -> None:
        super().__init__(observation_space, action_space)

        self.previous_observation = None
        self.restriction = IntervalUnionRestriction(self.action_space.base_space)
        self.has_restricted = False

    def preprocess_observation(self, env: AECEnv) -> Box:
        return np.array(list(env.state().values()), dtype=float)

    def act(self, observation: Box) -> RestrictorActionSpace:
        if not np.isnan(observation).any():
            if not self.has_restricted and self.previous_observation is not None:
                if np.allclose(observation, self.previous_observation, atol=0.001):
                    estimated_lambda = 3 / 2 * observation.sum()
                    self.restriction.remove(estimated_lambda / 4, estimated_lambda / 2)
                    self.has_restricted = True

            self.previous_observation = observation

        return copy.deepcopy(self.restriction)
