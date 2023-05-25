import functools
from copy import copy

import numpy as np
from gymnasium.spaces import Dict
from pettingzoo.utils import BaseWrapper


class RestrictionWrapper(BaseWrapper):
    """ Wrapper to extend the environment with a restrictor agent.

     Extended Agent-Environment Cycle:
         Reset() -> Restrictor
         Step() -> Agent
         Step() -> Restrictor
         ...
     """

    def __init__(self, env, restrictor_observation_space, restrictor_action_space, *,
                 restrictor_reward_fn=None, preprocess_restrictor_observation_fn=None,
                 restrictor_key='restrictor_0', restriction_key='restriction', 
                 observation_key='observation'):
        super().__init__(env)

        self.restrictor_observation_space = restrictor_observation_space
        self.restrictor_action_space = restrictor_action_space
        self.restrictor_reward_fn = restrictor_reward_fn
        self.preprocess_restrictor_observation_fn = preprocess_restrictor_observation_fn

        self.restrictor_key = restrictor_key
        self.restriction_key = restriction_key
        self.observation_key = observation_key

        # self.observations and self.restrictions are dictionaries which keep the latest values for each agent
        self.observations = None
        self.restrictions = None
        self.possible_agents = [self.restrictor_key] + self.possible_agents

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        if agent == self.restrictor_key:
            return self.restrictor_observation_space
        else:
            return Dict({
                self.observation_key: self.env.observation_space(agent),
                self.restriction_key: self.restrictor_action_space
            })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        if agent == self.restrictor_key:
            return self.restrictor_action_space
        else:
            return self.env.action_space(agent)

    def reset(self, seed=None, options=None):
        # First update attributes of agents in the original environment
        super().reset(seed, options)
        self.observations = {agent: None for agent in self.env.possible_agents}
        self.restrictions = {agent: None for agent in self.env.possible_agents}

        # Then update the restrictor attributes
        self.agents = copy(self.possible_agents)
        self.rewards[self.restrictor_key] = 0.0
        self._cumulative_rewards[self.restrictor_key] = 0.0
        self.terminations[self.restrictor_key] = False
        self.truncations[self.restrictor_key] = False
        self.infos[self.restrictor_key] = {}

        # Start an episode with the restrictor to obtain restrictions
        self.agent_selection = self.restrictor_key

    def step(self, action):
        if self.agent_selection == self.restrictor_key:
            # If the action was taken by the restrictor, check if it was terminated last step
            if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
                self._was_dead_step(action)
                self.agent_selection = self.env.agent_selection
                return

            # Otherwise set the restrictions that apply to the next agent.
            self.restrictions[self.env.agent_selection] = action

            # Switch to the next agent of the original environment
            self.agent_selection = self.env.agent_selection
        else:
            # If the action was taken by an agent, execute it in the original environment
            super().step(action)

            # Only setup the restrictor for the next cycle if there are agents left
            if self.agents:
                # If more restrictions are required, configure the restrictor
                # If all agents are now terminated or truncated then also terminate the restrictor
                self.agents = [self.restrictor_key] + self.env.agents
                print(f'{self.terminations=}, {self.truncations=}')
                self.truncations[self.restrictor_key] = False
                self.terminations[self.restrictor_key] = True \
                    if np.all(np.logical_or(np.array(list(self.terminations.values())),
                                            np.array(list(self.truncations.values())))) else False
                self.infos[self.restrictor_key] = {}
                self.rewards[self.restrictor_key] = self.restrictor_reward_fn(self.rewards, self.env) \
                    if self.restrictor_reward_fn else sum(self.rewards.values())
                self._cumulative_rewards[self.restrictor_key] += self.rewards[self.restrictor_key]

                # Switch back to the restrictor
                self.agent_selection = self.restrictor_key

    def observe(self, agent: str):
        if agent == self.restrictor_key:
            return self.preprocess_restrictor_observation_fn(self.env, self.env.agent_selection, self.observations) \
                if self.preprocess_restrictor_observation_fn else self.env
        else:
            return {
                self.observation_key: super().observe(agent),
                self.restriction_key: self.restrictions[agent]
            }
