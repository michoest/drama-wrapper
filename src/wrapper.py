import functools
from copy import copy

import numpy as np
from gymnasium.spaces import Dict
from pettingzoo.utils import BaseWrapper


class RestrictionWrapper(BaseWrapper):
    """ Wrapper to extend the environment with a governance agent.

     Extended Agent-Environment Cycle:
         Reset() -> Governance
         Step() -> Agent
         Step() -> Governance
         ...
     """

    def __init__(self, env, governance_observation_space, governance_action_space,
                 governance_reward_fn=None, preprocess_governance_observation_fn=None):
        super().__init__(env)

        self.governance_observation_space = governance_observation_space
        self.governance_action_space = governance_action_space
        self.governance_reward_fn = governance_reward_fn
        self.preprocess_governance_observation_fn = preprocess_governance_observation_fn

        # self.observations and self.restrictions are dictionaries which keep the latest values for each agent
        self.observations = None
        self.restrictions = None
        self.possible_agents = ['gov_0'] + self.possible_agents

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        if agent == 'gov_0':
            return self.governance_observation_space
        else:
            return Dict({
                'observation': self.env.observation_space(agent),
                'restriction': self.governance_action_space
            })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        if agent == 'gov_0':
            return self.governance_action_space
        else:
            return self.env.action_space(agent)

    def reset(self, seed=None, options=None):
        # First update attributes of agents in the original environment
        super().reset(seed, options)
        self.observations = {agent: None for agent in self.env.possible_agents}
        self.restrictions = {agent: None for agent in self.env.possible_agents}

        # Then update the governance attributes
        self.agents = copy(self.possible_agents)
        self.rewards['gov_0'] = 0.0
        self._cumulative_rewards['gov_0'] = 0.0
        self.terminations['gov_0'] = False
        self.truncations['gov_0'] = False
        self.infos['gov_0'] = {}

        # Start an episode with the governance to obtain restrictions
        self.agent_selection = 'gov_0'

    def step(self, action):
        if self.agent_selection == 'gov_0':
            # If the action was taken by the governance, check if it was terminated last step
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

            # Only setup the governance for the next cycle if there are agents left
            if len(self.agents) >= 1:
                # If more restrictions are required, configure the governance
                # If all agents are now terminated or truncated then also terminate the governance
                self.agents = ['gov_0'] + self.env.agents
                self.terminations['gov_0'] = True \
                    if np.all(np.logical_or(np.array(list(self.terminations.values())),
                                            np.array(list(self.truncations.values())))) else False
                self.truncations['gov_0'] = False
                self.infos['gov_0'] = {}
                self.rewards['gov_0'] = self.governance_reward_fn(self.rewards, self.env.state()) \
                    if self.governance_reward_fn else sum(self.rewards.values())
                self._cumulative_rewards['gov_0'] += self.rewards['gov_0']

                # Switch back to the governance
                self.agent_selection = 'gov_0'

    def observe(self, agent: str):
        if agent == 'gov_0':
            return self.preprocess_governance_observation_fn(self.observations, self.env.state()) \
                if self.preprocess_governance_observation_fn else self.env.state()
        else:
            return {
                'observation': super().observe(agent),
                'restriction': self.restrictions[agent]
            }
