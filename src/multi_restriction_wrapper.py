import functools

from gymnasium.spaces import Dict
from pettingzoo.utils import BaseWrapper


# If no functions are provided for some or all restrictors, use these defaults
def _default_restrictor_reward_fn(env, rewards):
    return sum(rewards.values())


def _default_preprocess_restrictor_observation_fn(env):
    return env.state()


class MultiRestrictionWrapper(BaseWrapper):
    """Wrapper to extend the environment with one or more restrictor agents.

    Extended Agent-Environment Cycle:
        Reset() -> Restrictor of Agent_0
        Step() -> Agent_0
        Step() -> Restrictor of Agent_1
        Step() -> Agent_1
        Step() -> Restrictor of Agent_2
        ...
    """

    def __init__(
        self,
        env,
        restrictors,
        agent_restrictor_mapping,
        *,
        restrictor_reward_fns=None,
        preprocess_restrictor_observation_fns=None,
        restriction_key="restriction",
        observation_key="observation",
    ):
        super().__init__(env)

        self.restrictors = restrictors
        self.agent_restrictor_mapping = agent_restrictor_mapping

        self.restrictor_reward_fns = {
            restrictor: restrictor_reward_fns[restrictor]
            if restrictor_reward_fns and restrictor_reward_fns.get(restrictor, None)
            else _default_restrictor_reward_fn
            for restrictor in self.restrictors
        }
        self.preprocess_restrictor_observation_fns = {
            restrictor: preprocess_restrictor_observation_fns[restrictor]
            if preprocess_restrictor_observation_fns
            and preprocess_restrictor_observation_fns.get(restrictor, None)
            else _default_preprocess_restrictor_observation_fn
            for restrictor in self.restrictors
        }

        self.restriction_key = restriction_key
        self.observation_key = observation_key

        # self.restrictions is a dictionary which keeps the latest value for each agent
        self.restrictions = None

        self.possible_agents = self.possible_agents + list(self.restrictors)

        # Check if restrictor action spaces match agent action spaces
        for agent in self.env.possible_agents:
            assert self.restrictors[
                self.agent_restrictor_mapping[agent]
            ].action_space.is_compatible_with(
                env.action_space(agent)
            ), f"The action space of {self.agent_restrictor_mapping[agent]} and \
                  {agent} are not compatible!"

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        if agent in self.restrictors:
            return self.restrictors[agent].observation_space
        else:
            return Dict(
                {
                    self.observation_key: self.env.observation_space(agent),
                    self.restriction_key: self.restrictors[
                        self.agent_restrictor_mapping[agent]
                    ].action_space,
                }
            )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        if agent in self.restrictors:
            return self.restrictors[agent].action_space
        else:
            return self.env.action_space(agent)

    def reset(self, seed=None, options=None):
        self.env.reset(seed, options)

        # Set properties
        self.rewards = {
            **self.env.rewards,
            **{restrictor: 0.0 for restrictor in self.restrictors},
        }
        self.terminations = {
            **self.env.terminations,
            **{restrictor: False for restrictor in self.restrictors},
        }
        self.truncations = {
            **self.env.truncations,
            **{restrictor: False for restrictor in self.restrictors},
        }
        self.infos = {
            **self.env.infos,
            **{restrictor: {} for restrictor in self.restrictors},
        }
        self.agents = self.env.agents + list(
            set(self.agent_restrictor_mapping[agent] for agent in self.env.agents)
        )
        self._cumulative_rewards = {
            **self.env._cumulative_rewards,
            **{restrictor: 0.0 for restrictor in self.restrictors},
        }

        self.restrictions = {agent: None for agent in self.env.agents}

        # Start an episode with the restrictor of the first agent to obtain a
        # restriction
        self.agent_selection = self.agent_restrictor_mapping[self.env.agent_selection]

    def step(self, action):
        if self.agent_selection in self.restrictors:
            # If the action was taken by the restrictor, check if it was terminated
            # last step
            if self.terminations[self.agent_selection]:
                self._was_dead_step(action)
                self.agent_selection = self.env.agent_selection
                return

            # Otherwise set the restrictions that apply to the next agent.
            assert (
                self.agent_restrictor_mapping[self.env.agent_selection]
                == self.agent_selection
            )
            self.restrictions[self.env.agent_selection] = action

            # Switch to the next agent of the original environment
            self.agent_selection = self.env.agent_selection
        else:
            # If the action was taken by an agent, execute it in the original
            # environment
            self.env.step(action)

            # Update properties
            self.agents = self.env.agents + list(
                set(self.agent_restrictor_mapping[agent] for agent in self.env.agents)
            )
            self.rewards = {
                **self.env.rewards,
                **{
                    restrictor: self.restrictor_reward_fns[restrictor](
                        self.env, self.rewards
                    )
                    for restrictor in self.restrictors
                },
            }
            self.terminations = {
                **self.env.terminations,
                **{
                    restrictor: all(
                        self.env.terminations[agent] or self.env.truncations[agent]
                        for agent in self.env.agents
                    )
                    for restrictor in self.restrictors
                },
            }
            self.truncations = {
                **self.env.truncations,
                **{restrictor: False for restrictor in self.restrictors},
            }
            self.infos = {
                **self.env.infos,
                **{restrictor: {} for restrictor in self.restrictors},
            }
            self._cumulative_rewards = {
                **self.env._cumulative_rewards,
                **{
                    restrictor: self._cumulative_rewards[restrictor]
                    + self.rewards[restrictor]
                    for restrictor in self.restrictors
                },
            }

            if self.env.agents and all(
                self.env.terminations[agent] or self.env.truncations[agent]
                for agent in self.env.agents
            ):
                # If there are alive agents left, get the next restriction
                self.agent_selection = self.env.agent_selection
            else:
                # Otherwise, get the next restriction
                self.agent_selection = self.agent_restrictor_mapping[
                    self.env.agent_selection
                ]

    def observe(self, agent: str):
        if agent in self.restrictors:
            return self.preprocess_restrictor_observation_fns[agent](self.env)
        else:
            return {
                self.observation_key: super().observe(agent),
                self.restriction_key: self.restrictions[agent],
            }
