import functools

from gymnasium.spaces import Dict
from pettingzoo.utils import BaseWrapper


class RestrictionWrapper(BaseWrapper):
    """Wrapper to extend the environment with a restrictor agent.

    Extended Agent-Environment Cycle:
        Reset() -> Restrictor
        Step() -> Agent
        Step() -> Restrictor
        ...
    """

    def __init__(
        self,
        env,
        restrictor_observation_space=None,
        restrictor_action_space=None,
        *,
        restrictor_reward_fn=None,
        preprocess_restrictor_observation_fn=None,
        restrictor_key="restrictor_0",
        restriction_key="restriction",
        observation_key="observation"
    ):
        super().__init__(env)

        # TODO: Decide how to handle observation and action space of the restrictor
        self.restrictor_observation_space = restrictor_observation_space
        self.restrictor_action_space = restrictor_action_space

        self.restrictor_reward_fn = restrictor_reward_fn or (
            lambda env, rewards: sum(rewards.values())
        )
        self.preprocess_restrictor_observation_fn = (
            preprocess_restrictor_observation_fn or (lambda env: env.state())
        )

        self.restrictor_key = restrictor_key
        self.restriction_key = restriction_key
        self.observation_key = observation_key

        # self.restrictions is a dictionary which keeps the latest value for each agent
        self.restrictions = None

        self.possible_agents = self.possible_agents + [self.restrictor_key]

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        if agent == self.restrictor_key:
            return self.restrictor_observation_space
        else:
            return Dict(
                {
                    self.observation_key: self.env.observation_space(agent),
                    self.restriction_key: self.restrictor_action_space,
                }
            )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        if agent == self.restrictor_key:
            return self.restrictor_action_space
        else:
            return self.env.action_space(agent)

    def reset(self, seed=None, options=None):
        self.env.reset(seed, options)

        # Set properties
        self.rewards = {**self.env.rewards, self.restrictor_key: 0.0}
        self.terminations = {**self.env.terminations, self.restrictor_key: False}
        self.truncations = {**self.env.truncations, self.restrictor_key: False}
        self.infos = {**self.env.infos, self.restrictor_key: {}}
        self.agents = self.env.agents + [self.restrictor_key]
        self._cumulative_rewards = {
            **self.env._cumulative_rewards,
            self.restrictor_key: 0.0,
        }

        self.restrictions = {agent: None for agent in self.env.agents}

        # Start an episode with the restrictor to obtain restrictions
        self.agent_selection = self.restrictor_key

    def step(self, action):
        if self.agent_selection == self.restrictor_key:
            # If the action was taken by the restrictor, check if it was terminated
            # last step
            if self.terminations[self.agent_selection]:
                self._was_dead_step(action)
                self.agent_selection = self.env.agent_selection
                return

            # Otherwise set the restrictions that apply to the next agent.
            self.restrictions[self.env.agent_selection] = action

            # Switch to the next agent of the original environment
            self.agent_selection = self.env.agent_selection
        else:
            # If the action was taken by an agent, execute it in the original
            # environment
            self.env.step(action)

            # Update properties
            self.agents = [self.restrictor_key] + self.env.agents
            self.rewards = {
                **self.env.rewards,
                self.restrictor_key: self.restrictor_reward_fn(self.env, self.rewards),
            }
            self.terminations = {
                **self.env.terminations,
                self.restrictor_key: all(
                    self.env.terminations[agent] or self.env.truncations[agent]
                    for agent in self.env.agents
                ),
            }
            self.truncations = {**self.env.truncations, self.restrictor_key: False}
            self.infos = {**self.env.infos, self.restrictor_key: {}}
            self._cumulative_rewards = {
                **self.env._cumulative_rewards,
                self.restrictor_key: self._cumulative_rewards[self.restrictor_key]
                + self.rewards[self.restrictor_key],
            }

            if self.env.agents and all(
                self.env.terminations[agent] or self.env.truncations[agent]
                for agent in self.env.agents
            ):
                # If there are alive agents left, get the next restriction
                self.agent_selection = self.env.agent_selection
            else:
                # Otherwise, get the next restriction
                self.agent_selection = self.restrictor_key

    def observe(self, agent: str):
        if agent == self.restrictor_key:
            return self.preprocess_restrictor_observation_fn(self.env)
        else:
            return {
                self.observation_key: super().observe(agent),
                self.restriction_key: self.restrictions[agent],
            }
