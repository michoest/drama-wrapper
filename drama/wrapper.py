# Typing
from typing import Union, Callable, Optional

# Standard modules
import functools

# External modules
from gymnasium.spaces import Dict
from pettingzoo import AECEnv
from pettingzoo.utils import BaseWrapper

# Internal modules
from drama.restrictions import Restriction
from drama.restrictors import Restrictor
from drama.utils import flatten


# If no functions are provided for some or all restrictors, use these defaults
def _default_restrictor_reward_fn(env, rewards):
    return sum(rewards.values())


def _default_preprocess_restrictor_observation_fn(env):
    return env.state()


def _default_postprocess_restriction_fn(restriction):
    return restriction


def _default_restriction_violation_fn(env, action, restriction: Restriction):
    env.step(restriction.sample())


class RestrictionWrapper(BaseWrapper):
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
        env: AECEnv,
        restrictors: Union[dict, Restrictor],
        *,
        agent_restrictor_mapping: Optional[dict] = None,
        restrictor_reward_fns: Union[dict, Callable, None] = None,
        preprocess_restrictor_observation_fns: Union[dict, Callable] = None,
        postprocess_restriction_fns: Union[dict, Callable] = None,
        restriction_violation_fns: Union[dict, Callable] = None,
        restriction_key: str = "restriction",
        observation_key: str = "observation",
        return_object: bool = False,
        **kwargs,
    ):
        super().__init__(env)

        if isinstance(restrictors, dict):
            assert agent_restrictor_mapping, "Agent-restrictor mapping required!"

        self.restrictors = (
            restrictors
            if isinstance(restrictors, dict)
            else {"restrictor_0": restrictors}  # Naming convention from PettingZoo
        )
        self.agent_restrictor_mapping = (
            agent_restrictor_mapping
            if isinstance(restrictors, dict)
            else {agent: "restrictor_0" for agent in self.env.possible_agents}
        )

        self.restrictor_reward_fns = (
            {
                restrictor: restrictor_reward_fns[restrictor]
                if restrictor_reward_fns and restrictor_reward_fns.get(restrictor, None)
                else _default_restrictor_reward_fn
                for restrictor in self.restrictors
            }
            if isinstance(restrictor_reward_fns, Union[dict, None])
            else {restrictor: restrictor_reward_fns for restrictor in self.restrictors}
        )

        # Set restrictor observation preprocessing functions
        if isinstance(preprocess_restrictor_observation_fns, Callable):
            self.preprocess_restrictor_observation_fns = {
                restrictor: preprocess_restrictor_observation_fns
                for restrictor in self.restrictors
            }
        else:
            self.preprocess_restrictor_observation_fns = {
                restrictor: _default_preprocess_restrictor_observation_fn
                for restrictor in self.restrictors
            }
            for name, restrictor in self.restrictors.items():
                if isinstance(
                    preprocess_restrictor_observation_fns, dict
                ) and preprocess_restrictor_observation_fns.get(name, None):
                    self.preprocess_restrictor_observation_fns[
                        name
                    ] = preprocess_restrictor_observation_fns[name]
                elif hasattr(restrictor, "preprocess_observation"):
                    self.preprocess_restrictor_observation_fns[
                        name
                    ] = restrictor.preprocess_observation

        # Set restriction postprocessing functions
        if isinstance(postprocess_restriction_fns, Callable):
            self.postprocess_restriction_fns = {
                restrictor: postprocess_restriction_fns
                for restrictor in self.restrictors
            }
        else:
            self.postprocess_restriction_fns = {
                restrictor: _default_postprocess_restriction_fn
                for restrictor in self.restrictors
            }
            for name, restrictor in self.restrictors.items():
                if isinstance(
                    postprocess_restriction_fns, dict
                ) and postprocess_restriction_fns.get(name, None):
                    self.postprocess_restriction_fns[
                        name
                    ] = postprocess_restriction_fns[name]
                elif hasattr(restrictor, "postprocess_restriction"):
                    self.postprocess_restriction_fns[
                        name
                    ] = restrictor.postprocess_restriction

        self.restriction_violation_fns = (
            {
                agent: restriction_violation_fns[agent]
                if restriction_violation_fns
                and restriction_violation_fns.get(agent, None)
                else _default_restriction_violation_fn
                for agent in self.env.possible_agents
            }
            if isinstance(restriction_violation_fns, Union[dict, None])
            else {
                agent: restriction_violation_fns for agent in self.env.possible_agents
            }
        )

        self.restriction_key = restriction_key
        self.observation_key = observation_key
        self.return_object = return_object
        self.kwargs = {**kwargs}

        # self.restrictions is a dictionary which keeps the latest value for each agent
        self.restrictions = None

        self.possible_agents = self.possible_agents + list(self.restrictors)

        # Check if restrictor action spaces (after post-processing) match
        # agent action spaces
        for agent in self.env.possible_agents:
            restrictor = self.restrictors[self.agent_restrictor_mapping[agent]]
            sample_restriction = self.postprocess_restriction_fns[
                self.agent_restrictor_mapping[agent]
            ](restrictor.action_space.sample())

            assert sample_restriction.base_space == env.action_space(
                agent
            ), f"The action spaces of {self.agent_restrictor_mapping[agent]} and {agent} are not compatible!"

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

            # Reset cumulative reward for the current restrictor
            # self._cumulative_rewards[self.agent_selection] = 0

            # Otherwise set the restrictions that apply to the next agent.
            assert (
                self.agent_restrictor_mapping[self.env.agent_selection]
                == self.agent_selection
            )
            # self.restrictions[self.env.agent_selection] = action
            self.restrictions[
                self.env.agent_selection
            ] = self.postprocess_restriction_fns[self.agent_selection](action)

            # Switch to the next agent of the original environment
            self.agent_selection = self.env.agent_selection
        else:
            # Check if the action violated the current restriction for the agent
            if action and not self.restrictions[self.agent_selection].contains(action):
                self.restriction_violation_fns[self.agent_selection](
                    self.env, action, self.restrictions[self.agent_selection]
                )
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
                        self.env, self.env.rewards
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
                    restrictor: self.restrictor_reward_fns[restrictor](
                        self.env, self.env._cumulative_rewards
                    )
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

    def observe(self, agent: str, return_object: bool = None, **kwargs):
        if agent in self.restrictors:
            return self.preprocess_restrictor_observation_fns[agent](self.env)
        else:
            return_object = (
                return_object if return_object is not None else self.return_object
            )
            return {
                self.observation_key: super().observe(agent),
                self.restriction_key: self.restrictions[agent]
                if return_object and self.restrictions[agent].is_np_flattenable
                else flatten(
                    self.restrictors[self.agent_restrictor_mapping[agent]].action_space,
                    self.restrictions[agent],
                    **{**self.kwargs, **kwargs},
                ),
            }
