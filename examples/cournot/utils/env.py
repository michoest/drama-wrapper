import gymnasium
import numpy as np
import gymnasium.spaces

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector


class NFGEnvironment(AECEnv):
    """
    The metadata holds environment constants. From gymnasium, we inherit the
    "render_modes" metadata which specifies which modes can be put into the render()
    method. At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {"render_modes": ["human"], "name": "nfg_v1"}

    def __init__(
        self,
        observation_spaces,
        action_spaces,
        utilities,
        number_of_steps,
        render_mode=None,
        action_mapping=None,
    ):
        super().__init__()
        self.possible_agents = list(utilities)
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces
        self.utilities = utilities

        self.number_of_steps = number_of_steps

        self.render_mode = render_mode
        self.action_mapping = action_mapping or (lambda action: action)

        self._state = None

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if len(self.agents) == len(self.possible_agents):
            print(
                ", ".join(
                    [
                        f"{agent}: {self.action_mapping(self._state[agent])}"
                        for agent in self.agents
                    ]
                )
            )
        else:
            print("Game over")

    def observe(self, agent):
        return np.array(
            [self._state[_agent] for _agent in self.agents if _agent != agent]
        )

    def close(self):
        pass

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._state = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.num_moves = 0

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        agent = self.agent_selection

        if self._agent_selector.is_first():
            self._state = {agent: None for agent in self.agents}
            self._clear_rewards()

        self._cumulative_rewards[agent] = 0
        self._state[self.agent_selection] = action

        if self._agent_selector.is_last():
            for agent in self.agents:
                self.rewards[agent] = self.utilities[agent](self._state)

            self.num_moves += 1
            self.truncations = {
                agent: self.num_moves >= self.number_of_steps for agent in self.agents
            }

        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def state(self):
        return self._state
