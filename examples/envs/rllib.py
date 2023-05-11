from typing import Optional

from ray import rllib
import gymnasium as gym
import numpy as np


class DummyEnvironment(rllib.env.MultiAgentEnv):
    def __init__(self, env_config=None):
        super().__init__()

        if env_config is None:
            env_config = {}

        self.agents = env_config.get('agents', ['0', '1'])
        self.number_of_steps = env_config.get('number_of_steps', 5)
        self.observation_space = env_config.get('observation_space',
                                                gym.spaces.Box(low=0.0, high=1.0, shape=(len(self.agents),)))
        self.action_space = env_config.get('action_space', gym.spaces.Discrete(3))
        self.current_step = None

    def step(self, actions):
        self.current_step += 1

        observations = {agent_id: self.observation_space.sample() for agent_id in self.agents}
        rewards = {agent_id: np.random.random() for agent_id in self.agents}

        return observations, rewards, {'__all__': self.current_step >= self.number_of_steps}, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.current_step = 0

        return {agent_id: self.observation_space.sample() for agent_id in self.agents}


class MatrixGameEnvironment(rllib.env.MultiAgentEnv):
    """Represents a matrix game played by multiple agents. The payoff matrix
    determines the number of players and actions. Usually, an episode only
    consists of a single step.
    """

    def __init__(self, env_config=None):
        assert 'payoffs' in env_config
        super().__init__()

        if env_config is None:
            env_config = {}

        self.payoffs = env_config['payoffs']
        (self.number_of_actions, *_, self.number_of_agents) = self.payoffs.shape

        self.episode_length = env_config.get('episode_length', 1)

        self.agents = env_config.get('agents', [str(index) for index in range(self.number_of_agents)])

        self.observation_space = gym.spaces.Box(low=0.0, high=self.number_of_actions - 1,
                                                shape=(self.number_of_agents,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self.number_of_actions)

        self.state = None
        self.current_step = None

    def step(self, actions):
        actions = tuple(actions.values())
        rewards = {agent_id: self.payoffs[actions][index] for index, agent_id in enumerate(self.agents)}
        self.state = np.array(actions)
        self.current_step += 1

        return {agent_id: self._get_observation(agent_id) for agent_id in self.agents}, rewards, {
            '__all__': self.current_step >= self.episode_length}, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.state = np.random.randint(0, self.number_of_actions, size=self.number_of_agents)
        self.current_step = 0

        return {agent_id: self._get_observation(agent_id) for agent_id in self.agents}

    def _get_observation(self, agent_id):
        return self.state


class NormalGameEnvironment(rllib.env.MultiAgentEnv):
    """Represents a normal-form game played by multiple agents, where all agents
    have the same action space. There are arbitrary utility functions for the
    agents. Usually, an episode only consists of a single step.
    """

    def __init__(self, env_config=None):
        assert 'utilities' in env_config
        super().__init__()

        if env_config is None:
            env_config = {}

        self.utilities = env_config['utilities']
        self.agents = self.utilities.keys()
        self.number_of_agents = len(self.agents)

        assert 'action_space' in env_config
        self.action_space = env_config['action_space']

        self.episode_length = env_config.get('episode_length', 1)

        self.observation_space = gym.spaces.Box(low=0.0, high=0.0, shape=(1,), dtype=np.float32)

        self.state = None
        self.current_step = None

    def step(self, actions):
        actions = actions.values()
        rewards = {agent_id: self.utilities[id](*actions) for agent_id in self.agents}
        self.current_step += 1

        return {agent_id: self._get_observation(agent_id) for agent_id in self.agents}, rewards, {
            '__all__': self.current_step >= self.episode_length}, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.state = np.array([0.0])
        self.current_step = 0

        return {agent_id: self._get_observation(agent_id) for agent_id in self.agents}

    def _get_observation(self, agent_id):
        return self.state
