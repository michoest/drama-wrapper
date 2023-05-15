import inspect
from enum import Enum
from typing import Optional

from ray import rllib
import gymnasium as gym


class NextAction(Enum):
    GOVERNANCE = 0
    AGENTS = 1


class UniformlyRestrictedEnvironment(rllib.env.MultiAgentEnv):
    def __init__(self, env_config=None):
        if env_config is None:
            env_config = {}

        assert 'env' in env_config, 'You have to provide an environment!'
        assert 'governance_action_space' in env_config, 'You must specify the restrictors action space'
        super().__init__()

        env = env_config['env']

        if inspect.isclass(env):
            self.env = env(env_config=env_config.get('env_config', {}))
        else:
            self.env = env

        self.next_action = None
        self.observations = None
        self.rewards = None

        self.governance_observation = None

        self.governance_reward_fn = env_config.get('governance_reward_fn', None)
        self.governance_observation_fn = env_config.get('governance_observation_fn', None)

        self.observation_space = gym.spaces.Dict({
            'observation': self.env.observation_space,
            'allowed_actions': env_config.get('governance_action_space')
        })
        self.action_space = self.env.action_space

        self.governance_observation_space = env_config.get('governance_observation_space',
                                                           gym.spaces.Discrete(1))
        self.governance_action_space = env_config.get('governance_action_space')

    def step(self, actions):
        if self.next_action == NextAction.GOVERNANCE:
            allowed_actions = actions['gov']
            self.next_action = NextAction.AGENTS

            observations = {agent_id: {
                'observation': self.observations[agent_id],
                'allowed_actions': allowed_actions
            } for agent_id in self.env.agents}

            return observations, self.rewards, {'__all__': False}, {}
        else:
            self.observations, self.rewards, dones, info = self.env.step(actions)
            self.next_action = NextAction.GOVERNANCE

            return {'gov': self._get_governance_observation()}, {'gov': self._get_governance_reward()}, dones, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.observations = self.env.reset()
        self.rewards = {}
        self.next_action = NextAction.GOVERNANCE

        return {'gov': self._get_governance_observation()}

    def _get_governance_observation(self):
        return self.governance_observation_fn(self) if self.governance_observation_fn else self.observations

    def _get_governance_reward(self):
        return self.governance_reward_fn(self) if \
            self.governance_reward_fn else sum(self.rewards.values())


class IndividuallyRestrictedEnvironment(rllib.env.MultiAgentEnv):
    def __init__(self, env_config=None):
        if env_config is None:
            env_config = {}

        assert 'env' in env_config, 'You have to provide an environment!'
        super().__init__()

        env = env_config['env']

        if inspect.isclass(env):
            self.env = env(env_config=env_config.get('env_config', {}))
        else:
            self.env = env

        self.next_action = None
        self.observations = None
        self.rewards = None

        self.governance_observation = None

        self.governance_reward_fn = env_config.get('governance_reward_fn', None)
        self.governance_observation_fn = env_config.get('governance_observation_fn', None)

        assert isinstance(self.env.action_space, gym.spaces.Discrete), \
            'UniformDiscreteGovernanceWrapper can only wrap environments with a discrete action space!'
        self.observation_space = gym.spaces.Dict({
            'observation': self.env.observation_space,
            'allowed_actions': env_config.get('governance_action_space',
                                              gym.spaces.MultiBinary(self.env.action_space.n))
        })
        self.action_space = self.env.action_space

        self.governance_observation_space = env_config.get('governance_observation_space',
                                                           gym.spaces.Discrete(1))
        self.governance_action_space = env_config.get('governance_action_space',
                                                      gym.spaces.MultiBinary(self.env.action_space.n))

    def step(self, actions):
        if self.next_action == NextAction.GOVERNANCE:
            # Governance has just acted
            allowed_actions = actions['gov']
            self.next_action = NextAction.AGENTS

            observations = {agent_id: {
                'observation': self.observations[agent_id],
                'allowed_actions': allowed_actions[agent_id]
            } for agent_id in self.env.agents}

            return observations, self.rewards, {'__all__': False}, {}
        else:
            self.observations, self.rewards, dones, info = self.env.step(actions)
            self.next_action = NextAction.GOVERNANCE

            return {'gov': self._get_governance_observation()}, {'gov': self._get_governance_reward()}, dones, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.observations = self.env.reset()
        self.rewards = {}
        self.next_action = NextAction.GOVERNANCE

        return {'gov': self._get_governance_observation()}

    def _get_governance_observation(self):
        return self.governance_observation_fn(self) if self.governance_observation_fn else self.observations

    def _get_governance_reward(self):
        return self.governance_reward_fn(self._get_governance_observation()) if \
            self.governance_reward_fn else sum(self.rewards.values())
