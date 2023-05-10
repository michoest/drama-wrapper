import inspect
from enum import Enum

from ray import rllib
import gymnasium as gym


class NextAction(Enum):
    GOVERNANCE = 0
    AGENTS = 1


class UniformlyRestrictedDiscreteEnvironment(rllib.env.MultiAgentEnv):
    def __init__(self, env_config={}):
        assert 'env' in env_config, 'You have to provide an environment!'
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

        assert isinstance(self.env.action_space, gym.spaces.Discrete), \
            'UniformDiscreteGovernanceWrapper can only wrap environments with a discrete action space!'
        self.observation_space = gym.spaces.Dict({
            'observation': self.env.observation_space,
            'allowed_actions': gym.spaces.MultiBinary(self.env.action_space.n)
        })
        self.action_space = self.env.action_space

        self.governance_observation_space = gym.spaces.Discrete(1)
        self.governance_action_space = gym.spaces.MultiBinary(self.env.action_space.n)

    def step(self, actions):
        if self.next_action == NextAction.GOVERNANCE:
            # Governance has just acted
            allowed_actions = actions['gov']
            self.next_action = NextAction.AGENTS

            observations = {id: {
                'observation': self.observations[id],
                'allowed_actions': allowed_actions
            } for id in self.env.agents}

            return observations, self.rewards, {'__all__': False}, {}
        else:
            # Agents have just acted
            self.observations, self.rewards, dones, info = self.env.step(actions)
            self.next_action = NextAction.GOVERNANCE

            return {'gov': self._get_governance_observation()}, {'gov': self._get_governance_reward()}, dones, info

    def reset(self):
        self.observations = self.env.reset()
        self.rewards = {}
        self.next_action = NextAction.GOVERNANCE

        return {'gov': self._get_governance_observation()}

    def _get_governance_observation(self):
        return self.governance_observation_space.sample()

    def _get_governance_reward(self):
        return self.governance_reward_fn(self._get_governance_observation()) if \
            self.governance_reward_fn else sum(self.rewards.values())


class UniformlyRestrictedContinuousEnvironment(rllib.env.MultiAgentEnv):
    def __init__(self, env_config={}):
        pass

    def step(self, actions):
        pass

    def reset(self):
        pass


class IndividuallyRestrictedDiscreteEnvironment(rllib.env.MultiAgentEnv):
    def __init__(self, env_config={}):
        assert 'env' in env_config, 'You have to provide an environment!'
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

        assert isinstance(self.env.action_space, gym.spaces.Discrete), \
            'UniformDiscreteGovernanceWrapper can only wrap environments with a discrete action space!'
        self.observation_space = gym.spaces.Dict({
            'observation': self.env.observation_space,
            'allowed_actions': gym.spaces.MultiBinary(self.env.action_space.n)
        })
        self.action_space = self.env.action_space

        self.governance_observation_space = gym.spaces.Discrete(1)
        self.governance_action_space = gym.spaces.MultiBinary(self.env.action_space.n)

    def step(self, actions):
        if self.next_action == NextAction.GOVERNANCE:
            # Governance has just acted
            allowed_actions = actions['gov']
            self.next_action = NextAction.AGENTS

            observations = {id: {
                'observation': self.observations[id],
                'allowed_actions': allowed_actions[id]
            } for id in self.env.agents}

            return observations, self.rewards, {'__all__': False}, {}
        else:
            # Agents have just acted
            self.observations, self.rewards, dones, info = self.env.step(actions)
            self.next_action = NextAction.GOVERNANCE

            return {'gov': self._get_governance_observation()}, {'gov': self._get_governance_reward()}, dones, info

    def reset(self):
        self.observations = self.env.reset()
        self.rewards = {}
        self.next_action = NextAction.GOVERNANCE

        return {'gov': self._get_governance_observation()}

    def _get_governance_observation(self):
        return self.governance_observation_space.sample()

    def _get_governance_reward(self):
        return self.governance_reward_fn(self._get_governance_observation()) if \
            self.governance_reward_fn else sum(self.rewards.values())


class IndividuallyRestrictedContinuousEnvironment(rllib.env.MultiAgentEnv):
    def __init__(self, env_config={}):
        pass

    def step(self, actions):
        pass

    def reset(self):
        pass
