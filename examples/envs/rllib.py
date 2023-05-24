import random
from decimal import Decimal, getcontext
from typing import Optional, Any

import gymnasium
from gymnasium.spaces import Dict, Box, Discrete
from ray import rllib
from ray.rllib.utils import try_import_torch
import numpy as np
from shapely import Point, Polygon

getcontext().prec = 5

torch, nn = try_import_torch()


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

        self.observation_space = Box(low=0.0, high=self.number_of_actions - 1,
                                     shape=(self.number_of_agents,), dtype=np.float32)
        self.action_space = Discrete(self.number_of_actions)

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

        self.observation_space = Box(low=0.0, high=0.0, shape=(1,), dtype=np.float32)

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


class Agent:
    """ The agent representation

    Args:
        x: x-coordinate starting position
        y: y-coordinate starting position
        radius: Radius of the agent
        perspective: Starting perspective
        step_size: moving distance with each step
    """

    def __init__(self, x, y, radius, perspective, step_size):
        self.x = Decimal(repr(x))
        self.y = Decimal(repr(y))
        self.last_action = Decimal(0.0)
        self.radius = Decimal(repr(radius))
        self.perspective = Decimal(repr(perspective))
        self.step_size = Decimal(repr(step_size))
        self.collided = False
        self.distance_target = False
        self.distance_improvement = Decimal(0.0)

    def step(self, direction, dt):
        """ Take a step in a specific direction

        Args:
            direction: Angle in which the next step should be taken
            dt
        """
        self.x += Decimal(repr(np.cos(np.radians(float(direction))))) * self.step_size * dt
        self.y += Decimal(repr(np.sin(np.radians(float(direction))))) * self.step_size * dt
        self.perspective = direction

    def set_distance_target(self, new_distance):
        """ Sets the improvement and new distance to the target

        Args:
             new_distance: The new distance to the target
        """
        self.distance_improvement = self.distance_target - new_distance
        self.distance_target = new_distance

    def geometric_representation(self):
        """ Returns the shapely geometry representation of the agent

        Returns:
            shapely geometry object
        """
        return Point(float(self.x), float(self.y)).buffer(float(self.radius))


class Obstacle:
    """ The obstacle representation

    Args:
        coordinates: Polygon coordinates for the shape of the obstacle
    """

    def __init__(self, coordinates: list):
        self.coordinates = np.array([[
            Decimal(repr(coordinate[0])), Decimal(repr(coordinate[1]))
        ] for coordinate in coordinates])
        self.x, self.y = self.geometric_representation().centroid.coords[0]
        self.x = Decimal(repr(self.x))
        self.y = Decimal(repr(self.y))
        self.distance = Decimal(0.0)

    def geometric_representation(self):
        """ Returns the shapely geometry representation of the obstalce

        Returns:
            shapely geometry object
        """
        return Polygon(self.coordinates)

    def collision_area(self, radius):
        """ Returns the area which would lead to a collision when the agent enters it

        Args:
            radius: The radius of the agent

        Returns:
            shapely geometry object
        """
        return Polygon(self.coordinates).buffer(radius)

    def __repr__(self):
        return f'<{self.coordinates}>'


class NavigationEnvironment(gymnasium.Env):

    def __init__(self, env_config):
        assert 'STEPS_PER_EPISODE' in env_config
        assert 'ACTION_RANGE' in env_config
        assert 'DT' in env_config
        assert 'REWARD' in env_config
        assert 'REWARD_COEFFICIENT' in env_config['REWARD']
        assert 'TIMESTEP_PENALTY_COEFFICIENT' in env_config['REWARD']
        assert 'GOAL' in env_config['REWARD']
        assert 'COLLISION' in env_config['REWARD']
        assert 'MAP' in env_config
        assert 'HEIGHT' in env_config['MAP']
        assert 'WIDTH' in env_config['MAP']
        assert 'AGENT' in env_config['MAP']
        assert 'GOAL' in env_config['MAP']

        self.STEPS_PER_EPISODE = env_config['STEPS_PER_EPISODE']
        self.ACTION_RANGE = Decimal(repr(env_config["ACTION_RANGE"]))
        self.HEIGHT = env_config['MAP']['HEIGHT']
        self.WIDTH = env_config['MAP']['WIDTH']
        self.REWARD_COEFFICIENT = Decimal(repr(env_config["REWARD"]["REWARD_COEFFICIENT"]))
        self.REWARD_GOAL = Decimal(repr(env_config["REWARD"]["GOAL"]))
        self.REWARD_COLLISION = Decimal(repr(env_config["REWARD"]["COLLISION"]))
        self.TIMESTEP_PENALTY_COEFFICIENT = Decimal(repr(env_config['REWARD']['TIMESTEP_PENALTY_COEFFICIENT']))
        self.DT = Decimal(repr(env_config["DT"]))
        self.GOAL_RADIUS = env_config['MAP']['GOAL']['radius']
        self.AGENT_SETUP = {'x': env_config['MAP']['AGENT']['x'], 'y': env_config['MAP']['AGENT']['y'],
                            'radius': env_config['MAP']['AGENT']['radius'],
                            'perspective': env_config['MAP']['AGENT']['angle'],
                            'step_size': env_config['MAP']['AGENT']['step_size']}

        self.goal = Point(env_config['MAP']['GOAL']['x'], env_config['MAP']['GOAL']['y']).buffer(self.GOAL_RADIUS)
        self.agents = ['agent']  # Can be extended to support multiple agents
        self.agent = Agent(**self.AGENT_SETUP)
        self.map = Polygon([(0.0, 0.0), (self.WIDTH, 0.0), (self.WIDTH, self.HEIGHT),
                            (0.0, self.HEIGHT)])
        self.current_step = 0
        self.previous_position = [Decimal(0.0), Decimal(0.0)]
        self.last_reward = 0.0
        self.trajectory = []

        if 'RANDOM_SEED' in env_config:
            self.seed(env_config['RANDOM_SEED'])

        # Observation and Action Space
        self.observation_space = Dict({
            'location': Box(low=-2.0, high=np.max([self.WIDTH, self.HEIGHT]) + 2.0, shape=(2,),
                            dtype=np.float32),
            'perspective': Box(low=-1.0, high=360.0, shape=(1,), dtype=np.float32),
            'target_angle': Box(low=-1.0, high=360.0, shape=(1,), dtype=np.float32),
            'target_distance': Box(low=-1.0, high=np.sqrt(self.WIDTH ** 2 + self.HEIGHT ** 2),
                                   shape=(1,), dtype=np.float32),
            'current_step': Box(low=-1.0, high=self.STEPS_PER_EPISODE, shape=(1,), dtype=np.float32)
        })
        self.action_space = Box(low=float(-self.ACTION_RANGE / 2), high=float(self.ACTION_RANGE / 2), shape=(1,),
                                dtype=np.float32)

        super().__init__()

    def seed(self, seed: int = None):
        """ Set the seed of the environment

        Args:
            seed (int)
        """
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        self.action_space.seed(seed)

    def angle_to_target(self):
        """ Calculates the angle between the agent and the goal

        Returns:
            angle_to_target
        """
        if Decimal(repr(self.goal.centroid.coords[0][0])) == self.agent.x:
            angle_agent = Decimal(0.0)
        else:
            angle_agent = self.agent.perspective - Decimal(
                repr((
                    np.rad2deg(np.arctan(float(np.abs(
                        Decimal(repr(self.goal.centroid.coords[0][1])) - self.agent.y
                    ) / np.abs(Decimal(repr(self.goal.centroid.coords[0][0])) - self.agent.x)))))))

        return angle_agent if angle_agent >= Decimal(0.0) else Decimal(360.0) + angle_agent

    def distance_to_target(self):
        """ Calculates the distance between the agent and the goal

        Returns:
            distance_to_target
        """
        return Decimal(repr(self.goal.centroid.distance(self.agent.geometric_representation())))

    def get_reward(self):
        """ Calculates the reward based on collisions, improvement and the distance to the goal

        Returns:
            reward
        """
        if self.agent.collided:
            reward = self.REWARD_COLLISION
        elif self.agent.distance_target <= self.GOAL_RADIUS:
            reward = self.REWARD_GOAL
        else:
            reward = self.REWARD_COEFFICIENT * self.agent.distance_improvement - (
                    Decimal(repr(self.current_step)) * self.TIMESTEP_PENALTY_COEFFICIENT)
        return float(reward)

    def detect_collision(self):
        """ Checks if the agent collided with the border

        Returns:
            violation (bool)
        """
        # Check if agent is on the map and not collided with the boundaries
        if not self.map.contains(self.agent.geometric_representation()) or self.agent.radius - Decimal(
                repr(self.map.exterior.distance(Point(self.agent.x, self.agent.y)))) > Decimal(0.0):
            return True
        return False

    def step(self, action: dict):
        """ Perform an environment iteration including moving the agent and obstacles.

        Args:
            action (list): Angle of the agent's next step

        Returns:
            observation (dict)
        """
        action = Decimal(repr(action['agent'][0]))
        step_direction = self.agent.perspective + action

        if step_direction < Decimal(0.0):
            step_direction += Decimal(360.0)
        elif step_direction >= Decimal(360.0):
            step_direction -= Decimal(360.0)

        self.agent.step(step_direction, self.DT)
        self.agent.last_action = action
        self.agent.collided = self.detect_collision()
        self.agent.set_distance_target(self.distance_to_target())
        self.last_reward = self.get_reward()
        self.trajectory.append([float(self.agent.x),
                                float(self.agent.y)])
        self.current_step += 1

        observation = {'agent': self._get_agent_observation()}
        done = self.agent.collided or (self.agent.distance_target <= self.GOAL_RADIUS)
        truncated = self.current_step >= self.STEPS_PER_EPISODE

        return observation, {'agent': self.last_reward}, {'__all__': done}, {'__all__': truncated}, {}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """ Resets and loads the structure of the map again

        Returns:
            observation (dict)
        """
        self.trajectory.append([float(self.agent.x),
                                float(self.agent.y)])
        self.agent = Agent(**self.AGENT_SETUP)
        self.current_step = 0


    def _get_agent_observation(self):
        return {'location': np.array([self.agent.x, self.agent.y], dtype=np.float32),
                'perspective': np.array([self.agent.perspective], dtype=np.float32),
                'target_angle': np.array([self.angle_to_target()], dtype=np.float32),
                'target_distance': np.array([self.agent.distance_target], dtype=np.float32),
                'current_step': np.array([self.current_step], dtype=np.float32)}
