import functools
from copy import copy
from decimal import Decimal

import numpy as np
from gymnasium.spaces import Box
from pettingzoo.utils import agent_selector
from shapely import Point, Polygon
from pettingzoo.utils.env import AECEnv


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


class NavigationEnvironment(AECEnv):

    metadata = {
        'name': 'simple_navigation'
    }

    def __init__(self, env_config=None, render_mode=None):
        if env_config is None:
            env_config = {}

        self.STEPS_PER_EPISODE = env_config.get('STEPS_PER_EPISODE', 40)
        self.ACTION_RANGE = Decimal(repr(env_config.get("ACTION_RANGE", 220.0)))
        self.HEIGHT = env_config.get('HEIGHT', 15.0)
        self.WIDTH = env_config.get('WIDTH', 15.0)
        self.REWARD_COEFFICIENT = Decimal(repr(env_config.get("REWARD_COEFFICIENT", 1.0)))
        self.REWARD_GOAL = Decimal(repr(env_config.get("REWARD_GOAL", 50.0)))
        self.REWARD_COLLISION = Decimal(repr(env_config.get("REWARD_COLLISION", -5.0)))
        self.TIMESTEP_PENALTY_COEFFICIENT = Decimal(repr(env_config.get('TIMESTEP_PENALTY_COEFFICIENT', 0.05)))
        self.DT = Decimal(repr(env_config.get("DT", 1.0)))
        self.GOAL_RADIUS = env_config.get('GOAL_RADIUS', 1.0)
        self.AGENT_SETUP = {'x': env_config.get('AGENT_X', 1.0), 'y': env_config.get('AGENT_Y', 1.0),
                            'radius': env_config.get('AGENT_RADIUS', 1.0),
                            'perspective': env_config.get('AGENT_PERSPECTIVE', 90.0),
                            'step_size': env_config.get('AGENT_STEP_SIZE', 1.0)}

        self.goal = Point(env_config.get('GOAL_X', 12.0), env_config.get('GOAL_Y', 12.0)).buffer(self.GOAL_RADIUS)
        self.agent = Agent(**self.AGENT_SETUP)
        self.map = Polygon([(0.0, 0.0), (self.WIDTH, 0.0), (self.WIDTH, self.HEIGHT),
                            (0.0, self.HEIGHT)])
        self.current_step = 0
        self.previous_position = [Decimal(0.0), Decimal(0.0)]
        self.trajectory = []

        self.possible_agents = ['agent_0']
        self._agent_selector = None
        self.render_mode = render_mode

        super(NavigationEnvironment, self).__init__()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(low=0.0, high=np.concatenate([np.full((2,), np.max([self.WIDTH, self.HEIGHT])),
                                                 np.full((2,), 360.0),
                                                 np.array([np.sqrt(self.WIDTH ** 2 + self.HEIGHT ** 2),
                                                           self.STEPS_PER_EPISODE])]), shape=(6,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Box(low=float(-self.ACTION_RANGE / 2), high=float(self.ACTION_RANGE / 2), shape=(1,),
                   dtype=np.float32)

    def state(self):
        return np.array([float(self.agent.x), float(self.agent.y), float(self.agent.perspective)], dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.trajectory.append([float(self.agent.x),
                                float(self.agent.y)])
        self.agent = Agent(**self.AGENT_SETUP)
        self.agent.distance_target = self.distance_to_target()
        self.current_step = 0

        self.agents = copy(self.possible_agents)
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

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

    def step(self, action):
        """ Perform an environment iteration including moving the agent and obstacles.

        Args:
            action: Angle of the agent's next step

        Returns:
            observation (dict)
        """
        if (
                self.terminations[self.agent_selection] or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0

        action = Decimal(repr(action[0]))
        step_direction = self.agent.perspective + action

        if step_direction < Decimal(0.0):
            step_direction += Decimal(360.0)
        elif step_direction >= Decimal(360.0):
            step_direction -= Decimal(360.0)

        self.agent.step(step_direction, self.DT)
        self.agent.last_action = action
        self.agent.collided = self.detect_collision()
        self.agent.set_distance_target(self.distance_to_target())
        self.rewards = {'agent_0': self.get_reward()}
        self.infos = {'agent_0': {'solved': self.agent.distance_target <= self.GOAL_RADIUS}}
        self.trajectory.append([float(self.agent.x),
                                float(self.agent.y)])
        self.current_step += 1

        self.terminations = {agent: self.agent.collided or (self.agent.distance_target <= self.GOAL_RADIUS)}
        self.truncations = {agent: False}
        if self.current_step >= self.STEPS_PER_EPISODE:
            self.truncations = {agent: True}

        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

    def observe(self, agent):
        return np.array([self.agent.x, self.agent.y, self.agent.perspective, self.angle_to_target(),
                         self.agent.distance_target, self.current_step], dtype=np.float32)

    def render(self):
        print(self.agent.x, self.agent.y)

    def close(self):
        pass
