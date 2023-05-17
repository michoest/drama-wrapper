from decimal import Decimal

import numpy as np
from ray.rllib import Policy
from shapely import Polygon, Point

from examples.envs.rllib import Obstacle, Agent
from examples.restrictors.helpers import project_intervals_into_action_space, inverse_space, \
    get_restrictions_for_polygon, midpoint

MULTI_GEOM_TYPES = ['MultiPolygon', 'MultiLineString', 'GeometryCollection', 'MultiPoint']
NO_EXTERIOR_TYPES = ['Point', 'LineString']

SHAPE_COLLECTION = [
    # Rectangle
    np.array([[0, 0], [0, 1], [1, 1], [1, 0]]),
    # Trapeze
    np.array([[0.0, 0.0], [0.33, 1.0], [0.66, 1.0], [1.0, 0.0]]),
    # Triangle
    np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]]),
    # Octagon
    np.array([[0.0, 0.66], [0.33, 1.0], [0.66, 1.0], [1.0, 0.66], [1.0, 0.33], [0.66, 0.0], [0.33, 0.0], [0.0, 0.33]])
]


class NavigationRestrictor(Policy):

    def __init__(self, observation_space, action_space, governance_config):
        if governance_config is None:
            governance_config = {}
        super().__init__(observation_space=observation_space, action_space=action_space,
                         config=governance_config)

        assert 'COUNT' in governance_config
        assert 'POSITION_COVARIANCE' in governance_config
        assert 'MEAN_SIZE' in governance_config
        assert 'VARIANCE_SIZE' in governance_config
        assert 'RANGE_SIZE' in governance_config
        assert 'START_SEED' in governance_config
        assert 'SAFETY_ANGLE' in governance_config

        self.COUNT = governance_config['COUNT']
        self.POSITION_COVARIANCE = governance_config['POSITION_COVARIANCE']
        self.MEAN_SIZE = governance_config['MEAN_SIZE']
        self.VARIANCE_SIZE = governance_config['VARIANCE_SIZE']
        self.RANGE_SIZE = governance_config['RANGE_SIZE']
        self.START_SEED = governance_config['START_SEED']
        self.SAFETY_ANGLE = governance_config['SAFETY_ANGLE']

        self.obstacles = []
        self.map_collision_area = None
        self.seed = 0

    def get_initial_state(self):
        return [[-180.0, 180.0]]

    def is_recurrent(self) -> bool:
        return True

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        actions = []
        for obs in obs_batch:
            if obs['step'] == 0:
                self.generate_obstacles(obs['map'][0], obs['map'][1])
                self.map_collision_area = Polygon([(0.0, 0.0), (obs['map'][1], 0.0), (obs['map'][1], obs['map'][0]),
                                                   (0.0, obs['map'][0])]).exterior.buffer(obs['step_radius'][0])

            agent = Agent(x=obs['location'][0], y=obs['location'][1], perspective=obs['perspective'][0],
                          radius=obs['step_radius'][0], step_size=obs['step_size'][0])
            step_circle = Point(obs['location'][0], obs['location'][1]).buffer(float(agent.step_size) * obs['dt'][0])
            restrictions = []

            for obstacle in self.obstacles + [self.map_collision_area]:
                if isinstance(obstacle, Obstacle):
                    obstacle = obstacle.collision_area(float(agent.radius))

                is_in_collision_area = obstacle.contains(
                    Point(float(agent.x), float(agent.y))) or obstacle.boundary.contains(
                    Point(float(agent.x), float(agent.y)))

                obstacle_step_circle_intersection = step_circle.intersection(obstacle) if not is_in_collision_area else (
                    step_circle.boundary.difference(obstacle))

                # If intersection consists of multiple parts, iterate through them
                if obstacle_step_circle_intersection.geom_type in MULTI_GEOM_TYPES:
                    restrictions_for_part = []

                    for polygon in obstacle_step_circle_intersection.geoms:
                        restriction = get_restrictions_for_polygon(
                            polygon.exterior.coords if not is_in_collision_area and not (
                                    polygon.geom_type in NO_EXTERIOR_TYPES) else polygon.coords, agent)

                        restrictions_for_part.append(restriction)

                    # Bring each restriction into the action space
                    restrictions_for_part = project_intervals_into_action_space(restrictions_for_part,
                                                                                low=Decimal(-180), high=Decimal(180))
                    for restriction in restrictions_for_part:
                        if restriction[0] < Decimal(-180.0):
                            restrictions_for_part.append([Decimal(-180.0), restriction[1]])
                            restriction[0] = Decimal(360) + restriction[0]
                            restriction[1] = Decimal(180)

                    # Merge overlapping restrictions for different parts
                    if len(restrictions_for_part) > 1:
                        for index, restriction in enumerate(restrictions_for_part):
                            if index != (len(restrictions_for_part) - 1):
                                if restriction[1] == restrictions_for_part[index + 1][0]:
                                    restrictions_for_part[index + 1][0] = restriction[0]
                                    restriction[0] = Decimal(np.inf)
                        restrictions_for_part = [res for res in restrictions_for_part if res[0] != Decimal(np.inf)]

                        # When agent is inside the collision area, inverse the space to get restrictions
                        if is_in_collision_area:
                            restrictions_for_part = inverse_space(restrictions_for_part,
                                                                  low=Decimal(-180.0), high=Decimal(180.0))
                    else:
                        restrictions_for_part = [np.flip(restrictions_for_part[0])
                                                 ] if is_in_collision_area else restrictions_for_part

                    restrictions += restrictions_for_part
                else:
                    object_restrictions = get_restrictions_for_polygon(
                        obstacle_step_circle_intersection.exterior.coords if not is_in_collision_area and not (
                                obstacle_step_circle_intersection.geom_type in NO_EXTERIOR_TYPES
                        ) else obstacle_step_circle_intersection.coords, agent)
                    restrictions.append(np.flip(object_restrictions) if is_in_collision_area else object_restrictions)
                    restrictions = project_intervals_into_action_space(restrictions,
                                                                       low=Decimal(-180.0), high=Decimal(180.0))

            restrictions = [restriction for restriction in restrictions if restriction[0] != restriction[1]]

            # Build allowed action space from restrictions
            allowed_action_space = [[-obs['action_range'][0] / 2, obs['action_range'][0] / 2]]
            for restriction in restrictions:
                for index, allowed_subset in enumerate(allowed_action_space):
                    if restriction[0] <= restriction[1]:
                        if restriction[0] < allowed_subset[0] <= restriction[1] <= allowed_subset[1]:
                            allowed_subset[0] = restriction[1]
                        if restriction[1] > allowed_subset[1] >= restriction[0] >= allowed_subset[0]:
                            allowed_subset[1] = restriction[0]
                        if restriction[0] >= allowed_subset[0] and restriction[1] <= allowed_subset[1]:
                            if allowed_subset[0] != restriction[0]:
                                allowed_action_space.append([allowed_subset[0], restriction[0]])
                            if allowed_subset[1] != restriction[1]:
                                allowed_action_space.append([restriction[1], allowed_subset[1]])
                            allowed_subset[0] = np.inf
                        if restriction[0] < allowed_subset[0] and restriction[1] > allowed_subset[1]:
                            allowed_subset[0] = np.inf
                    else:
                        if restriction[0] <= allowed_subset[0] and restriction[1] <= allowed_subset[0] or (
                                restriction[0] >= allowed_subset[1]) and restriction[1] >= allowed_subset[1]:
                            allowed_subset[0] = np.inf
                        if allowed_subset[1] > restriction[0] > allowed_subset[0]:
                            allowed_subset[1] = restriction[0]
                        if allowed_subset[0] < restriction[1] < allowed_subset[1]:
                            allowed_subset[0] = restriction[1]

            allowed_action_space = np.array(
                [subset for subset in allowed_action_space if subset[0] != np.inf and subset[0] != subset[1]])

            if len(allowed_action_space) > 0:
                allowed_action_space[allowed_action_space[:, 0] != -obs['action_range'][0] / 2, 0] += self.SAFETY_ANGLE
                allowed_action_space[allowed_action_space[:, 1] != obs['action_range'][0] / 2, 1] -= self.SAFETY_ANGLE

            actions.append([[float(subset[0]), float(subset[1])] for subset in allowed_action_space
                            if subset[0] < subset[1]])

        return actions, state_batches, {}

    def generate_obstacles(self, height, width, seed: int = 42, max_iterations: int = 10000):
        def is_valid(el_coordinates):
            if (minimum_distance > el_coordinates[0]) or (el_coordinates[0] > width - minimum_distance
            ) or (minimum_distance > el_coordinates[1]) or (el_coordinates[1] > height - minimum_distance):
                return True

            for geometry in self.obstacles:
                geometry_coordinates = np.array(geometry.coordinates, dtype=np.float32)
                if Point(midpoint(geometry_coordinates)
                         ).distance(Point(el_coordinates)) < minimum_distance + np.sqrt(
                    2 * ((max(geometry_coordinates[:, 1]) - min(
                        geometry_coordinates[:, 1])) / 2) ** 2):
                    return True

            return False

        if seed is not None:
            np.random.seed(seed)

        self.obstacles = []

        iteration = 0
        while len(self.obstacles) < self.COUNT:
            iteration += 1

            size_obstacle = np.clip(np.random.normal(self.MEAN_SIZE, self.VARIANCE_SIZE),
                                    self.MEAN_SIZE - self.RANGE_SIZE,
                                    self.MEAN_SIZE + self.RANGE_SIZE)

            minimum_distance = np.sqrt(2 * (size_obstacle / 2) ** 2) + 0.95

            position = np.random.multivariate_normal([width / 2, height / 2],
                                                     self.POSITION_COVARIANCE)

            position[0] = np.clip(position[0], 0.0, width - size_obstacle)
            position[1] = np.clip(position[1], 0.0, height - size_obstacle)
            coordinates = SHAPE_COLLECTION[
                              np.random.randint(0, len(SHAPE_COLLECTION) - 1)] * size_obstacle + position - (
                                  size_obstacle / 2)

            if is_valid(position) or iteration > max_iterations:
                self.obstacles.append(Obstacle(coordinates=coordinates))
