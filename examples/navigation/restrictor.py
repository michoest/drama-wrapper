from decimal import Decimal

import numpy as np
from gymnasium.spaces import Box, Dict
from shapely import Polygon, Point, LineString

from examples.navigation.env import NavigationEnvironment, Agent
from drama.restrictions import IntervalUnionRestriction
from drama.restrictors import Restrictor, IntervalUnionActionSpace

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


def get_restrictions_for_polygon(polygon_coordinates, agent):
    """ Calculates the restriction angles for the agent and a single polygon

    Args:
        polygon_coordinates (list): List of polygon corner points that define the shape of the obstacle
        agent

    Returns:
        restrictions (list): List of intervals which would lead to a collision. For example [[-10, 30]]
    """
    max_angle = Decimal(-np.inf)
    min_angle = Decimal(np.inf)
    agent_on_action_space_boundary = agent.y == Decimal(repr(polygon_coordinates[0][0])
                                                        ) if len(polygon_coordinates) > 0 else False
    boundary_crossed_negative = False
    boundary_crossed_positive = False

    for index, coordinates in enumerate(polygon_coordinates):
        coordinates = list(coordinates)
        coordinates[0] = Decimal(repr(coordinates[0]))
        coordinates[1] = Decimal(repr(coordinates[1]))

        # Check if next coordinates go beyond max and min action space boundaries.
        # For example: Coordinate 1 -> -170 and coordinate 2 -> -190 with boundary -180
        if index != 0:
            coordinate_direction_line = (coordinates[0], coordinates[1],
                                         Decimal(repr(polygon_coordinates[index - 1][0])),
                                         Decimal(repr(polygon_coordinates[index - 1][1])))
            action_space_boundary_line = (agent.x, agent.y,
                                          agent.x - agent.radius - agent.step_size, agent.y)
            line_crossed = line_intersection(*coordinate_direction_line, *action_space_boundary_line)

            if not boundary_crossed_positive and line_crossed in ['negative_positive', 'negative_line']:
                boundary_crossed_negative = True
            elif not boundary_crossed_negative and line_crossed in ['positive_negative', 'line_negative']:
                boundary_crossed_positive = True
            elif boundary_crossed_negative and line_crossed in ['positive_negative', 'line_negative',
                                                                'line_right_out'
                                                                ] and not agent_on_action_space_boundary:
                boundary_crossed_negative = False
            elif boundary_crossed_positive and line_crossed in ['negative_positive', 'negative_line']:
                boundary_crossed_positive = False
            if agent_on_action_space_boundary and line_crossed in ['line_positive']:
                agent_on_action_space_boundary = False
            if agent_on_action_space_boundary and line_crossed in ['line_negative']:
                agent_on_action_space_boundary = False

        # Angle to polygon corner
        if Decimal(coordinates[0]) == agent.x:
            angle_to_coordinates = Decimal(90.0)
        else:
            angle_to_coordinates = Decimal(repr(np.rad2deg(np.arctan(float(
                np.abs(coordinates[1] - agent.y) / np.abs(
                    coordinates[0] - agent.x))))))

        # Subtract 180 if polygon corner lies left to agent
        if agent.x > coordinates[0]:
            angle_to_coordinates = Decimal(180.0) - angle_to_coordinates

        # Negative if polygon corner is below agent
        if agent.y > coordinates[1] or index == 0 and agent.y == coordinates[1] and index + 1 != len(
                polygon_coordinates) and Decimal(repr(polygon_coordinates[index + 1][1])) < agent.y:
            angle_to_coordinates = -angle_to_coordinates

        # Correct if polygon corner goes beyond possible action space
        if boundary_crossed_negative and angle_to_coordinates != -180:
            angle_to_coordinates = angle_to_coordinates - Decimal(360.0)
        elif boundary_crossed_positive and angle_to_coordinates != 180:
            angle_to_coordinates = angle_to_coordinates + Decimal(360.0)

        if angle_to_coordinates > max_angle:
            max_angle = angle_to_coordinates
        if angle_to_coordinates < min_angle:
            min_angle = angle_to_coordinates

    return [min_angle - agent.perspective,
            max_angle - agent.perspective]


def line_intersection(c1, c2, n1, n2, agent_x1, agent_y1, agent_x2, agent_y2):
    """ Determines how the first line intersects with the second.
        For example, if the second line is crossed from the bottom up.
        Used to see if a restriction goes beyond the possible range of actions ([-180,180]).

    Args:
        c1 (Decimal): x-coordinate of the first line's starting position
        c2 (Decimal): y-coordinate of the first line's starting position
        n1 (Decimal): x-coordinate of the first line's closing position
        n2 (Decimal): y-coordinate of the first line's closing position
        agent_x1 (Decimal): x-coordinate of the agent's line starting position
        agent_y1 (Decimal): y-coordinate of the agent's line starting position
        agent_x2 (Decimal): x-coordinate of the agent's line closing position
        agent_y2 (Decimal): y-coordinate of the agent's line closing position

    Returns:
        intersection_type (str): The first part indicates the start and the second the end of the first line
        with respect to the agent's line. For example, negative_positive
    """
    intersection = len(
        np.array(LineString([
            (float(agent_x1), float(agent_y1)),
            (float(agent_x2), float(agent_y2))
        ]).intersection(LineString([(c1, c2), (n1, n2)])).coords)) > 0
    if intersection and c2 > agent_y1 > n2:
        return 'negative_positive'
    if intersection and c2 < agent_y1 < n2:
        return 'positive_negative'
    if intersection and c2 == agent_y1 and n2 > agent_y1:
        return 'positive_line'
    if intersection and c2 == agent_y1 and n2 < agent_y1:
        return 'negative_line'
    if intersection and c2 == agent_y1 and n2 == agent_y1 and c1 >= agent_x1:
        return 'line_right_out'
    if intersection and c2 == agent_y1 and n2 == agent_y1 and c1 < agent_x1:
        return 'line_line'
    if intersection and c2 < agent_y1 and n2 == agent_y1:
        return 'line_negative'
    if intersection and c2 > agent_y1 and n2 == agent_y1:
        return 'line_positive'
    return 'none'


def project_intervals_into_action_space(intervals, low: Decimal, high: Decimal):
    """ Projects action spaces that go beyond [-180, 180] back into the range

    Args:
        intervals (list): Allowed action space
        low (float): Minimum of the allowed action space (In our case -180)
        high (float): Maximum of the allowed action space (In our case 180)

    Returns:
        maximum (float)
    """
    action_space_range = high - low
    for subspace in intervals:
        if subspace[0] != Decimal(np.inf):
            if subspace[0] > high:
                subspace[0] -= action_space_range
            elif subspace[0] < low:
                subspace[0] += action_space_range
            if subspace[1] > high:
                subspace[1] -= action_space_range
            elif subspace[1] < low:
                subspace[1] += action_space_range

    return [subspace for subspace in intervals if subspace[0] != Decimal(np.inf)]


def inverse_space(space, low: Decimal, high: Decimal):
    """ Finds the allowed given restrictions

    Args:
        space (list): Restrictions
        low (float): Minimum of the allowed action space
        high (float): Maximum of the allowed action space

    Returns:
        allowed (list)
    """
    inverse = [[low, high]]

    for original_subspace in space:
        to_test = []
        if original_subspace[0] > original_subspace[1]:
            if not original_subspace[0] == high:
                to_test.append([original_subspace[0], high])
            if not original_subspace[1] == low:
                to_test.append([low, original_subspace[1]])
        else:
            to_test = [original_subspace]
        for subspace in to_test:
            for index, inverse_subspace in enumerate(inverse):
                if subspace[0] < inverse_subspace[0] <= subspace[1] <= inverse_subspace[1]:
                    inverse_subspace[0] = subspace[1]
                if subspace[1] > inverse_subspace[1] >= subspace[0] >= inverse_subspace[0]:
                    inverse_subspace[1] = subspace[0]
                if subspace[0] >= inverse_subspace[0] and subspace[1] <= inverse_subspace[1]:
                    if inverse_subspace[0] != subspace[0]:
                        inverse.append([inverse_subspace[0], subspace[0]])
                    if inverse_subspace[1] != subspace[1]:
                        inverse.append([subspace[1], inverse_subspace[1]])
                    inverse_subspace[0] = Decimal(np.inf)

    inverse = [not_allowed_space for not_allowed_space in inverse if
               not_allowed_space[0] != Decimal(np.inf) and not_allowed_space[0] != not_allowed_space[1]]
    return inverse


def midpoint(coordinates: np.ndarray):
    """ Calculates the midpoint of a polygon

    Args:
        coordinates (list): Coordinates that define the shape of the polygon
    """
    return [(max(coordinates[:, 0]) - min(coordinates[:, 0])) / 2 + min(coordinates[:, 0]),
            (max(coordinates[:, 1]) - min(coordinates[:, 1])) / 2 + min(coordinates[:, 1])]


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


class NavigationRestrictor(Restrictor):

    def __init__(self, obstacle_count, obstacle_position_covariance, obstacle_mean_size, obstacle_variance_size,
                 obstacle_size_range, start_seed, safety_angle, min_angle, max_angle):
        super().__init__(
            observation_space=Dict({
                'state': Box(0.0, np.inf, shape=(3,), dtype=np.float32),
                'width': Box(0.0, np.inf, shape=(1,), dtype=np.float32),
                'height': Box(0.0, np.inf, shape=(1,), dtype=np.float32),
                'agent_radius': Box(0.0, np.inf, shape=(1,), dtype=np.float32),
                'step_size': Box(0.0, np.inf, shape=(1,), dtype=np.float32),
                'step': Box(0.0, np.inf, shape=(1,), dtype=np.float32)
            }),
            action_space=IntervalUnionActionSpace(Box(low=min_angle, high=max_angle, dtype=np.float32)))

        self.count = obstacle_count
        self.position_covariance = obstacle_position_covariance
        self.mean_size = obstacle_mean_size
        self.variance_size = obstacle_variance_size
        self.size_range = obstacle_size_range
        self.safety_angle = safety_angle
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.seed = start_seed

        self.obstacles = []
        self.map_collision_area = None

    def preprocess_observation(self, env: NavigationEnvironment):
        return {
            'state': env.state(),
            'width': np.array([env.WIDTH], dtype=np.float32),
            'height': np.array([env.HEIGHT], dtype=np.float32),
            'agent_radius': np.array([float(env.agent.radius)], dtype=np.float32),
            'step_size': np.array([float(env.agent.step_size)], dtype=np.float32),
            'step': np.array([env.current_step], dtype=np.float32)
        }

    def act(self, observation: dict) -> IntervalUnionRestriction:
        if observation['step'][0] == 0:
            self.generate_obstacles(observation['height'][0], observation['width'][0], seed=self.seed)
            self.map_collision_area = Polygon([(0.0, 0.0), (observation['width'][0], 0.0),
                                               (observation['width'][0], observation['height']),
                                               (0.0, observation['height'][0])]).exterior.buffer(
                observation['agent_radius'][0])
            self.seed += 1

        agent = Agent(x=observation['state'][0], y=observation['state'][1],
                      perspective=observation['state'][2],
                      radius=observation['agent_radius'][0], step_size=observation['step_size'][0])
        step_circle = Point(observation['state'][0], observation['state'][1]).buffer(float(agent.step_size))
        restrictions = []

        for obstacle in self.obstacles + [self.map_collision_area]:
            if isinstance(obstacle, Obstacle):
                obstacle = obstacle.collision_area(float(agent.radius))

            is_in_collision_area = obstacle.contains(
                Point(float(agent.x), float(agent.y))) or obstacle.boundary.contains(
                Point(float(agent.x), float(agent.y)))

            obstacle_step_circle_intersection = step_circle.intersection(
                obstacle) if not is_in_collision_area else (
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

        interval_union_restriction = IntervalUnionRestriction(Box(self.min_angle, self.max_angle, dtype=np.float32))
        for restriction in restrictions:
            if restriction[0] != restriction[1]:
                if restriction[1] < restriction[0]:
                    interval_union_restriction.remove(-180.0, float(restriction[1]) + self.safety_angle)
                    interval_union_restriction.remove(float(restriction[0]) - self.safety_angle, 180.0)
                else:
                    interval_union_restriction.remove(float(restriction[0]) - self.safety_angle,
                                                      float(restriction[1]) + self.safety_angle)
        return interval_union_restriction

    def generate_obstacles(self, height, width, seed: int = 42, max_iterations: int = 10000):
        def is_valid(el_coordinates):
            out_of_map = minimum_distance > el_coordinates[0] or el_coordinates[0] > width - minimum_distance or (
                    minimum_distance > el_coordinates[1]) or el_coordinates[1] > height - minimum_distance

            collision = np.any(
                [Point(midpoint(geometry.coordinates)).distance(Point(el_coordinates)) < minimum_distance + np.sqrt(
                    2 * (
                            (max([float(t) for t in geometry.coordinates[:, 1]]) - min(
                                [float(t) for t in geometry.coordinates[:, 1]])) / 2) ** 2) for geometry in self.obstacles])

            return not out_of_map and not collision

        rng = np.random.RandomState(seed)

        self.obstacles = []

        iteration = 0
        while len(self.obstacles) < self.count:
            iteration += 1

            size_obstacle = np.clip(rng.normal(self.mean_size, self.variance_size),
                                    self.mean_size - self.size_range,
                                    self.mean_size + self.size_range)

            minimum_distance = np.sqrt(2 * (size_obstacle / 2) ** 2) + 0.95

            position = rng.multivariate_normal([width / 2, height / 2],
                                               self.position_covariance)

            position[0] = np.clip(position[0], 0.0, width - size_obstacle)
            position[1] = np.clip(position[1], 0.0, height - size_obstacle)
            coordinates = SHAPE_COLLECTION[
                              rng.randint(0, len(SHAPE_COLLECTION) - 1)] * size_obstacle + position - (
                                  size_obstacle / 2)

            if is_valid(position) or iteration > max_iterations:
                self.obstacles.append(Obstacle(coordinates=coordinates))
