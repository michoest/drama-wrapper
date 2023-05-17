from decimal import Decimal

import numpy as np
from shapely import LineString


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
        intersection_type (str): The first part indicates the start and the second the end of the first line with respect to the agent's line. For example, negative_positive
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