import numpy as np
from gymnasium.spaces import Discrete


class TrafficAgent:
    def __init__(self, route_list, source_target_map) -> None:
        self.route_list = route_list
        self.source_target_map = source_target_map

        self.action_space = Discrete(len(route_list))

    def act(self, observation):
        # Destructure observation, either with or without restriction
        if "restriction" in observation:
            observation, restriction = (
                observation["observation"],
                observation["restriction"],
            )
        else:
            restriction = None

        position, target, edge_travel_times = (
            observation["position"],
            observation["target"],
            observation["travel_times"],
        )

        # A route is feasible if it leads from the agent's position to its target, and it is allowed
        feasible_routes = np.array([(restriction is None or restriction.contains(index)) and s_t == (position, target) for index, s_t in enumerate(self.source_target_map)])

        # print(f'{feasible_routes=}')
        
        # If there is no allowed route, choose any route
        # if not allowed_routes.any():
        #     allowed_routes = np.array([s_t == (position, target) for index, s_t in enumerate(self.source_target_map)])

        # Calculate travel times for all routes, with np.inf for infeasible routes
        travel_times = np.array([
            sum(edge_travel_times[edge] for edge in route)
            if restriction is None
            or restriction.contains(i)
            and (st == (position, target))
            else np.inf
            for i, [route, st] in enumerate(
                zip(self.route_list, self.source_target_map)
            )
        ])

        # print(f'{travel_times=}')

        # if travel_times.min() == np.inf:
        #     # Choose any route
        #     return np.random.choice(len(self.route_list))
        # else:
            # Select routes with minimal travel time
        optimal_routes = np.argmin(travel_times, keepdims=True)
        # print(f'{optimal_routes=}')

        # Choose a route randomly from optimal routes
        return np.random.choice(optimal_routes)
