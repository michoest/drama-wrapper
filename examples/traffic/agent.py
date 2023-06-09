import random

from gymnasium.spaces import Discrete


class TrafficAgent:
    def __init__(self, routes, route_indices, edge_indices) -> None:
        self.routes = routes
        self.route_indices = route_indices
        self.edge_indices = edge_indices

        self.action_space = Discrete(len(routes))

    def act(self, observation):
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

        possible_routes = self.routes[(position, target)]

        if restriction is not None:
            allowed_routes = [
                route
                for route in possible_routes
                if all(restriction.contains(edge) for edge in route)
            ]
        else:
            allowed_routes = possible_routes

        travel_times = {
            route: sum(edge_travel_times[edge] for edge in route)
            for route in allowed_routes
        }

        minimal_travel_time = min(travel_times.values())
        optimal_routes = [
            route
            for route in allowed_routes
            if travel_times[route] == minimal_travel_time
        ]

        return self.route_indices[random.choice(optimal_routes)]
