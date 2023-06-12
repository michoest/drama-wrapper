import random
from collections import Counter

import gymnasium
from gymnasium.spaces import Box, Discrete, Dict
import numpy as np

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

from utils import analyze_graph, edge_path_to_node_path, latency


class TrafficEnvironment(AECEnv):
    metadata = {"render_modes": ["human"], "name": "traffic_v1"}

    def __init__(self, graph, agents, possible_agent_routes, number_of_routes, edge_latencies, route_list, number_of_steps):
        super().__init__()

        self.possible_agents = agents

        # edge_list, _, edge_latencies, routes, route_list, route_indices, source_target_map = analyze_graph(
        #     graph
        # )

        self._number_of_nodes = graph.number_of_nodes()
        self._number_of_edges = graph.number_of_edges()
        self._number_of_routes = number_of_routes

        self._observation_space = Dict(
            {
                "position": Discrete(self._number_of_nodes),
                "target": Discrete(self._number_of_nodes),
                "travel_times": Box(0, np.inf, shape=(self._number_of_edges,)),
            }
        )
        self._action_space = Discrete(self._number_of_routes)

        # self.routes = routes
        # self.route_indices = route_indices
        self.route_list = route_list
        # self.edge_list = edge_list
        self.edge_latencies = edge_latencies

        self.possible_agent_routes = possible_agent_routes

        self.number_of_steps = number_of_steps

        self.render_mode = "human"

        self._actions = None

    def observation_space(self, agent):
        return self._observation_space

    def action_space(self, agent):
        return self._action_space

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if len(self.agents) == len(self.possible_agents):
            print(", ".join(f"{agent} ({self._properties[agent][0]} -> {self._properties[agent][1]}): {edge_path_to_node_path(self.route_list[self._actions[agent]], self.edge_list) if self._actions[agent] is not None else None}" for agent in self.agents))  # fmt: skip
        else:
            print("Game over")

    def observe(self, agent):
        position, target = self._properties[agent]

        agents_per_edge = Counter(
            edge
            for _agent, route in self._routes_taken.items()
            if _agent != agent
            for edge in route
        )
        travel_times = {
            edge: latency(
                self.edge_latencies[edge], agents_per_edge[edge] / self.max_num_agents
            )
            for edge in range(self._number_of_edges)
        }

        return {
            "position": position,
            "target": target,
            "travel_times": np.array(list(travel_times.values())),
        }

    def close(self):
        pass

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._properties = {
            agent: random.choice(self.possible_agent_routes) for agent in self.agents
        }
        self._actions = {agent: None for agent in self.agents}
        self._routes_taken = {agent: tuple() for agent in self.agents}
        self._travel_times = {
            edge: latency(self.edge_latencies[edge], 0)
            for edge in range(self._number_of_edges)
        }
        self._state = {agent: None for agent in self.agents}
        self.current_step = 0

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        self._actions[self.agent_selection] = action
        self._routes_taken[self.agent_selection] = self.route_list[action]

        agents_per_edge = Counter(
            edge for route in self._routes_taken.values() for edge in route
        )
        self._travel_times = {
            edge: latency(
                self.edge_latencies[edge], agents_per_edge[edge] / self.max_num_agents
            )
            for edge in range(self._number_of_edges)
        }

        self.rewards = {
            agent: -sum(self._travel_times[edge] for edge in self._routes_taken[agent])
            for agent in self.agents
        }
        self._cumulative_rewards = self.rewards.copy()

        self.truncations = {agent: self.current_step >= self.number_of_steps for agent in self.agents}

        self.agent_selection = self._agent_selector.next()

        if self.render_mode == "human":
            self.render()

        self.current_step += 1

    def state(self):
        return np.array(list(self._travel_times.values()))
