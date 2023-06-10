import networkx as nx


def create_graph(edges):
    graph = nx.DiGraph()

    for [s, t], latency in edges:
        graph.add_edge(s, t, latency=latency)

    return graph


def analyze_graph(graph):
    number_of_nodes = graph.number_of_nodes()

    edge_list = list(graph.edges)
    edge_latencies = {i: graph[s][t]["latency"] for i, [s, t] in enumerate(graph.edges)}
    edge_indices = {e: i for i, e in enumerate(edge_list)}

    routes = {
        (s, t): [
            tuple(edge_indices[e] for e in path)
            for path in nx.all_simple_edge_paths(graph, s, t)
        ]
        for s in range(number_of_nodes)
        for t in range(number_of_nodes)
        if s != t
    }
    route_list = [y for x in routes.values() for y in x]
    route_indices = {tuple(r): i for i, r in enumerate(route_list)}

    return edge_list, edge_indices, edge_latencies, routes, route_list, route_indices


def edge_path_to_node_path(edge_path, edge_list):
    return "->".join(
        (str(edge_list[edge_path[0]][0]),)
        + tuple(str(edge_list[edge][1]) for edge in edge_path)
    )


def latency(edge, utilization):
    a, b, c = edge

    return a + b * (utilization**c)
