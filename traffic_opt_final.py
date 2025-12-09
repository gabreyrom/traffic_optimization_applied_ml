#%%
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString, MultiLineString
import networkx as nx

#%%
# Path to your shapefile
path = "./trimmed_manhattan_shape/trimmed_manhattan.shp"

gdf = gpd.read_file(path)

print("CRS:", gdf.crs)
print("Geometry types:", gdf.geom_type.unique())
print("Columns:", gdf.columns.tolist())

# If CRS is geographic (lat/lon), reproject to EPSG:2263 (NY State Plane)
if gdf.crs is not None and gdf.crs.is_geographic:
    gdf = gdf.to_crs(epsg=2263)
    print("Reprojected CRS: ", gdf.crs)

#%%
# Quick sanity plot of the shapefile
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10), sharey=True)
gdf.boundary.plot(ax=ax1, linewidth=1)
gdf.plot(ax=ax1, alpha=0.5, color="lightblue", edgecolor="black")
ax1.set_title("Manhattan Boundary and Background")

gdf.plot(ax=ax2, edgecolor="black", facecolor="none")
ax2.set_title("Manhattan Boundary only")

plt.show()

#%%
# Core types
Node = Tuple[float, float]     # (x, y)
EdgeId = Tuple[Node, Node]     # directed edge
EdgeFlows = Dict[EdgeId, float]

@dataclass
class Edge:
    start: Node
    end: Node
    length_m: float          # meters
    v_free: float            # free-flow speed (m/s)
    capacity: float          # max cars that fit on this segment
    alpha: float = 0.15
    beta: float = 4.0

    @property
    def free_time(self) -> float:
        # free-flow travel time (seconds)
        return self.length_m / self.v_free

    def travel_time(self, flow: float) -> float:
        # BPR-style congestion model
        x = flow / self.capacity if self.capacity > 0 else 0.0
        return self.free_time * (1.0 + self.alpha * (x ** self.beta))


Graph = Dict[Node, List[EdgeId]]

@dataclass
class TrafficNetwork:
    graph: Graph
    edges: Dict[EdgeId, Edge]

#%%
def make_node(x: float, y: float, ndigits: int = 3) -> Node:
    # rounding keeps nodes consistent instead of having tiny float differences
    return round(x, ndigits), round(y, ndigits)

#%%
# --- Stochastic edge parameters ---

def sample_car_length() -> float:
    # lognormal with median around 4.5m
    mu = math.log(4.5)
    sigma = 0.15
    length = random.lognormvariate(mu, sigma)
    return max(length, 3.5)  # clamp to > 3.5m


def sample_gap() -> float:
    # lognormal with median around 1.5m
    mu = math.log(1.5)
    sigma = 0.25
    gap = random.lognormvariate(mu, sigma)
    return max(gap, 0.5)


def sample_spacing() -> float:
    # space per vehicle = car length + gap
    return sample_car_length() + sample_gap()


def sample_free_speed_time_based() -> float:
    """
    Sample free-flow speed via a lognormal model on time-per-meter.
    This gives a fatter tail toward low speeds.
    """
    median_s_per_m = 0.09  # ~40 km/h
    mu = math.log(median_s_per_m)
    sigma = 0.3            # spread; larger => more slow edges

    s_per_m = random.lognormvariate(mu, sigma)  # seconds per meter
    v_free = 1.0 / s_per_m                      # m/s

    # clamp speeds to a plausible urban range ~ 18–90 km/h
    v_free = max(min(v_free, 25.0), 5.0)
    return v_free

#%%
def network_from_streets_gdf(streets_gdf: gpd.GeoDataFrame) -> TrafficNetwork:
    graph: Dict[Node, List[EdgeId]] = defaultdict(list)
    edges: Dict[EdgeId, Edge] = {}

    for idx, row in streets_gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue

        # Handle MultiLineString and LineString
        if isinstance(geom, MultiLineString):
            line_geoms = list(geom.geoms)
        elif isinstance(geom, LineString):
            line_geoms = [geom]
        else:
            continue

        for line in line_geoms:
            coords = list(line.coords)
            if len(coords) < 2:
                continue

            for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:]):
                u = make_node(x1, y1)
                v = make_node(x2, y2)

                seg = LineString([(x1, y1), (x2, y2)])
                length_m = seg.length

                # stochastic capacity & free-flow speed
                spacing = sample_spacing()                # m per car
                capacity = max(length_m / spacing, 1.0)   # cars
                v_free = sample_free_speed_time_based()   # m/s

                # directed u -> v
                e1 = (u, v)
                if e1 not in edges:
                    edges[e1] = Edge(
                        start=u,
                        end=v,
                        length_m=length_m,
                        v_free=v_free,
                        capacity=capacity,
                    )
                    graph[u].append(e1)

                # bidirectional by default
                e2 = (v, u)
                if e2 not in edges:
                    edges[e2] = Edge(
                        start=v,
                        end=u,
                        length_m=length_m,
                        v_free=v_free,
                        capacity=capacity,
                    )
                    graph[v].append(e2)

    return TrafficNetwork(graph=dict(graph), edges=edges)

#%% md
# Build the graph
#%%
manhattan_network = network_from_streets_gdf(gdf)
print(f"# nodes: {len(manhattan_network.graph)}")
print(f"# edges: {len(manhattan_network.edges)}")

#%% md
# Convert to NetworkX (for sanity plots)
#%%
def to_networkx(network: TrafficNetwork) -> nx.DiGraph:
    G = nx.DiGraph()
    for node in network.graph.keys():
        G.add_node(node, x=node[0], y=node[1])
    for e_id, edge in network.edges.items():
        G.add_edge(edge.start, edge.end,
                   free_time=edge.free_time,
                   capacity=edge.capacity)
    return G

G_nx = to_networkx(manhattan_network)
print("# NX nodes:", G_nx.number_of_nodes(), "# NX edges:", G_nx.number_of_edges())

pos = {n: (n[0], n[1]) for n in G_nx.nodes()}
plt.figure(figsize=(8, 8))
nx.draw(G_nx, pos=pos, node_size=1, linewidths=0.1, width=0.1)
plt.axis("equal")
plt.title("Road network graph from trimmed_manhattan.shp")
plt.show()

#%% md
# Agents + OD selection
#%%
@dataclass
class Agent:
    origin: Node
    destination: Node
    path: List[Node] = None

#%%
def sample_agents_random(network: TrafficNetwork, num_agents: int) -> List[Agent]:
    nodes = list(network.graph.keys())
    agents: List[Agent] = []
    for _ in range(num_agents):
        o = random.choice(nodes)
        d = random.choice(nodes)
        while d == o:
            d = random.choice(nodes)
        agents.append(Agent(origin=o, destination=d))
    return agents

#%%
def sample_agents_north_south(network: TrafficNetwork,
                               num_agents: int,
                               band_split: float = 0.5) -> List[Agent]:
    nodes = list(network.graph.keys())
    ys = [y for (_, y) in nodes]
    y_min, y_max = min(ys), max(ys)
    y_mid = y_min + band_split * (y_max - y_min)

    north_nodes = [n for n in nodes if n[1] >= y_mid]
    south_nodes = [n for n in nodes if n[1] <= y_mid]

    assert north_nodes and south_nodes, "North/south bands are empty – check CRS or band_split."

    agents: List[Agent] = []
    for _ in range(num_agents):
        o = random.choice(north_nodes)
        d = random.choice(south_nodes)
        agents.append(Agent(origin=o, destination=d))
    return agents

#%%
def pick_north_south_pair(network: TrafficNetwork) -> Tuple[Node, Node]:
    nodes = list(network.graph.keys())
    ys = [y for (_, y) in nodes]
    y_min, y_max = min(ys), max(ys)

    # north = 10% top band, south = 10% bottom band
    north_nodes = [n for n in nodes if n[1] >= y_min + 0.8 * (y_max - y_min)]
    south_nodes = [n for n in nodes if n[1] <= y_min + 0.2 * (y_max - y_min)]

    o = random.choice(north_nodes)
    d = random.choice(south_nodes)
    return o, d

#%%
def sample_agents_fixed(network: TrafficNetwork,
                        num_agents: int,
                        origin: Node,
                        destination: Node) -> List[Agent]:
    return [Agent(origin=origin, destination=destination) for _ in range(num_agents)]

#%% md
# Dijkstra on TrafficNetwork
#%%
import heapq

def dijkstra(network: TrafficNetwork,
             source: Node,
             target: Node,
             edge_flows: EdgeFlows) -> List[Node]:
    graph, edges = network.graph, network.edges

    dist: Dict[Node, float] = {source: 0.0}
    prev: Dict[Node, Node] = {}
    pq = [(0.0, source)]
    visited = set()

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)

        if u == target:
            break

        for edge_id in graph.get(u, []):
            edge = edges[edge_id]
            v = edge.end
            w = edge.travel_time(edge_flows[edge_id])  # cost depends on current flow
            nd = d + w

            if v not in dist or nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if target not in dist:
        return []  # no path found

    # Reconstruct path
    path: List[Node] = []
    cur = target
    while cur != source:
        path.append(cur)
        cur = prev[cur]
    path.append(source)
    path.reverse()
    return path

#%%
def edges_from_path(path: List[Node]) -> List[EdgeId]:
    return list(zip(path[:-1], path[1:]))

#%% md
# SPA routing: selfish, sequential assignment
#%%
def spa_route_all(network: TrafficNetwork,
                  agents: List[Agent]) -> Tuple[List[Agent], EdgeFlows]:
    edge_flows: EdgeFlows = defaultdict(float)

    for agent in agents:
        path = dijkstra(network, agent.origin, agent.destination, edge_flows)
        agent.path = path

        for e in edges_from_path(path):
            edge_flows[e] += 1.0  # 1 vehicle per agent

    return agents, edge_flows

#%% md
# World Utility
#%%
def total_system_travel_time(network: TrafficNetwork,
                             edge_flows: EdgeFlows) -> float:
    total = 0.0
    for e_id, flow in edge_flows.items():
        edge = network.edges[e_id]
        total += flow * edge.travel_time(flow)
    return total

#%% md
# Fixed OD pair and SPA shortest path (for reward shaping)
#%%
origin, destination = pick_north_south_pair(manhattan_network)

zero_flows = defaultdict(float)
shortest_path = dijkstra(manhattan_network, origin, destination, zero_flows)
shortest_edges = set(edges_from_path(shortest_path))

print("Origin:", origin)
print("Destination:", destination)
print("Shortest path length (edges):", len(shortest_path))

#%%
def dist_to_dest(node: Node, dest: Node) -> float:
    return math.dist(node, dest)

#%%
def edge_marginal_cost(edge: Edge, flow_before: float) -> float:
    """
    Given that 'flow_before' cars already use this edge,
    what is the marginal increase in global G if one more car uses it?
    """
    f1 = flow_before
    f2 = flow_before + 1

    t1 = edge.travel_time(f1)
    t2 = edge.travel_time(f2)

    G1 = f1 * t1
    G2 = f2 * t2

    return G2 - G1

#%% md
# COIN / RL helpers
#%%
def get_neighbors(network: TrafficNetwork, node: Node) -> List[Node]:
    return [network.edges[e_id].end for e_id in network.graph.get(node, [])]

#%%
def choose_action_eps_greedy(network: TrafficNetwork,
                             Q: Dict[Tuple[Node, Node, Node], float],
                             state_node: Node,
                             dest_node: Node,
                             epsilon: float) -> Node:
    neighbors = get_neighbors(network, state_node)
    if not neighbors:
        return state_node  # dead end, stay put (will break later)

    # exploration
    if random.random() < epsilon:
        return random.choice(neighbors)

    # exploitation: pick neighbor with highest Q
    best_q = None
    best_actions = []
    for a in neighbors:
        key = (state_node, dest_node, a)
        q = Q.get(key, 0.0)
        if best_q is None or q > best_q:
            best_q = q
            best_actions = [a]
        elif q == best_q:
            best_actions.append(a)

    return random.choice(best_actions)

#%% md
# COIN-RL with shaped rewards and fixed OD
#%%
def train_coin_rl(
    network: TrafficNetwork,
    episodes: int = 200,
    num_agents: int = 100,
    od_mode: str = "north_south",   # kept for compatibility
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon_start: float = 0.9,
    epsilon_end: float = 0.05,
    max_steps_per_agent: int = 1500
) -> Dict[Tuple[Node, Node, Node], float]:
    """
    COIN-style Q-learning with shaped rewards and a fixed origin/destination.

    Uses the globally defined `origin`, `destination`, and `shortest_edges`:
      - origin, destination: fixed north/south pair
      - shortest_edges: edges on the SPA shortest path between origin and destination

    Reward combines:
      - negative marginal congestion cost (at free-flow),
      - progress toward the destination,
      - penalty for using SPA-shortest-path edges.

    Q[(state_node, dest_node, action_node)] estimates the long-term (shaped)
    return of choosing `action_node` from `state_node` when heading to
    `dest_node` (here always the fixed `destination`).
    """
    Q: Dict[Tuple[Node, Node, Node], float] = {}

    # reward weights (you can tune these)
    lambda_cong = 1.0   # weight on marginal congestion cost
    lambda_prog = 0.05  # weight on progress toward destination
    lambda_sp   = 0.5   # penalty for using SPA shortest-path edges

    for ep in range(episodes):
        # Linear epsilon decay
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * (
            ep / max(1, episodes - 1)
        )

        # Sample agents for this episode: ALL use the fixed OD pair
        agents = sample_agents_fixed(network, num_agents, origin, destination)

        for agent in agents:
            s = agent.origin
            d = agent.destination
            steps = 0

            while s != d and steps < max_steps_per_agent:
                neighbors = get_neighbors(network, s)
                if not neighbors:
                    break  # dead end

                a = choose_action_eps_greedy(network, Q, s, d, epsilon)

                # find edge
                edge_id = (s, a)
                if edge_id not in network.edges:
                    break

                edge = network.edges[edge_id]

                # (1) COIN-style congestion term: use free-flow marginal cost
                f_before = 0.0
                delta_G = edge_marginal_cost(edge, f_before)

                # (2) progress term: decrease in distance to destination
                old_dist = dist_to_dest(s, d)
                new_dist = dist_to_dest(a, d)
                progress = old_dist - new_dist  # > 0 if moving closer

                # (3) penalty for using SPA shortest-path edges
                on_spa_shortest = 1.0 if edge_id in shortest_edges else 0.0

                # Combined shaped reward
                reward = (
                    -lambda_cong * delta_G
                    + lambda_prog * progress
                    - lambda_sp * on_spa_shortest
                )

                # next state
                s_next = a

                # Q-learning update
                key = (s, d, a)
                old_q = Q.get(key, 0.0)

                # best next Q
                next_neighbors = get_neighbors(network, s_next)
                if next_neighbors:
                    max_next_q = max(
                        Q.get((s_next, d, a_next), 0.0)
                        for a_next in next_neighbors
                    )
                else:
                    max_next_q = 0.0

                target = reward + gamma * max_next_q
                Q[key] = old_q + alpha * (target - old_q)

                # move
                s = s_next
                steps += 1

        print(f"Episode {ep+1}/{episodes} finished, epsilon={epsilon:.3f}")

    return Q

#%%
def route_with_trained_Q(
    network: TrafficNetwork,
    Q: Dict[Tuple[Node, Node, Node], float],
    num_agents: int = 200,
    od_mode: str = "fixed",
    max_steps_per_agent: int = 1500
) -> Tuple[List[Agent], EdgeFlows, float]:
    """
    Evaluate a trained Q-table by routing agents and computing the resulting edge flows and G.
    By default uses the fixed global (origin, destination); if od_mode == "north_south"
    you can still sample north-south commuters instead.
    """
    edge_flows: EdgeFlows = defaultdict(float)

    # sample new agents
    if od_mode == "random":
        agents = sample_agents_random(network, num_agents)
    elif od_mode == "north_south":
        agents = sample_agents_north_south(network, num_agents)
    else:  # "fixed" or anything else -> use fixed OD
        agents = sample_agents_fixed(network, num_agents, origin, destination)

    for agent in agents:
        s = agent.origin
        d = agent.destination
        path = [s]
        steps = 0

        while s != d and steps < max_steps_per_agent:
            neighbors = get_neighbors(network, s)
            if not neighbors:
                break

            # greedy: epsilon = 0
            best_q = None
            best_actions = []
            for a in neighbors:
                key = (s, d, a)
                q = Q.get(key, 0.0)
                if best_q is None or q > best_q:
                    best_q = q
                    best_actions = [a]
                elif q == best_q:
                    best_actions.append(a)

            a = random.choice(best_actions)

            edge_id = (s, a)
            if edge_id not in network.edges:
                break

            edge_flows[edge_id] += 1.0
            s = a
            path.append(s)
            steps += 1

        agent.path = path

    G = total_system_travel_time(network, edge_flows)
    return agents, edge_flows, G

#%% md
# Experiment wrappers
#%%
def run_spa_experiment(network: TrafficNetwork,
                       num_agents: int = 500,
                       od_mode: str = "north_south") -> Tuple[List[Agent], EdgeFlows, float]:
    if od_mode == "random":
        agents = sample_agents_random(network, num_agents)
    elif od_mode in ("north_south", "north-south"):
        agents = sample_agents_north_south(network, num_agents)
    elif od_mode == "fixed":
        agents = sample_agents_fixed(network, num_agents, origin, destination)
    else:
        raise ValueError(f"Unknown od_mode: {od_mode}")

    agents, edge_flows = spa_route_all(network, agents)
    G = total_system_travel_time(network, edge_flows)

    print(f"SPA experiment with {num_agents} agents ({od_mode} O/D)")
    print(f"Total system travel time G: {G:.2f}")
    print(f"Used edges: {len([e for e, f in edge_flows.items() if f > 0])} / {len(network.edges)}")

    return agents, edge_flows, G

#%%
def run_coin_experiment_rl(
    network: TrafficNetwork,
    num_agents: int = 500,
    od_mode: str = "fixed",
    episodes: int = 200,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon_start: float = 0.9,
    epsilon_end: float = 0.05,
    max_steps_per_agent: int = 1500,
    seed: Optional[int] = 42
) -> Tuple[Dict[Tuple[Node, Node, Node], float], List[Agent], EdgeFlows, float]:
    """
    Train a COIN-style RL policy and evaluate it on a fresh set of agents.
    Returns: Q, agents_coin, edge_flows_coin, G_coin
    """
    if seed is not None:
        random.seed(seed)

    print("=== Training COIN-RL policy ===")
    Q = train_coin_rl(
        network=network,
        episodes=episodes,
        num_agents=num_agents,
        od_mode=od_mode,
        alpha=alpha,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        max_steps_per_agent=max_steps_per_agent,
    )

    print("\n=== Evaluating COIN-RL policy ===")
    agents_coin, edge_flows_coin, G_coin = route_with_trained_Q(
        network=network,
        Q=Q,
        num_agents=num_agents,
        od_mode=od_mode,
        max_steps_per_agent=max_steps_per_agent,
    )

    print(f"COIN-RL experiment with {num_agents} agents ({od_mode} O/D)")
    print(f"Total system travel time G_COIN_RL: {G_coin:.2f}")
    print(f"Used edges: {len([e for e, f in edge_flows_coin.items() if f > 0])} / {len(network.edges)}")

    return Q, agents_coin, edge_flows_coin, G_coin

#%% md
# Run main experiments
#%%
random.seed(42)

num_agents = 1000

# SPA baseline (fixed OD so it's comparable to COIN-RL)
agents_spa, flows_spa, G_spa = run_spa_experiment(
    manhattan_network,
    num_agents=num_agents,
    od_mode="fixed"
)

# Train + evaluate COIN-RL
Q, agents_coin, flows_coin, G_coin = run_coin_experiment_rl(
    manhattan_network,
    num_agents=num_agents,
    od_mode="fixed",
    episodes=200
)

print(f"\nG_SPA      = {G_spa:.2f}")
print(f"G_COIN_RL  = {G_coin:.2f}")
print(f"Improvement = {(G_spa - G_coin) / G_spa * 100:.2f}%")

#%% md
# Per-agent travel times
#%%
def compute_agent_travel_times(network: TrafficNetwork,
                               edge_flows: EdgeFlows,
                               agents: List[Agent]) -> List[float]:
    times = []
    for agent in agents:
        if not agent.path or len(agent.path) < 2:
            times.append(0.0)
            continue
        total = 0.0
        for u, v in edges_from_path(agent.path):
            edge = network.edges[(u, v)]
            flow = edge_flows[(u, v)]
            total += edge.travel_time(flow)
        times.append(total)
    return times

spa_times = compute_agent_travel_times(manhattan_network, flows_spa, agents_spa)
coin_times = compute_agent_travel_times(manhattan_network, flows_coin, agents_coin)

print("SPA avg time:", sum(spa_times) / len(spa_times))
print("COIN-RL avg time:", sum(coin_times) / len(coin_times))

#%% md
# Visualization helpers
#%%
def routes_to_gdf(agents: List[Agent], crs) -> gpd.GeoDataFrame:
    geoms = []
    for agent in agents:
        if agent.path is not None and len(agent.path) > 1:
            geoms.append(LineString(agent.path))
    return gpd.GeoDataFrame(geometry=geoms, crs=crs)

def edge_flows_to_gdf(edge_flows: EdgeFlows,
                      network: TrafficNetwork,
                      crs) -> gpd.GeoDataFrame:
    geoms = []
    flows = []
    for (u, v), f in edge_flows.items():
        if f <= 0:
            continue
        geoms.append(LineString([u, v]))
        flows.append(f)
    return gpd.GeoDataFrame({"flow": flows}, geometry=geoms, crs=crs)

#%% md
# SPA vs COIN routes
#%%
routes_spa = routes_to_gdf(agents_spa, crs=gdf.crs)
routes_coin = routes_to_gdf(agents_coin, crs=gdf.crs)

fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# SPA routes
gdf.plot(ax=axes[0], linewidth=0.3, color="lightgray")
routes_spa.plot(ax=axes[0], linewidth=1.2, alpha=0.9)
axes[0].set_title(f"SPA routes (G = {G_spa:.1f})")
axes[0].set_axis_off()

# COIN routes
gdf.plot(ax=axes[1], linewidth=0.3, color="lightgray")
routes_coin.plot(ax=axes[1], linewidth=1.2, alpha=0.9)
axes[1].set_title(f"COIN routes (G = {G_coin:.1f})")
axes[1].set_axis_off()

plt.tight_layout()
plt.show()

#%% md
# SPA vs COIN heatmaps
#%%
flows_spa_gdf = edge_flows_to_gdf(flows_spa, manhattan_network, crs=gdf.crs)
flows_coin_gdf = edge_flows_to_gdf(flows_coin, manhattan_network, crs=gdf.crs)

fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# SPA heatmap
gdf.plot(ax=axes[0], linewidth=0.2, color="lightgray")
flows_spa_gdf.plot(
    ax=axes[0],
    column="flow",
    linewidth=2,
    alpha=0.9,
    legend=True
)
axes[0].set_title("SPA edge flows")
axes[0].set_axis_off()

# COIN heatmap
gdf.plot(ax=axes[1], linewidth=0.2, color="lightgray")
flows_coin_gdf.plot(
    ax=axes[1],
    column="flow",
    linewidth=2,
    alpha=0.9,
    legend=True
)
axes[1].set_title("COIN edge flows")
axes[1].set_axis_off()

plt.tight_layout()
plt.show()