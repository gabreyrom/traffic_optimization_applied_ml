from __future__ import annotations

import itertools
import math
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import geopandas as gpd
import heapq
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from shapely.geometry import LineString, MultiLineString

# =========================
# Types and constants
# =========================
Node = Tuple[float, float]
EdgeId = Tuple[Node, Node]
EdgeFlows = Dict[EdgeId, float]
Graph = Dict[Node, List[EdgeId]]

FT_TO_M = 0.3048006096
SPACE_PER_VEH_M = 7.5
MIN_STORAGE_CAP = 1
BPR_ALPHA = 0.15
BPR_BETA = 4.0
Q_SAT_VPHPL = 900.0
DELTA_VPH = 50.0

TIME_SCALE = 1000.0
LAMBDA_DR = 0.3
LAMBDA_PROG = 0.1
LAMBDA_TERM = 10.0
LAMBDA_EXP = 0.1
LAMBDA_STEP = 0.01
LAMBDA_MISS = 50.0

HIGHWAY_DEFAULT_LANES = {
    "motorway": 6,
    "trunk": 5,
    "primary": 5,
    "secondary": 4,
    "tertiary": 2,
    "residential": 2,
    "service": 1,
}

FF_NOISE_EPS = 0.02


@dataclass
class Edge:
    start: Node
    end: Node
    length_m: float
    v_free: float
    storage_cap: int
    flow_cap_vph: float
    alpha: float = BPR_ALPHA
    beta: float = BPR_BETA

    @property
    def free_time(self) -> float:
        return self.length_m / self.v_free if self.v_free > 0 else float("inf")

    def travel_time(self, flow_vph: float) -> float:
        if self.flow_cap_vph <= 0:
            return float("inf")
        x = max(0.0, flow_vph / self.flow_cap_vph)
        return self.free_time * (1.0 + self.alpha * (x ** self.beta))


@dataclass
class TrafficNetwork:
    graph: Graph
    edges: Dict[EdgeId, Edge]


@dataclass
class Agent:
    origin: Node
    destination: Node
    path: Optional[List[Node]] = None
    experienced_time_s: Optional[float] = None


@dataclass
class RewardDebug:
    enabled: bool = True
    n: int = 0
    sums: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    mins: Dict[str, float] = field(default_factory=lambda: defaultdict(lambda: float("inf")))
    maxs: Dict[str, float] = field(default_factory=lambda: defaultdict(lambda: float("-inf")))

    def add(self, **vals: Any) -> None:
        if not self.enabled:
            return
        self.n += 1
        for k, v in vals.items():
            if v is None:
                continue
            fv = float(v)
            self.sums[k] += fv
            self.mins[k] = min(self.mins[k], fv)
            self.maxs[k] = max(self.maxs[k], fv)

    def summary(self, name: str = "Reward breakdown") -> None:
        if not self.enabled or self.n == 0:
            print(f"[{name}] (no samples)")
            return
        print(f"\n[{name}] n={self.n}")
        for k in sorted(self.sums.keys()):
            mean = self.sums[k] / self.n
            print(f"  {k:8s}  mean={mean: .4f}   min={self.mins[k]: .4f}   max={self.maxs[k]: .4f}")


class RunningStats:
    def __init__(self, eps: float = 1e-8):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.eps = eps

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)

    @property
    def std(self) -> float:
        return math.sqrt(self.M2 / max(self.n - 1, 1)) + self.eps


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        from collections import deque as _deque

        self._buf = _deque(maxlen=capacity)

    def push(self, s, d, a, reward: float, s_next, done: bool) -> None:
        self._buf.append((s, d, a, float(reward), s_next, bool(done)))

    def sample(self, batch_size: int):
        return random.sample(self._buf, min(batch_size, len(self._buf)))

    def __len__(self) -> int:
        return len(self._buf)


def make_node(x: float, y: float, ndigits: int = 3) -> Node:
    return round(x, ndigits), round(y, ndigits)


def parse_lanes(row: dict, default: int = 1) -> int:
    val = row.get("lanes", None)
    if val is None or (isinstance(val, float) and math.isnan(val)):
        hw = row.get("highway", None)
        return max(1, int(HIGHWAY_DEFAULT_LANES.get(hw, default)))
    try:
        if isinstance(val, str):
            for sep in (";", "|", ","):
                if sep in val:
                    val = val.split(sep)[0]
                    break
            val = val.strip()
        lanes = int(float(val))
        return max(1, lanes)
    except Exception:
        hw = row.get("highway", None)
        return max(1, int(HIGHWAY_DEFAULT_LANES.get(hw, default)))


def sample_free_speed_time_based() -> float:
    median_s_per_m = 0.09
    mu = math.log(median_s_per_m)
    sigma = 0.3
    s_per_m = random.lognormvariate(mu, sigma)
    v_free = 1.0 / s_per_m
    return max(min(v_free, 25.0), 5.0)


def network_from_streets_gdf(streets_gdf: gpd.GeoDataFrame) -> Tuple[TrafficNetwork, List[int]]:
    graph: Dict[Node, List[EdgeId]] = defaultdict(list)
    edges: Dict[EdgeId, Edge] = {}
    storage_caps: List[int] = []

    for _, row in streets_gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        if isinstance(geom, MultiLineString):
            line_geoms = list(geom.geoms)
        elif isinstance(geom, LineString):
            line_geoms = [geom]
        else:
            continue

        lanes = parse_lanes(row, default=1)
        for line in line_geoms:
            coords = list(line.coords)
            if len(coords) < 2:
                continue
            u = make_node(*coords[0])
            v = make_node(*coords[-1])
            length_m = float(line.length) * FT_TO_M
            storage_cap = max(MIN_STORAGE_CAP, int((length_m * lanes) // SPACE_PER_VEH_M))
            storage_caps.append(storage_cap)
            v_free = sample_free_speed_time_based()
            q_speed_vphpl = (v_free / SPACE_PER_VEH_M) * 3600.0
            q_lane_vph = min(q_speed_vphpl, Q_SAT_VPHPL)
            flow_cap_vph = lanes * q_lane_vph

            def _add_edge(a: Node, b: Node):
                eid = (a, b)
                if eid in edges:
                    if length_m > edges[eid].length_m:
                        edges[eid].length_m = length_m
                        edges[eid].storage_cap = storage_cap
                        edges[eid].flow_cap_vph = flow_cap_vph
                        edges[eid].v_free = v_free
                    return
                edges[eid] = Edge(
                    start=a,
                    end=b,
                    length_m=length_m,
                    v_free=v_free,
                    storage_cap=storage_cap,
                    flow_cap_vph=flow_cap_vph,
                )
                graph[a].append(eid)

            _add_edge(u, v)
            _add_edge(v, u)

    return TrafficNetwork(graph=dict(graph), edges=edges), storage_caps


def count_dead_ends(network: TrafficNetwork) -> int:
    outdeg = {n: len(network.graph.get(n, [])) for n in network.graph.keys()}
    indeg = defaultdict(int)
    for (u, v) in network.edges.keys():
        indeg[v] += 1
    dead = [n for n in set(outdeg) | set(indeg) if (outdeg.get(n, 0) + indeg.get(n, 0)) <= 1]
    return len(dead)


def to_networkx(network: TrafficNetwork) -> nx.DiGraph:
    g = nx.DiGraph()
    for node in network.graph.keys():
        g.add_node(node, x=node[0], y=node[1])
    for edge in network.edges.values():
        g.add_edge(
            edge.start,
            edge.end,
            free_time=edge.free_time,
            storage_cap=edge.storage_cap,
            flow_cap_vph=edge.flow_cap_vph,
            length_m=edge.length_m,
            v_free=edge.v_free,
        )
    return g


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


def sample_agents_north_south(network: TrafficNetwork, num_agents: int, band_split: float = 0.5) -> List[Agent]:
    nodes = list(network.graph.keys())
    ys = [y for (_, y) in nodes]
    y_min, y_max = min(ys), max(ys)
    y_mid = y_min + band_split * (y_max - y_min)
    north_nodes = [n for n in nodes if n[1] >= y_mid]
    south_nodes = [n for n in nodes if n[1] <= y_mid]
    assert north_nodes and south_nodes, "North/south bands empty."
    return [Agent(origin=random.choice(north_nodes), destination=random.choice(south_nodes)) for _ in range(num_agents)]


def pick_north_south_pair(network: TrafficNetwork) -> Tuple[Node, Node]:
    nodes = list(network.graph.keys())
    ys = [y for (_, y) in nodes]
    y_min, y_max = min(ys), max(ys)
    north_nodes = [n for n in nodes if n[1] >= y_min + 0.8 * (y_max - y_min)]
    south_nodes = [n for n in nodes if n[1] <= y_min + 0.2 * (y_max - y_min)]
    return random.choice(north_nodes), random.choice(south_nodes)


def sample_agents_fixed(network: TrafficNetwork, num_agents: int, origin: Node, destination: Node) -> List[Agent]:
    return [Agent(origin=origin, destination=destination) for _ in range(num_agents)]


def dijkstra(network: TrafficNetwork, source: Node, target: Node, edge_flow_vph: EdgeFlows) -> List[Node]:
    graph, edges = network.graph, network.edges
    dist: Dict[Node, float] = {source: 0.0}
    prev: Dict[Node, Node] = {}
    pq = [(0.0, source)]
    visited = set()

    while pq:
        d_u, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == target:
            break
        for edge_id in graph.get(u, []):
            edge = edges[edge_id]
            v = edge.end
            w = edge.travel_time(edge_flow_vph.get(edge_id, 0.0))
            nd = d_u + w
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if target not in dist:
        return []
    path: List[Node] = []
    cur = target
    while cur != source:
        path.append(cur)
        cur = prev[cur]
    path.append(source)
    path.reverse()
    return path


def edges_from_path(path: List[Node]) -> List[EdgeId]:
    return list(zip(path[:-1], path[1:]))


def spa_route_all(network: TrafficNetwork, agents: List[Agent], horizon_s: float = 3600.0) -> Tuple[List[Agent], EdgeFlows]:
    edge_counts: EdgeFlows = defaultdict(float)
    scale_to_vph = 3600.0 / horizon_s
    for agent in agents:
        edge_flow_vph = {e: c * scale_to_vph for e, c in edge_counts.items()}
        path = dijkstra(network, agent.origin, agent.destination, edge_flow_vph)
        agent.path = path
        travel_s = 0.0
        for e in edges_from_path(path):
            edge = network.edges[e]
            flow_vph = edge_counts[e] * scale_to_vph
            travel_s += edge.travel_time(flow_vph)
            edge_counts[e] += 1.0
        agent.experienced_time_s = travel_s
    return agents, edge_counts


def edge_marginal_cost(edge: Edge, flow_before_vph: float, delta_vph: float = DELTA_VPH) -> float:
    f1 = flow_before_vph
    f2 = flow_before_vph + delta_vph
    t1 = edge.travel_time(f1)
    t2 = edge.travel_time(f2)
    return (f2 * t2) - (f1 * t1)


def edge_assignment_cost(edge: Edge, flow_before_vph: float, delta_vph: float = DELTA_VPH) -> float:
    base = edge_marginal_cost(edge, flow_before_vph, delta_vph=delta_vph)
    if FF_NOISE_EPS > 0:
        base *= (1.0 + FF_NOISE_EPS * (random.random() - 0.5))
    return base


def dijkstra_coin_ff_marginal(network: TrafficNetwork, source: Node, target: Node, edge_flow_vph: EdgeFlows) -> List[Node]:
    graph, edges = network.graph, network.edges
    dist: Dict[Node, float] = {source: 0.0}
    prev: Dict[Node, Node] = {}
    pq = [(0.0, source)]
    visited = set()

    while pq:
        cost_u, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == target:
            break
        for edge_id in graph.get(u, []):
            edge = edges[edge_id]
            v = edge.end
            f_before_vph = float(edge_flow_vph.get(edge_id, 0.0))
            marginal = edge_assignment_cost(edge, flow_before_vph=f_before_vph)
            new_cost = cost_u + marginal
            if v not in dist or new_cost < dist[v]:
                dist[v] = new_cost
                prev[v] = u
                heapq.heappush(pq, (new_cost, v))

    if target not in dist:
        return []
    path: List[Node] = []
    cur = target
    while cur != source:
        path.append(cur)
        cur = prev[cur]
    path.append(source)
    path.reverse()
    return path


def ford_fulkerson_route_all(network: TrafficNetwork, agents: List[Agent], horizon_s: float = 3600.0) -> Tuple[List[Agent], EdgeFlows]:
    edge_counts: EdgeFlows = defaultdict(float)
    scale_to_vph = 3600.0 / horizon_s
    for agent in agents:
        edge_flow_vph = {e: c * scale_to_vph for e, c in edge_counts.items()}
        path = dijkstra_coin_ff_marginal(network, agent.origin, agent.destination, edge_flow_vph=edge_flow_vph)
        agent.path = path
        travel_s = 0.0
        for e in edges_from_path(path):
            edge = network.edges[e]
            flow_vph = edge_counts[e] * scale_to_vph
            travel_s += edge.travel_time(flow_vph)
            edge_counts[e] += 1.0
        agent.experienced_time_s = travel_s
    return agents, edge_counts


def total_system_travel_time(network: TrafficNetwork, edge_counts: EdgeFlows, horizon_s: float = 3600.0) -> float:
    total_s = 0.0
    scale = 3600.0 / horizon_s
    for e_id, count in edge_counts.items():
        edge = network.edges[e_id]
        flow_vph = count * scale
        total_s += count * edge.travel_time(flow_vph)
    return total_s


def edge_global_cost(edge: Edge, count: float, horizon_s: float) -> float:
    flow_vph = count * (3600.0 / horizon_s)
    return count * edge.travel_time(flow_vph)


def edge_marginal_global_cost(edge: Edge, count_before: float, horizon_s: float) -> float:
    return edge_global_cost(edge, count_before + 1.0, horizon_s) - edge_global_cost(edge, count_before, horizon_s)


def build_ff_expert_dataset(
    network: TrafficNetwork,
    num_agents: int = 200,
    od_mode: str = "fixed",
    horizon_s: float = 3600.0,
    fixed_od: Optional[Tuple[Node, Node]] = None,
):
    if od_mode == "random":
        agents = sample_agents_random(network, num_agents)
    elif od_mode in ("north_south", "north-south"):
        agents = sample_agents_north_south(network, num_agents)
    else:
        if fixed_od is None:
            raise ValueError("fixed_od=(origin, destination) is required when od_mode='fixed'")
        agents = sample_agents_fixed(network, num_agents, fixed_od[0], fixed_od[1])

    agents_ff, edge_counts_ff = ford_fulkerson_route_all(network, agents, horizon_s=horizon_s)

    expert_steps = []
    for agent in agents_ff:
        if not agent.path or agent.path[-1] != agent.destination:
            continue
        d = agent.destination
        for (s, a) in edges_from_path(agent.path):
            expert_steps.append((s, d, a, (s, a)))

    raw_dg: Dict[EdgeId, float] = {}
    for (_, _, _, edge_id) in expert_steps:
        if edge_id in raw_dg:
            continue
        edge = network.edges[edge_id]
        c_before = max(edge_counts_ff.get(edge_id, 0.0) - 1.0, 0.0)
        raw_dg[edge_id] = edge_marginal_global_cost(edge, c_before, horizon_s)

    expert_edge_quality: Dict[EdgeId, float] = {}
    if raw_dg:
        min_dg = min(raw_dg.values())
        max_dg = max(raw_dg.values())
        span = max(max_dg - min_dg, 1.0)
        for eid, dg in raw_dg.items():
            expert_edge_quality[eid] = 1.0 - (dg - min_dg) / span

    return expert_steps, agents_ff, edge_counts_ff, expert_edge_quality


def get_neighbors(network: TrafficNetwork, node: Node) -> List[Node]:
    return [network.edges[e_id].end for e_id in network.graph.get(node, [])]


def edge_congestion_bin(edge: Edge, flow_vph: float) -> int:
    cap_vph = max(edge.flow_cap_vph, 1e-6)
    ratio = flow_vph / cap_vph
    if ratio < 0.3:
        return 0
    if ratio < 0.7:
        return 1
    return 2


def dist_to_dest(node: Node, dest: Node) -> float:
    x1, y1 = node
    x2, y2 = dest
    return math.hypot(x2 - x1, y2 - y1)


def build_reverse_adj_weighted(network: TrafficNetwork) -> Dict[Node, List[Tuple[Node, float]]]:
    rev = defaultdict(list)
    for (u, v), edge in network.edges.items():
        rev[v].append((u, float(edge.free_time)))
    return dict(rev)


def shortest_path_to_dest_freeflow(dest: Node, rev_adj_w: Dict[Node, List[Tuple[Node, float]]]) -> Dict[Node, float]:
    dist = {dest: 0.0}
    pq = [(0.0, dest)]
    while pq:
        dv, v = heapq.heappop(pq)
        if dv != dist.get(v, float("inf")):
            continue
        for u, w in rev_adj_w.get(v, []):
            nd = dv + w
            if nd < dist.get(u, float("inf")):
                dist[u] = nd
                heapq.heappush(pq, (nd, u))
    return dist


def softmax_sample(values: List[float], temperature: float = 1.0) -> int:
    if not values:
        raise ValueError("softmax_sample called with empty list")
    if temperature <= 0:
        max_val = max(values)
        candidates = [i for i, v in enumerate(values) if v == max_val]
        return random.choice(candidates)
    scaled = [v / temperature for v in values]
    m = max(scaled)
    exps = [math.exp(v - m) for v in scaled]
    z = sum(exps)
    probs = [e / z for e in exps]
    r = random.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r <= cum:
            return i
    return len(values) - 1


def build_reverse_adj(network: TrafficNetwork) -> Dict[Node, List[Node]]:
    rev_adj: Dict[Node, List[Node]] = defaultdict(list)
    for (u, v) in network.edges.keys():
        rev_adj[v].append(u)
    return dict(rev_adj)


def nodes_that_can_reach_dest_from_rev(rev_adj: Dict[Node, List[Node]], dest: Node) -> Set[Node]:
    reachable = {dest}
    q = deque([dest])
    while q:
        x = q.popleft()
        for prev in rev_adj.get(x, []):
            if prev not in reachable:
                reachable.add(prev)
                q.append(prev)
    return reachable


def nodes_that_can_reach_dest(network: TrafficNetwork, dest: Node) -> Set[Node]:
    return nodes_that_can_reach_dest_from_rev(build_reverse_adj(network), dest)


def q_learning_train_true_coin_from_ff_expert(
    network: TrafficNetwork,
    expert_steps,
    flows_expert: EdgeFlows,
    expert_edge_ids: Set[EdgeId],
    expert_edge_quality: Optional[Dict[EdgeId, float]] = None,
    num_episodes: int = 20,
    num_agents_per_episode: int = 100,
    od_mode: str = "fixed",
    fixed_od: Optional[Tuple[Node, Node]] = None,
    alpha: float = 0.3,
    gamma: float = 0.95,
    temp_start: float = 1.0,
    temp_end: float = 0.1,
    max_steps_per_agent: int = 1500,
    lambda_dr: float = LAMBDA_DR,
    lambda_prog: float = LAMBDA_PROG,
    lambda_term: float = LAMBDA_TERM,
    lambda_exp: float = LAMBDA_EXP,
    lambda_step: float = LAMBDA_STEP,
    lambda_miss: float = LAMBDA_MISS,
    time_scale: float = TIME_SCALE,
    horizon_s: float = 3600.0,
    debug_rewards: bool = True,
    visited_penalty_train: float = 0.0,
    lambda_time_train: float = 1.0,
    time_scale_s: float = 1000.0,
    t_cap_s: float = 3000.0,
    tau: float = 0.05,
    replay_capacity: int = 10_000,
    replay_batch: int = 64,
    replay_min: int = 256,
    refresh_sp_every: int = 1,
    refresh_expert_every: int = 0,
    num_agents_expert_refresh: int = 100,
) -> Dict[Tuple[Node, Node, Node], float]:
    q_table: Dict[Tuple[Node, Node, Node], float] = {}
    quality = expert_edge_quality if expert_edge_quality is not None else {}

    rev_adj: Dict[Node, List[Node]] = defaultdict(list)
    rev_adj_w: Dict[Node, List[Tuple[Node, float]]] = defaultdict(list)
    for (u, v), edge in network.edges.items():
        rev_adj[v].append(u)
        rev_adj_w[v].append((u, float(edge.free_time)))

    reachable_cache: Dict[Node, Set[Node]] = {}
    sp_cache: Dict[Node, Dict[Node, float]] = {}
    scale_to_vph = 3600.0 / float(horizon_s)

    def reachable_set(dest: Node) -> Set[Node]:
        rs = reachable_cache.get(dest)
        if rs is not None:
            return rs
        seen: Set[Node] = {dest}
        qd = deque([dest])
        while qd:
            v = qd.popleft()
            for u in rev_adj.get(v, []):
                if u not in seen:
                    seen.add(u)
                    qd.append(u)
        reachable_cache[dest] = seen
        return seen

    def sp_dist_map(dest: Node, edge_counts: Optional[EdgeFlows] = None) -> Dict[Node, float]:
        if edge_counts is None:
            m = sp_cache.get(dest)
            if m is not None:
                return m
        dist: Dict[Node, float] = {dest: 0.0}
        pq = [(0.0, dest)]
        while pq:
            dv, v = heapq.heappop(pq)
            if dv > dist.get(v, float("inf")):
                continue
            for u, w_ff in rev_adj_w.get(v, []):
                if edge_counts is not None:
                    edge = network.edges.get((u, v))
                    w = float(edge.travel_time(edge_counts.get((u, v), 0.0) * scale_to_vph)) if edge else w_ff
                else:
                    w = w_ff
                nd = dv + w
                if nd < dist.get(u, float("inf")):
                    dist[u] = nd
                    heapq.heappush(pq, (nd, u))
        if edge_counts is None:
            sp_cache[dest] = dist
        return dist

    def progress_sp(sp: Dict[Node, float], s: Node, a: Node) -> float:
        old_d = sp.get(s, float("inf"))
        new_d = sp.get(a, float("inf"))
        if not np.isfinite(old_d) or old_d <= 0.0:
            return 0.0
        return max(0.0, (old_d - new_d) / max(old_d, 1.0))

    def filter_neighbors_reach_tau(sp: Dict[Node, float], s: Node, neighbors: List[Node]) -> List[Node]:
        old_d = sp.get(s, float("inf"))
        if not np.isfinite(old_d) or old_d <= 0.0:
            return neighbors
        filtered = [nxt for nxt in neighbors if (old_d - sp.get(nxt, float("inf"))) / max(old_d, 1.0) >= -tau]
        return filtered if filtered else neighbors

    def sample_agents_for_mode(n: int) -> List[Agent]:
        if od_mode == "random":
            return sample_agents_random(network, n)
        if od_mode in ("north_south", "north-south"):
            return sample_agents_north_south(network, n)
        if fixed_od is None:
            raise ValueError("fixed_od=(origin, destination) is required when od_mode='fixed'")
        return sample_agents_fixed(network, n, fixed_od[0], fixed_od[1])

    dg_stats = RunningStats()
    for (_, _, _, edge_id) in expert_steps:
        edge = network.edges[edge_id]
        c_before = max(flows_expert.get(edge_id, 0.0) - 1.0, 0.0)
        dg_stats.update(edge_marginal_global_cost(edge, c_before, horizon_s))

    def clip_delta_g(delta_g: float) -> float:
        ceiling = dg_stats.mean + 3.0 * dg_stats.std if dg_stats.n >= 2 else 1e6
        return min(max(delta_g, 0.0), ceiling)

    replay = ReplayBuffer(capacity=replay_capacity)
    expert_steps_mut = list(expert_steps)
    expert_ids = set(expert_edge_ids)
    expert_qual = dict(quality)

    for (s, d, a, edge_id) in expert_steps_mut:
        edge = network.edges[edge_id]
        c_before = max(flows_expert.get(edge_id, 0.0) - 1.0, 0.0)
        delta_g = edge_marginal_global_cost(edge, c_before, horizon_s)
        dg_stats.update(delta_g)
        dr = -lambda_dr * (clip_delta_g(delta_g) / time_scale)
        sp = sp_dist_map(d)
        prog = progress_sp(sp, s, a)
        exp_bon = lambda_exp * expert_qual.get(edge_id, 1.0) if edge_id in expert_ids else 0.0
        reward = dr + lambda_prog * prog + exp_bon - lambda_step
        if a == d:
            reward += lambda_term
        key = (s, d, a)
        old_q = q_table.get(key, 0.0)
        q_table[key] = old_q + alpha * (reward - old_q)

    for ep in range(num_episodes):
        temp = temp_start + (temp_end - temp_start) * ep / max(num_episodes - 1, 1)
        if refresh_expert_every > 0 and ep > 0 and ep % refresh_expert_every == 0:
            new_steps, _, _, new_quality = build_ff_expert_dataset(
                network,
                num_agents=num_agents_expert_refresh,
                od_mode=od_mode,
                horizon_s=horizon_s,
                fixed_od=fixed_od,
            )
            expert_steps_mut = new_steps
            expert_ids = {eid for (_, _, _, eid) in new_steps}
            expert_qual = new_quality

        edge_counts: EdgeFlows = defaultdict(float)
        ep_sp_cache: Dict[Node, Dict[Node, float]] = {}

        def get_ep_sp(dest: Node) -> Dict[Node, float]:
            if dest not in ep_sp_cache:
                if refresh_sp_every > 0 and ep % refresh_sp_every == 0:
                    ep_sp_cache[dest] = sp_dist_map(dest, edge_counts)
                else:
                    ep_sp_cache[dest] = sp_dist_map(dest)
            return ep_sp_cache[dest]

        agents = sample_agents_for_mode(num_agents_per_episode)
        total_reward = 0.0
        total_steps = 0
        dbg = RewardDebug(enabled=debug_rewards)

        for agent in agents:
            s = agent.origin
            d = agent.destination
            visited: Set[Node] = set()
            steps = 0
            rs = reachable_set(d)
            ep_sp_cache.pop(d, None)
            sp = get_ep_sp(d)

            while s != d and steps < max_steps_per_agent:
                neighbors = [nxt for nxt in get_neighbors(network, s) if nxt in rs]
                if not neighbors:
                    break
                neighbors = filter_neighbors_reach_tau(sp, s, neighbors)
                if not neighbors:
                    break
                visited.add(s)
                q_vals = [q_table.get((s, d, nxt), 0.0) for nxt in neighbors]
                a = neighbors[softmax_sample(q_vals, temperature=temp)]
                edge_id = (s, a)
                edge = network.edges.get(edge_id)
                if edge is None:
                    break
                count_before = edge_counts[edge_id]
                delta_g = edge_marginal_global_cost(edge, count_before, horizon_s)
                dg_stats.update(delta_g)
                dr = -lambda_dr * (clip_delta_g(delta_g) / time_scale)
                prog = progress_sp(sp, s, a)
                exp_bon = lambda_exp * expert_qual.get(edge_id, 1.0) if edge_id in expert_ids else 0.0
                time_pen = 0.0
                if lambda_time_train != 0.0:
                    flow_vph = count_before * scale_to_vph
                    t_edge = min(float(edge.travel_time(flow_vph)), float(t_cap_s))
                    time_pen = -lambda_time_train * (t_edge / float(time_scale_s))
                reward = dr + lambda_prog * prog + exp_bon - lambda_step + time_pen
                if visited_penalty_train != 0.0 and a in visited:
                    reward -= visited_penalty_train
                edge_counts[edge_id] += 1.0
                s_next = a
                steps += 1
                total_steps += 1
                done = (s_next == d) or (steps >= max_steps_per_agent)
                if done and s_next != d:
                    reward -= lambda_miss
                elif s_next == d:
                    reward += lambda_term
                if done:
                    target = reward
                else:
                    nn = [n for n in get_neighbors(network, s_next) if n in rs]
                    nn = filter_neighbors_reach_tau(sp, s_next, nn)
                    target = reward + gamma * (max(q_table.get((s_next, d, n), 0.0) for n in nn) if nn else 0.0)
                key = (s, d, a)
                old_q = q_table.get(key, 0.0)
                q_table[key] = old_q + alpha * (target - old_q)
                replay.push(s, d, a, reward, s_next, done)
                dbg.add(total=reward, dr=dr, prog=lambda_prog * prog, exp=exp_bon, step=-lambda_step, time=time_pen, delta_G=delta_g)
                total_reward += reward
                s = s_next

        if len(replay) >= replay_min:
            batch = replay.sample(replay_batch)
            for (rb_s, rb_d, rb_a, rb_r, rb_sn, rb_done) in batch:
                rb_rs = reachable_set(rb_d)
                rb_sp = sp_dist_map(rb_d)
                if rb_done:
                    rb_target = rb_r
                else:
                    rb_nn = [n for n in get_neighbors(network, rb_sn) if n in rb_rs]
                    rb_nn = filter_neighbors_reach_tau(rb_sp, rb_sn, rb_nn)
                    rb_target = rb_r + gamma * (max(q_table.get((rb_sn, rb_d, n), 0.0) for n in rb_nn) if rb_nn else 0.0)
                rb_key = (rb_s, rb_d, rb_a)
                q_table[rb_key] = q_table.get(rb_key, 0.0) + alpha * (rb_target - q_table.get(rb_key, 0.0))

        if debug_rewards:
            avg_r = total_reward / max(total_steps, 1)
            avg_steps = total_steps / max(len(agents), 1)
            print(f"[TRUE-COIN+Expert] Ep {ep + 1}/{num_episodes}: temp={temp:.3f}, avg_r/step={avg_r:.4f}, avg_steps={avg_steps:.1f}")
            dbg.summary(name=f"Episode {ep + 1} reward breakdown")

    return q_table


def route_with_trained_Q_flow_adaptive_softmax(
    network: TrafficNetwork,
    q_table: Dict[Tuple[Node, Node, Node], float],
    num_agents: int = 200,
    od_mode: str = "fixed",
    max_steps_per_agent: int = 1500,
    lambda_live_cost: float = 5.0,
    epsilon_eval: float = 0.0,
    temperature: float = 0.7,
    horizon_s: float = 3600.0,
    visited_penalty_val: float = 3.0,
    fixed_od: Optional[Tuple[Node, Node]] = None,
) -> Tuple[List[Agent], EdgeFlows, float]:
    edge_flows: EdgeFlows = defaultdict(float)
    if od_mode == "random":
        agents = sample_agents_random(network, num_agents)
    elif od_mode in ("north_south", "north-south"):
        agents = sample_agents_north_south(network, num_agents)
    else:
        if fixed_od is None:
            raise ValueError("fixed_od=(origin, destination) is required when od_mode='fixed'")
        agents = sample_agents_fixed(network, num_agents, fixed_od[0], fixed_od[1])

    scale_to_vph = 3600.0 / horizon_s
    rev_adj = build_reverse_adj(network)
    reachable_cache: Dict[Node, Set[Node]] = {}

    def reachable_to_dest(d: Node) -> Set[Node]:
        if d not in reachable_cache:
            reachable_cache[d] = nodes_that_can_reach_dest_from_rev(rev_adj, d)
        return reachable_cache[d]

    for agent in agents:
        s = agent.origin
        d = agent.destination
        path = [s]
        steps = 0
        visited: Set[Node] = set()
        travel_s = 0.0
        reach_set = reachable_to_dest(d)
        if s not in reach_set:
            agent.path = path
            agent.experienced_time_s = 0.0
            continue

        while s != d and steps < max_steps_per_agent:
            neighbors = [nxt for nxt in get_neighbors(network, s) if nxt in reach_set]
            if not neighbors:
                break
            visited.add(s)
            if epsilon_eval > 0.0 and random.random() < epsilon_eval:
                a = random.choice(neighbors)
            else:
                utilities: List[float] = []
                candidate_actions: List[Node] = []
                for nxt in neighbors:
                    edge_id = (s, nxt)
                    edge = network.edges.get(edge_id)
                    if edge is None:
                        continue
                    count_live = edge_flows.get(edge_id, 0.0)
                    flow_live_vph = count_live * scale_to_vph
                    t_free = float(edge.free_time)
                    t_live = float(edge.travel_time(flow_live_vph))
                    congestion_frac = (t_live - t_free) / max(t_free, 1.0)
                    base_q = q_table.get((s, d, nxt), 0.0)
                    vpen = visited_penalty_val if nxt in visited else 0.0
                    utility = base_q - lambda_live_cost * congestion_frac - vpen
                    utilities.append(utility)
                    candidate_actions.append(nxt)
                if not candidate_actions:
                    break
                if len(utilities) > 1:
                    u_min, u_max = min(utilities), max(utilities)
                    u_range = u_max - u_min
                    if u_range > 1e-9:
                        utilities = [(u - u_min) / u_range for u in utilities]
                idx = softmax_sample(utilities, temperature=temperature)
                a = candidate_actions[idx]

            edge_id = (s, a)
            edge = network.edges.get(edge_id)
            if edge is None:
                break
            flow_live_vph = edge_flows.get(edge_id, 0.0) * scale_to_vph
            travel_s += float(edge.travel_time(flow_live_vph))
            edge_flows[edge_id] += 1.0
            s = a
            path.append(s)
            steps += 1

        agent.path = path
        agent.experienced_time_s = travel_s

    g_world = total_system_travel_time(network, edge_flows, horizon_s=horizon_s)
    return agents, edge_flows, g_world


def run_spa_experiment(
    network: TrafficNetwork,
    num_agents: int = 500,
    od_mode: str = "north_south",
    horizon_s: float = 3600.0,
    fixed_od: Optional[Tuple[Node, Node]] = None,
) -> Tuple[List[Agent], EdgeFlows, float]:
    if od_mode == "random":
        agents = sample_agents_random(network, num_agents)
    elif od_mode in ("north_south", "north-south"):
        agents = sample_agents_north_south(network, num_agents)
    elif od_mode == "fixed":
        if fixed_od is None:
            raise ValueError("fixed_od=(origin, destination) is required when od_mode='fixed'")
        agents = sample_agents_fixed(network, num_agents, fixed_od[0], fixed_od[1])
    else:
        raise ValueError(f"Unknown od_mode: {od_mode}")
    agents, edge_counts = spa_route_all(network, agents, horizon_s=horizon_s)
    g_world = total_system_travel_time(network, edge_counts, horizon_s=horizon_s)
    return agents, edge_counts, g_world


def run_ff_experiment(
    network: TrafficNetwork,
    num_agents: int = 500,
    od_mode: str = "fixed",
    horizon_s: float = 3600.0,
    fixed_od: Optional[Tuple[Node, Node]] = None,
) -> Tuple[List[Agent], EdgeFlows, float]:
    if od_mode == "random":
        agents = sample_agents_random(network, num_agents)
    elif od_mode in ("north_south", "north-south"):
        agents = sample_agents_north_south(network, num_agents)
    else:
        if fixed_od is None:
            raise ValueError("fixed_od=(origin, destination) is required when od_mode='fixed'")
        agents = sample_agents_fixed(network, num_agents, fixed_od[0], fixed_od[1])
    agents, edge_counts = ford_fulkerson_route_all(network, agents, horizon_s=horizon_s)
    g_world = total_system_travel_time(network, edge_counts, horizon_s=horizon_s)
    return agents, edge_counts, g_world


def run_ff_expert_coin_rl_experiment(
    network: TrafficNetwork,
    alpha=0.3,
    gamma=0.95,
    temp_start: float = 1.0,
    temp_end: float = 0.1,
    epsilon_eval: float = 0.0,
    temperature: float = 0.7,
    lambda_live_cost: float = 10.0,
    visited_penalty_val: float = 3.0,
    lambda_dr: float = 1.0,
    lambda_prog: float = 0.05,
    lambda_exp: float = 0.4,
    lambda_step: float = 0.01,
    visited_penalty_train: float = 1.0,
    lambda_time_train: float = 1.0,
    time_scale_s: float = 1000.0,
    t_cap_s: float = 3000.0,
    tau: float = 0.05,
    time_scale: float = TIME_SCALE,
    lambda_term: float = 5.0,
    lambda_miss: float = 50.0,
    num_agents_baseline: int = 100,
    num_agents_expert: int = 200,
    num_agents_eval_rl: int = 100,
    od_mode: str = "fixed",
    fixed_od: Optional[Tuple[Node, Node]] = None,
    num_episodes_rl: int = 20,
    steps_per_agent: int = 1500,
    horizon_s: float = 3600.0,
    replay_capacity: int = 10_000,
    replay_batch: int = 64,
    replay_min: int = 256,
    refresh_sp_every: int = 1,
    refresh_expert_every: int = 0,
    num_agents_expert_refresh: int = 100,
    debug: bool = True,
):
    agents_spa, flows_spa, g_spa = run_spa_experiment(
        network, num_agents=num_agents_baseline, od_mode=od_mode, horizon_s=horizon_s, fixed_od=fixed_od
    )
    agents_ff, flows_ff, g_ff = run_ff_experiment(
        network, num_agents=num_agents_baseline, od_mode=od_mode, horizon_s=horizon_s, fixed_od=fixed_od
    )
    expert_steps, _, ff_flows_expert, expert_edge_quality = build_ff_expert_dataset(
        network, num_agents=num_agents_expert, od_mode=od_mode, horizon_s=horizon_s, fixed_od=fixed_od
    )
    expert_edge_ids: Set[EdgeId] = {edge_id for (_, _, _, edge_id) in expert_steps}

    q_ff_rl = q_learning_train_true_coin_from_ff_expert(
        network=network,
        expert_steps=expert_steps,
        flows_expert=ff_flows_expert,
        expert_edge_ids=expert_edge_ids,
        expert_edge_quality=expert_edge_quality,
        num_episodes=num_episodes_rl,
        num_agents_per_episode=num_agents_expert,
        od_mode=od_mode,
        fixed_od=fixed_od,
        alpha=alpha,
        gamma=gamma,
        temp_start=temp_start,
        temp_end=temp_end,
        max_steps_per_agent=steps_per_agent,
        lambda_dr=lambda_dr,
        lambda_prog=lambda_prog,
        lambda_term=lambda_term,
        lambda_exp=lambda_exp,
        lambda_step=lambda_step,
        lambda_miss=lambda_miss,
        time_scale=time_scale,
        horizon_s=horizon_s,
        visited_penalty_train=visited_penalty_train,
        debug_rewards=debug,
        lambda_time_train=lambda_time_train,
        time_scale_s=time_scale_s,
        t_cap_s=t_cap_s,
        tau=tau,
        replay_capacity=replay_capacity,
        replay_batch=replay_batch,
        replay_min=replay_min,
        refresh_sp_every=refresh_sp_every,
        refresh_expert_every=refresh_expert_every,
        num_agents_expert_refresh=num_agents_expert_refresh,
    )
    agents_ff_rl, flows_ff_rl, g_ff_rl = route_with_trained_Q_flow_adaptive_softmax(
        network=network,
        q_table=q_ff_rl,
        num_agents=num_agents_eval_rl,
        od_mode=od_mode,
        fixed_od=fixed_od,
        max_steps_per_agent=steps_per_agent,
        lambda_live_cost=lambda_live_cost,
        epsilon_eval=epsilon_eval,
        temperature=temperature,
        horizon_s=horizon_s,
        visited_penalty_val=visited_penalty_val,
    )
    return (
        agents_spa,
        flows_spa,
        g_spa,
        agents_ff,
        flows_ff,
        g_ff,
        agents_ff_rl,
        flows_ff_rl,
        g_ff_rl,
        q_ff_rl,
    )


def fraction_reached_agentspecific(agents: List[Agent]) -> float:
    reached = sum(1 for a in agents if a.path and a.path[-1] == a.destination)
    return reached / max(len(agents), 1)


def compute_agent_travel_times(network: TrafficNetwork, edge_flows: EdgeFlows, agents: List[Agent], horizon_s: float = 3600.0) -> List[float]:
    times = []
    scale_to_vph = 3600.0 / horizon_s
    for agent in agents:
        if agent.experienced_time_s is not None:
            times.append(float(agent.experienced_time_s))
            continue
        if not agent.path or len(agent.path) < 2:
            times.append(0.0)
            continue
        total = 0.0
        for u, v in edges_from_path(agent.path):
            edge = network.edges[(u, v)]
            count = edge_flows.get((u, v), 0.0)
            flow_vph = count * scale_to_vph
            total += edge.travel_time(flow_vph)
        times.append(total)
    return times


def routes_to_gdf(agents: List[Agent], crs) -> gpd.GeoDataFrame:
    geoms = []
    for agent in agents:
        if agent.path is not None and len(agent.path) > 1:
            geoms.append(LineString(agent.path))
    return gpd.GeoDataFrame(geometry=geoms, crs=crs)


def edge_flows_to_gdf(edge_flows: EdgeFlows, network: TrafficNetwork, crs, horizon_s: float = 3600.0):
    geoms, flows_vph = [], []
    scale_to_vph = 3600.0 / horizon_s
    for (u, v), c in edge_flows.items():
        if c <= 0:
            continue
        geoms.append(LineString([u, v]))
        flows_vph.append(c * scale_to_vph)
    return gpd.GeoDataFrame({"flow_vph": flows_vph}, geometry=geoms, crs=crs)


def total_distance_from_flows(network: TrafficNetwork, flows: EdgeFlows) -> float:
    dist_m = 0.0
    for edge_id, f in flows.items():
        if f <= 0.0:
            continue
        edge = network.edges[edge_id]
        dist_m += edge.length_m * f
    return dist_m


def print_scenario_metrics(name: str, network: TrafficNetwork, flows: EdgeFlows, g_world: float, num_agents: int) -> None:
    avg_time_s = g_world / num_agents
    avg_time_min = avg_time_s / 60.0
    total_dist_m = total_distance_from_flows(network, flows)
    total_dist_km = total_dist_m / 1000.0
    avg_dist_m = total_dist_m / num_agents
    avg_dist_km = avg_dist_m / 1000.0
    print(f"\n[{name}] metrics:")
    print(f"  Average travel time per agent: {avg_time_s:8.2f} s  ({avg_time_min:6.2f} min)")
    print(f"  Total distance traveled:       {total_dist_km:8.2f} km")
    print(f"  Average distance per agent:    {avg_dist_km:8.2f} km")


def plot_routes_three_way(
    gdf: gpd.GeoDataFrame,
    agents_spa: List[Agent],
    g_spa: float,
    agents_coin_ff: List[Agent],
    g_coin_ff: float,
    agents_coin_ff_rl: List[Agent],
    g_coin_ff_rl: float,
    num_agents: int,
    horizon_s: float,
    out_path: Optional[str] = None,
):
    horizon_hr = horizon_s / 3600.0
    routes_spa = routes_to_gdf(agents_spa, crs=gdf.crs)
    routes_ff = routes_to_gdf(agents_coin_ff, crs=gdf.crs)
    routes_ff_rl = routes_to_gdf(agents_coin_ff_rl, crs=gdf.crs)
    fig, axes = plt.subplots(1, 3, figsize=(21, 8))
    fig.suptitle(f"Routes Comparison - {num_agents} trips over {horizon_hr:.1f} hours", fontsize=16, y=0.98)
    gdf.plot(ax=axes[0], linewidth=0.3, color="lightgray")
    routes_spa.plot(ax=axes[0], linewidth=1.2, alpha=0.9)
    axes[0].set_title(f"SPA\nG = {g_spa/3600:.1f} veh*hrs", fontsize=12)
    axes[0].set_axis_off()
    gdf.plot(ax=axes[1], linewidth=0.3, color="lightgray")
    routes_ff.plot(ax=axes[1], linewidth=1.2, alpha=0.9)
    axes[1].set_title(f"FF\nG = {g_coin_ff/3600:.1f} veh*hrs", fontsize=12)
    axes[1].set_axis_off()
    gdf.plot(ax=axes[2], linewidth=0.3, color="lightgray")
    routes_ff_rl.plot(ax=axes[2], linewidth=1.2, alpha=0.9)
    axes[2].set_title(f"FF-RL\nG = {g_coin_ff_rl/3600:.1f} veh*hrs", fontsize=12)
    axes[2].set_axis_off()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_heatmaps_three_way(
    gdf: gpd.GeoDataFrame,
    network: TrafficNetwork,
    flows_spa: EdgeFlows,
    flows_ff: EdgeFlows,
    flows_ff_rl: EdgeFlows,
    num_agents: int,
    horizon_s: float,
    out_path: Optional[str] = None,
):
    horizon_hr = horizon_s / 3600.0

    def flows_to_gdf_counts(flows: EdgeFlows) -> gpd.GeoDataFrame:
        geoms = []
        counts = []
        for (u, v), c in flows.items():
            if c <= 0 or (u, v) not in network.edges:
                continue
            geoms.append(LineString([u, v]))
            counts.append(float(c))
        return gpd.GeoDataFrame({"count": counts}, geometry=geoms, crs=gdf.crs)

    spa_gdf = flows_to_gdf_counts(flows_spa)
    ff_gdf = flows_to_gdf_counts(flows_ff)
    rl_gdf = flows_to_gdf_counts(flows_ff_rl)
    if spa_gdf.empty or ff_gdf.empty or rl_gdf.empty:
        return

    vmax = max(spa_gdf["count"].max(), ff_gdf["count"].max(), rl_gdf["count"].max())
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    fig.suptitle(f"Edge Usage Heatmaps - {num_agents} trips over {horizon_hr:.1f} hours", fontsize=15, y=1.01)
    cmap = "turbo"

    gdf.plot(ax=axes[0], linewidth=0.2, color="lightgray")
    spa_gdf.plot(ax=axes[0], column="count", linewidth=1.5, alpha=0.9, cmap=cmap, vmin=0, vmax=vmax, legend=True)
    axes[0].set_title("SPA Edge Counts")
    axes[0].set_axis_off()

    gdf.plot(ax=axes[1], linewidth=0.2, color="lightgray")
    ff_gdf.plot(ax=axes[1], column="count", linewidth=1.5, alpha=0.9, cmap=cmap, vmin=0, vmax=vmax, legend=True)
    axes[1].set_title("FF Edge Counts")
    axes[1].set_axis_off()

    gdf.plot(ax=axes[2], linewidth=0.2, color="lightgray")
    rl_gdf.plot(ax=axes[2], column="count", linewidth=1.5, alpha=0.9, cmap=cmap, vmin=0, vmax=vmax, legend=True)
    axes[2].set_title("FF-RL Edge Counts")
    axes[2].set_axis_off()

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def grid_search_coin(
    network: TrafficNetwork,
    param_grid: dict,
    *,
    seed: int = 26,
    num_agents: int = 1000,
    num_agents_expert: Optional[int] = None,
    horizon_s: float = 3600.0,
    od_mode: str = "north_south",
    fixed_od: Optional[Tuple[Node, Node]] = None,
    steps_per_agent: int = 1500,
    min_reach: float = 0.90,
    debug: bool = False,
    verbose_every: int = 1,
):
    if num_agents_expert is None:
        num_agents_expert = num_agents // 3
    keys = list(param_grid.keys())
    values_lists = [param_grid[k] for k in keys]
    combos = list(itertools.product(*values_lists))
    rows = []
    feasible_rows = []
    for i, vals in enumerate(combos, start=1):
        params = dict(zip(keys, vals))
        random.seed(seed)
        np.random.seed(seed)
        t0 = time.time()
        try:
            (
                agents_spa,
                flows_spa,
                g_spa,
                agents_ff,
                flows_ff,
                g_ff,
                agents_ff_rl,
                flows_ff_rl,
                g_ff_rl,
                _,
            ) = run_ff_expert_coin_rl_experiment(
                network,
                num_agents_baseline=num_agents,
                num_agents_expert=num_agents_expert,
                num_agents_eval_rl=num_agents,
                od_mode=od_mode,
                fixed_od=fixed_od,
                steps_per_agent=steps_per_agent,
                horizon_s=horizon_s,
                **params,
                debug=debug,
            )
            reach_spa = fraction_reached_agentspecific(agents_spa)
            reach_ff = fraction_reached_agentspecific(agents_ff)
            reach_rl = fraction_reached_agentspecific(agents_ff_rl)
            elapsed = time.time() - t0
            row = {
                "trial": i,
                "elapsed_s": elapsed,
                "reach_spa": reach_spa,
                "reach_ff": reach_ff,
                "reach_ff_rl": reach_rl,
                "G_spa_vehhrs": g_spa / 3600.0,
                "G_ff_vehhrs": g_ff / 3600.0,
                "G_ff_rl_vehhrs": g_ff_rl / 3600.0,
                "feasible": (reach_rl >= min_reach),
            }
            row.update(params)
            rows.append(row)
            if row["feasible"]:
                feasible_rows.append(row)
            if verbose_every and (i % verbose_every == 0):
                print(f"[{i}/{len(combos)}] reach_rl={reach_rl:.3f} G_rl={row['G_ff_rl_vehhrs']:.1f} feasible={row['feasible']}")
        except Exception as exc:
            elapsed = time.time() - t0
            row = {"trial": i, "elapsed_s": elapsed, "error": repr(exc), "feasible": False}
            row.update(params)
            rows.append(row)

    results_df = pd.DataFrame(rows)
    best_row = None
    if feasible_rows:
        feasible_df = pd.DataFrame(feasible_rows)
        best_idx = feasible_df["G_ff_rl_vehhrs"].idxmin()
        best_row = feasible_df.loc[best_idx]
    return results_df, best_row


def animate_routes_three_way_light(
    gdf: gpd.GeoDataFrame,
    agents_spa: List[Agent],
    agents_ff: List[Agent],
    agents_ff_rl: List[Agent],
    num_agents: int,
    out_path: Optional[str] = None,
    max_frames: int = 80,
    fps: int = 8,
    figsize=(10, 6),
    dpi: int = 80,
):
    if out_path is None:
        out_path = f"routes_three_way_{num_agents}_agents.mp4"

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    titles = ["SPA routes", "FF routes", "FF-RL routes"]
    for ax, title in zip(axes, titles):
        gdf.plot(ax=ax, color="lightgrey", linewidth=0.3)
        ax.set_axis_off()
        ax.set_title(title, fontsize=10)
    fig.suptitle(f"Animation for {num_agents} agents", fontsize=12, y=0.97)

    scenarios_agents = [agents_spa, agents_ff, agents_ff_rl]
    scenarios_paths_xy = []
    global_max_len = 0
    for agents in scenarios_agents:
        paths_xy = []
        for a in agents:
            if not getattr(a, "path", None) or len(a.path) < 2:
                continue
            xs, ys = zip(*a.path)
            paths_xy.append((xs, ys))
            global_max_len = max(global_max_len, len(xs))
        scenarios_paths_xy.append(paths_xy)

    if global_max_len == 0:
        plt.close(fig)
        return

    n_frames = min(max_frames, global_max_len)
    scenarios_indices_per_frame = []
    for paths_xy in scenarios_paths_xy:
        idx_lists = []
        for xs, _ in paths_xy:
            l = len(xs)
            if l <= 1:
                idx_lists.append([0] * n_frames)
                continue
            idxs = [min(int((k / (n_frames - 1)) * (l - 1)), l - 1) for k in range(n_frames)]
            idx_lists.append(idxs)
        scenarios_indices_per_frame.append(idx_lists)

    scenarios_lines = []
    for ax, paths_xy in zip(axes, scenarios_paths_xy):
        scenario_lines = []
        for _xs, _ys in paths_xy:
            line, = ax.plot([], [], linewidth=1.5)
            scenario_lines.append(line)
        scenarios_lines.append(scenario_lines)

    def init():
        for scenario_lines in scenarios_lines:
            for line in scenario_lines:
                line.set_data([], [])
        return [line for scenario_lines in scenarios_lines for line in scenario_lines]

    def update(frame):
        for paths_xy, idx_lists, lines in zip(scenarios_paths_xy, scenarios_indices_per_frame, scenarios_lines):
            for (xs, ys), idxs, line in zip(paths_xy, idx_lists, lines):
                idx = idxs[frame]
                line.set_data(xs[: idx + 1], ys[: idx + 1])
        return [line for scenario_lines in scenarios_lines for line in scenario_lines]

    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=True, interval=1000 // max(fps, 1))
    out_path = str(out_path)
    if out_path.lower().endswith(".gif"):
        writer = PillowWriter(fps=fps)
        anim.save(out_path, writer=writer, dpi=dpi)
    else:
        writer = FFMpegWriter(fps=fps, bitrate=1200)
        anim.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)


class TrafficOptimizationExperiment:
    """
    Class wrapper around the notebook methods, plus one high-level method:
    `run_and_save(...)`.
    """

    def __init__(
        self,
        shapefile_path: str | Path = "../trimmed_manhattan_shape/trimmed_manhattan.shp",
        seed: int = 26,
    ):
        self.shapefile_path = Path(shapefile_path)
        self.seed = int(seed)
        self.gdf: Optional[gpd.GeoDataFrame] = None
        self.network: Optional[TrafficNetwork] = None
        self.origin: Optional[Node] = None
        self.destination: Optional[Node] = None

    def load_data(self) -> gpd.GeoDataFrame:
        random.seed(self.seed)
        np.random.seed(self.seed)
        gdf = gpd.read_file(str(self.shapefile_path))
        if gdf.crs is not None and gdf.crs.is_geographic:
            gdf = gdf.to_crs(epsg=2263)
        self.gdf = gdf
        return gdf

    def build_network(self) -> TrafficNetwork:
        if self.gdf is None:
            self.load_data()
        assert self.gdf is not None
        network, _ = network_from_streets_gdf(self.gdf)
        self.network = network
        return network

    def pick_fixed_od(self) -> Tuple[Node, Node]:
        if self.network is None:
            self.build_network()
        assert self.network is not None
        self.origin, self.destination = pick_north_south_pair(self.network)
        return self.origin, self.destination

    def run_and_save(
        self,
        output_dir: str | Path = "outputs",
        num_agents: int = 1000,
        num_agents_expert: Optional[int] = None,
        num_episodes_rl: int = 20,
        horizon_s: float = 3600.0,
        od_mode: str = "fixed",
        steps_per_agent: int = 1000,
        alpha: float = 0.12,
        gamma: float = 0.97,
        temp_start: float = 2.0,
        temp_end: float = 0.05,
        temperature: float = 1.0,
        lambda_live_cost: float = 30.0,
        visited_penalty_val: float = 25.0,
        tau: float = 0.03,
        visited_penalty_train: float = 0.02,
        lambda_time_train: float = 1.0,
        time_scale_s: float = 400.0,
        t_cap_s: float = 1200.0,
        lambda_dr: float = 10.0,
        lambda_prog: float = 20.0,
        lambda_exp: float = 0.30,
        lambda_step: float = 0.0,
        time_scale: float = 500.0,
        lambda_term: float = 2500.0,
        lambda_miss: float = 500.0,
        animation_frames: int = 80,
        animation_fps: int = 8,
        save_gif: bool = True,
        debug: bool = True,
    ) -> Dict[str, Any]:
        random.seed(self.seed)
        np.random.seed(self.seed)
        if num_agents_expert is None:
            num_agents_expert = max(num_agents // 3, 33)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if self.gdf is None:
            self.load_data()
        if self.network is None:
            self.build_network()
        if od_mode == "fixed" and (self.origin is None or self.destination is None):
            self.pick_fixed_od()

        assert self.gdf is not None
        assert self.network is not None
        fixed_od = (self.origin, self.destination) if od_mode == "fixed" else None

        (
            agents_spa,
            flows_spa,
            g_spa,
            agents_ff,
            flows_ff,
            g_ff,
            agents_ff_rl,
            flows_ff_rl,
            g_ff_rl,
            q_ff_rl,
        ) = run_ff_expert_coin_rl_experiment(
            self.network,
            alpha=alpha,
            gamma=gamma,
            temp_start=temp_start,
            temp_end=temp_end,
            epsilon_eval=0.0,
            temperature=temperature,
            lambda_live_cost=lambda_live_cost,
            visited_penalty_val=visited_penalty_val,
            lambda_dr=lambda_dr,
            lambda_prog=lambda_prog,
            lambda_exp=lambda_exp,
            lambda_step=lambda_step,
            visited_penalty_train=visited_penalty_train,
            lambda_time_train=lambda_time_train,
            time_scale_s=time_scale_s,
            t_cap_s=t_cap_s,
            tau=tau,
            time_scale=time_scale,
            lambda_term=lambda_term,
            lambda_miss=lambda_miss,
            num_agents_baseline=num_agents,
            num_agents_expert=num_agents_expert,
            num_agents_eval_rl=num_agents,
            od_mode=od_mode,
            fixed_od=fixed_od,
            num_episodes_rl=num_episodes_rl,
            steps_per_agent=steps_per_agent,
            horizon_s=horizon_s,
            debug=debug,
        )

        routes_png = out_dir / "routes_three_way.png"
        heatmap_png = out_dir / "heatmaps_three_way.png"
        anim_mp4 = out_dir / "routes_three_way.mp4"
        anim_gif = out_dir / "routes_three_way.gif"

        plot_routes_three_way(
            self.gdf,
            agents_spa,
            g_spa,
            agents_ff,
            g_ff,
            agents_ff_rl,
            g_ff_rl,
            num_agents=num_agents,
            horizon_s=horizon_s,
            out_path=str(routes_png),
        )
        plot_heatmaps_three_way(
            self.gdf,
            self.network,
            flows_spa,
            flows_ff,
            flows_ff_rl,
            num_agents=num_agents,
            horizon_s=horizon_s,
            out_path=str(heatmap_png),
        )
        animate_routes_three_way_light(
            self.gdf,
            agents_spa,
            agents_ff,
            agents_ff_rl,
            num_agents=num_agents,
            out_path=str(anim_mp4),
            max_frames=animation_frames,
            fps=animation_fps,
        )
        if save_gif:
            animate_routes_three_way_light(
                self.gdf,
                agents_spa,
                agents_ff,
                agents_ff_rl,
                num_agents=num_agents,
                out_path=str(anim_gif),
                max_frames=animation_frames,
                fps=animation_fps,
            )

        results = {
            "SPA": (agents_spa, flows_spa, g_spa),
            "FF": (agents_ff, flows_ff, g_ff),
            "FF-RL": (agents_ff_rl, flows_ff_rl, g_ff_rl),
        }
        return {
            "results": results,
            "q_table": q_ff_rl,
            "origin": self.origin,
            "destination": self.destination,
            "routes_png": str(routes_png),
            "heatmap_png": str(heatmap_png),
            "animation_mp4": str(anim_mp4),
            "animation_gif": str(anim_gif) if save_gif else None,
        }

