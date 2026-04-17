# Traffic Optimization with COIN + Expert Imitation (FF-RL)

## What this project does

This project tackles **urban traffic assignment**: given a set of vehicles with origin-destination pairs on the Manhattan street network, find routes that minimize total system travel time (social optimum) — not just individual travel time.

Three algorithms are compared:
| Algorithm | Short name | Idea |
|-----------|-----------|------|
| Shortest Path Assignment | **SPA** | Each agent greedily routes on the current fastest path (Dijkstra). Selfish; causes Nash equilibrium with congestion. |
| Ford-Fulkerson COIN | **FF** | Routes agents using *marginal system cost* (how much each vehicle increases total delay for everyone). Approximately socially optimal. |
| Reinforcement Learning COIN | **FF-RL** | A Q-learning agent trained to imitate FF behavior and minimize system cost via difference rewards. |

---

## The Network

- **Graph**: Manhattan street network loaded from a shapefile (`trimmed_manhattan_shape/`).
- **Nodes**: Intersections represented as `(longitude, latitude)` tuples.
- **Edges**: Street segments with attributes:
  - `length_m` — physical length in meters
  - `v_free` — free-flow speed (m/s)
  - `capacity` — vehicles per hour (vph)
  - `free_time = length_m / v_free` — free-flow travel time (seconds)
- **Travel time (BPR function)**:
  ```
  T(flow) = t_free × (1 + α × (flow / capacity)^β)
  ```
  Standard Bureau of Public Roads formula. As flow approaches capacity, travel time grows sharply.

---

## World Utility

The optimization objective is **total system travel time**:
```
G = Σ_edge  count_e × T_e(flow_e)
```
where `count_e` is vehicles on edge `e` and `T_e` is BPR travel time. Lower G = better.

Results are reported in **vehicle-hours** (`G / 3600`).

---

## Algorithm 1 — SPA (Shortest Path Assignment)

1. Agents route one by one.
2. Each agent runs **Dijkstra** on the current flow-weighted graph.
3. After routing, the edge flows are updated.
4. Result: sequential greedy assignment. Fast but produces high congestion (no coordination).

---

## Algorithm 2 — FF (Ford-Fulkerson marginal cost)

1. Agents route one by one using **Dijkstra with marginal cost** as edge weight:
   ```
   marginal_cost(e) = G_e(count + 1) - G_e(count)
   ```
   This is how much one more vehicle on edge `e` *increases total system delay*.
2. A soft storage penalty is added to penalize over-capacity edges.
3. Result: near-socially-optimal assignment. Used as the **expert** for RL training.

---

## Algorithm 3 — FF-RL (TRUE COIN + Expert Imitation)

This is the core research contribution. It combines two ideas:

### 3a. COIN (Collective Intelligence)

From Wolpert & Tumer (2002): instead of a global reward that each agent can't control, give each agent a **difference reward**:
```
DR_i = G(joint action) - G(joint action without agent i's contribution)
```
In our case, the difference reward for traversing edge `e` is:
```
DR(e) = -λ_dr × ΔG(e)  =  -λ_dr × [G_e(count+1) - G_e(count)] / time_scale
```
This is always ≤ 0: a penalty proportional to how much system cost the agent adds. Agents are incentivized to use uncongested paths.

### 3b. Expert Imitation (from FF)

The FF algorithm is run once to collect **expert trajectories**: `(current_node, destination, next_node, edge_id)` tuples. These are used to:
- **Initialize Q-values** before online training (warm start).
- **Shape rewards** during training: edges taken by the FF expert receive a bonus `λ_exp × quality_score`.

Quality scores are computed per expert edge: lower marginal cost at expert execution time → higher quality score (closer to 1.0). This makes the bonus proportional to how good the expert's choice actually was, not just binary.

### 3c. Q-Learning Setup

**State**: `(current_node, destination)` — where am I and where am I going.  
**Action**: `next_node` — which neighbor to move to.  
**Q-table**: `Q[(s, d, a)]` — expected return from state `(s, d)` taking action `a`.

**Action filtering** (two layers):
1. **Reachability filter**: only allow moves to nodes that can still reach the destination (computed via reverse-graph BFS, cached per destination).
2. **Tau filter**: only allow moves within `τ` of the current shortest-path distance. Prevents large detours.

**Full reward at each step**:
```
R = DR                          # difference reward (≤ 0, COIN signal)
  + λ_prog × progress           # reward for moving closer to destination
  + λ_exp × quality_score       # expert edge bonus (0 if not expert edge)
  - λ_step                      # per-step cost (path length regularization)
  - λ_time × (t_BPR / scale)    # time penalty proportional to congested travel time
  [+ λ_term   if arrived]       # large bonus for reaching destination
  [- λ_miss   if max_steps hit] # penalty for failing to reach
```

**Q-update (standard Bellman)**:
```
Q(s,d,a) ← Q(s,d,a) + α × [R + γ × max_a' Q(s',d,a') - Q(s,d,a)]
```

---

## Training Pipeline (`run_ff_expert_coin_rl_experiment`)

```
1. Run SPA            → baseline metrics
2. Run FF             → baseline metrics + expert dataset
3. Build expert dataset
   └─ FF routes N agents
   └─ Record (s, d, a, edge_id) tuples
   └─ Compute quality score per edge (marginal cost → [0,1])
4. Train Q-learning
   Phase 0: offline init
   └─ For each expert step: compute reward, initialize Q[s,d,a]
   Phase 1: online episodes (×num_episodes)
   └─ Sample agents
   └─ Route with Boltzmann policy (temperature-annealed softmax)
   └─ Update Q online
   └─ Push transitions to replay buffer
   └─ Replay mini-batch updates after each episode
5. Evaluate trained Q
   └─ Route agents with softmax policy + live congestion penalty
   └─ Compute G, reachability, travel times
```

---

## Key Design Decisions

### Boltzmann (Softmax) Exploration
Replaces epsilon-greedy. Action probabilities follow:
```
P(a) ∝ exp(Q(s,d,a) / temperature)
```
Temperature anneals from `temp_start` (exploratory) to `temp_end` (near-greedy). Consistent with the evaluation policy.

### Adaptive delta_G Clipping
`RunningStats` tracks the online mean and std of delta_G values. Instead of a hard cap (the old `1e6`), delta_G is clipped at `mean + 3×std`. This suppresses outliers adaptively as the distribution is learned.

> **Why not z-score normalization?** Z-scoring would make below-average delta_G give a *positive* DR, rewarding agents for staying near uncongested areas (like the origin). That causes agents to loop near the start instead of routing to the destination. DR must stay ≤ 0.

### Experience Replay
Transitions `(s, d, a, reward, s_next, done)` are stored in a fixed-capacity buffer. After each episode, a random mini-batch is sampled for additional Q-updates. This breaks temporal correlation and reuses past experience.

### Congestion-Aware Progress Reward
The progress reward uses shortest-path distance to destination. Per-episode, this is recomputed using live BPR travel times (from accumulated `edge_counts`) rather than free-flow times. Agents get credit for progress toward the destination as it is congested in the current episode, not as it was at free-flow.

### Quality-Weighted Expert Bonus
Rather than a binary `1 if expert edge else 0`, the expert bonus is `λ_exp × quality_score` where `quality_score ∈ [0,1]` is derived from the expert edge's marginal cost at FF execution time. High-quality expert edges (low marginal cost = good system choice) get the full bonus; lower-quality edges get a proportionally reduced bonus.

---

## Files

| File | Description |
|------|-------------|
| `coin/traffic_opt_base_case.ipynb` | Main notebook: SPA vs FF vs FF-RL on fixed OD pair |
| `coin/traffic_opt_coin_100/1000/3000.ipynb` | Scaled runs with 100/1000/3000 agents |
| `coin/traffic_opt_complete.ipynb` | Complete experiment suite |
| `trimmed_manhattan_shape/` | Street network shapefile |
| `past_intents/` | Earlier versions of the algorithm |

---

## Metrics

| Metric | Description |
|--------|-------------|
| `fraction_reached` | Fraction of agents that reached their destination |
| `G` (veh-hours) | Total system travel time — the main optimization target |
| `avg_travel_time` | Mean per-agent BPR travel time |
| `total_distance_km` | Total km traveled across all agents |

A good FF-RL run should satisfy:
- `fraction_reached ≥ 0.90`
- `G_ff_rl ≤ G_ff` (RL matches or beats the FF expert)
- `G_ff_rl ≤ G_spa` (both beat the selfish baseline)

---

## Hyperparameter Reference

| Parameter | Role | Typical value |
|-----------|------|---------------|
| `alpha` | Q-learning rate | 0.12 |
| `gamma` | Discount factor | 0.97 |
| `temp_start / temp_end` | Boltzmann temperature schedule | 2.0 → 0.05 |
| `lambda_dr` | Weight on difference reward | 10.0 |
| `lambda_prog` | Weight on progress reward | 20.0 |
| `lambda_exp` | Max expert bonus (scaled by quality) | 0.30 |
| `lambda_term` | Terminal bonus for reaching destination | 2500 |
| `lambda_miss` | Penalty for not reaching destination | 500 |
| `lambda_time_train` | Weight on per-step BPR time penalty | 1.0 |
| `tau` | Max allowed detour fraction | 0.03 |
| `num_episodes` | Online training episodes | 20–40 |
| `refresh_sp_every` | Recalculate SP with live congestion every K episodes | 1 |
