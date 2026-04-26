from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from traffic_opt_experiment import Agent, EdgeFlows, Node


@dataclass
class ScenarioResult:
    agents: list[Agent]
    flows: EdgeFlows
    world_utility_s: float
    reached_fraction: float


@dataclass
class ExperimentRunResult:
    spa: ScenarioResult
    ff: ScenarioResult
    ff_rl: ScenarioResult
    q_table: Dict[Tuple[Node, Node, Node], float]
    origin: Optional[Node]
    destination: Optional[Node]
    duration_s: float
    metadata: Dict[str, Any]


@dataclass
class ArtifactPaths:
    routes_plot: Optional[str] = None
    heatmap_plot: Optional[str] = None
    animation_mp4: Optional[str] = None
    animation_gif: Optional[str] = None


@dataclass
class PipelineOutput:
    experiment: ExperimentRunResult
    artifacts: ArtifactPaths

