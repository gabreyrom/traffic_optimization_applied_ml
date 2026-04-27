from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


Node = Tuple[float, float]


@dataclass
class NetworkConfig:
    shapefile_path: str = "../trimmed_manhattan_shape/trimmed_manhattan.shp"
    seed: int = 26
    od_mode: str = "fixed"
    fixed_od: Optional[Tuple[Node, Node]] = None


@dataclass
class ExperimentConfig:
    num_agents: int = 1000
    num_agents_expert: Optional[int] = None
    num_episodes_rl: int = 20
    horizon_s: float = 3600.0
    steps_per_agent: int = 1000
    alpha: float = 0.12
    gamma: float = 0.97
    temp_start: float = 2.0
    temp_end: float = 0.05
    temperature: float = 1.0
    lambda_live_cost: float = 30.0
    visited_penalty_val: float = 25.0
    tau: float = 0.03
    visited_penalty_train: float = 0.02
    lambda_time_train: float = 1.0
    time_scale_s: float = 400.0
    t_cap_s: float = 1200.0
    lambda_dr: float = 10.0
    lambda_prog: float = 20.0
    lambda_exp: float = 0.30
    lambda_step: float = 0.0
    time_scale: float = 500.0
    lambda_term: float = 2500.0
    lambda_miss: float = 500.0
    early_stop_enabled: bool = True
    early_stop_patience: int = 6
    early_stop_min_delta: float = 1e-4
    early_stop_min_episodes: int = 5
    early_stop_target_reach: Optional[float] = None
    early_stop_low_reach_threshold: float = 0.01
    early_stop_stagnation_window: int = 4
    debug: bool = True


@dataclass
class ArtifactConfig:
    output_dir: str = "outputs"
    save_routes_plot: bool = True
    save_heatmap_plot: bool = True
    save_animation_mp4: bool = True
    save_animation_gif: bool = False
    animation_frames: int = 80
    animation_fps: int = 8


@dataclass
class LoggingConfig:
    level: str = "INFO"
    logger_name: str = "traffic_opt"


@dataclass
class AppConfig:
    network: NetworkConfig = field(default_factory=NetworkConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _to_fixed_od(value: Any) -> Optional[Tuple[Node, Node]]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("fixed_od must be [[ox, oy], [dx, dy]]")
    o, d = value
    if not isinstance(o, (list, tuple)) or not isinstance(d, (list, tuple)) or len(o) != 2 or len(d) != 2:
        raise ValueError("fixed_od must be [[ox, oy], [dx, dy]]")
    return (float(o[0]), float(o[1])), (float(d[0]), float(d[1]))


def app_config_from_dict(data: Dict[str, Any]) -> AppConfig:
    n = data.get("network", {}) or {}
    e = data.get("experiment", {}) or {}
    a = data.get("artifacts", {}) or {}
    l = data.get("logging", {}) or {}

    network = NetworkConfig(
        shapefile_path=n.get("shapefile_path", NetworkConfig.shapefile_path),
        seed=int(n.get("seed", NetworkConfig.seed)),
        od_mode=n.get("od_mode", NetworkConfig.od_mode),
        fixed_od=_to_fixed_od(n.get("fixed_od", None)),
    )
    experiment = ExperimentConfig(**{**ExperimentConfig().__dict__, **e})
    artifacts = ArtifactConfig(**{**ArtifactConfig().__dict__, **a})
    logging_cfg = LoggingConfig(**{**LoggingConfig().__dict__, **l})
    return AppConfig(network=network, experiment=experiment, artifacts=artifacts, logging=logging_cfg)


def load_config_yaml(path: str | Path) -> AppConfig:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load config.yaml (`pip install pyyaml`).") from exc

    p = Path(path)
    raw = yaml.safe_load(p.read_text()) if p.exists() else {}
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("config.yaml root must be a mapping/object")
    return app_config_from_dict(raw)
