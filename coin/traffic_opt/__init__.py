from .config import AppConfig, load_config_yaml

__all__ = [
    "AppConfig",
    "load_config_yaml",
    "TrafficOptimizationPipeline",
]


def __getattr__(name):
    if name == "TrafficOptimizationPipeline":
        from .pipeline import TrafficOptimizationPipeline

        return TrafficOptimizationPipeline
    raise AttributeError(name)
